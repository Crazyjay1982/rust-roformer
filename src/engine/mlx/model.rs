//! Mel-Band RoFormer model graph on MLX tensors.
//!
//! This is the architecture, written out in Rust: band split, alternating
//! time/freq transformer blocks, per-band mask estimator, mask application.
//! There is no exported graph here to execute: the shapes, the order of the
//! two attention axes and every numerical convention below are this file's own
//! code, which is why the arm runs the checkpoint's native window with no graph
//! surgery.
//!
//! ## What the checkpoint fixes
//!
//! `dim = 384`, `depth = 6`, one layer per time/freq transformer, `heads = 8`,
//! `dim_head = 64`, `num_bands = 60`, stereo input, `num_stems = 1`,
//! `mask_estimator_depth = 2`, `n_fft = 2048` / `hop = 441` / `win = 2048`.
//! All of it is a constant below, and `weights.rs` refuses a file whose tensor
//! count disagrees.
//!
//! ## The numerical conventions, and where each one is pinned
//!
//! * [`L2Norm`] is `x / max(||x||, 1e-12) * sqrt(dim) * weight`. Pinned by
//!   `l2norm_keeps_the_sqrt_dim_factor` and `l2norm_device_path_matches_the_formula`:
//!   the first against the arithmetic a "divide by the norm" fast path would
//!   give, the second against the host reference on real rows.
//! * [`gelu`] is the exact erf form, `0.5x(1 + erf(x/√2))`, not the tanh
//!   approximation. Pinned by `gelu_is_the_exact_erf_form` (host reference) and
//!   `gelu_device_path_matches_erf` (the op the model runs).
//! * RoPE is `fast::rope(traditional = true, base = 10000, dims = 64)`, which is
//!   the `repeat_interleave` + pairwise `rotate_half` semantics the checkpoint was
//!   trained with. Only a forward against the trained model's own reference
//!   output can test this, so it is covered by the parity arm in `separate.rs`
//!   and by nothing else here.
//! * Attention is a manual matmul + softmax with the `1/sqrt(dim_head)` scale
//!   applied *before* the softmax max subtraction. The fused variant is not used
//!   because it diverges in practice; again only the parity arm can show that.
//! * Each transformer block ends with an L2Norm output norm, so
//!   `layers_{i}.{time,freq}_transformer.norm.weight` is consumed. There is no
//!   model-level final norm.
//! * The mask estimator MLP has `depth + 1` Linears (`mask_mlp_shapes`), Tanh
//!   *between* them, then a GLU on the last pair of halves.
//!
//! ## Layout
//!
//! The 60 frequency bands come from a **mel filterbank** (Slaney scale,
//! unnormalized) and **overlap**: a frequency bin can belong to one or two bands.
//! Band features are therefore *gathered* by index and the resulting masks are
//! *scatter-added* back and divided by how many bands covered each bin. The
//! tensor library has no scatter-add, so the scatter is a fixed one-hot matmul:
//! `M (2050, G)` with `M[slot, g] = 1` iff gathered slot `g` maps to full slot
//! `slot = 2f + ch`.
//!
//! Feature layout, per band, freqs ascending, stereo complex:
//!
//! ```text
//! j = f * 4 + ch * 2 + ri          ([`feature_index`])
//! ```
//!
//! i.e. for each frequency bin the four features are `ch0.re, ch0.im, ch1.re,
//! ch1.im`. The same layout holds on the way out of the mask MLPs, before the
//! `(slot, re/im)` regrouping.
//!
//! Weights are stored **transposed** `[in, out]` at load time (see `weights.rs`)
//! so every Linear is a plain `matmul(x, w) + b`.

use mlx_rs::complex64;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    add, concatenate_axis, erf, expand_dims_axes, matmul, max_axis, maximum, multiply, sigmoid,
    softmax_axis, sqrt, sum_axis, tanh,
};
use mlx_rs::{fast, Array};

use super::stft;

// ──────────────────────────── constants ────────────────────────────

/// Model width.
pub const DIM: usize = 384;
/// Number of blocks, each running a time transformer and a freq transformer.
pub const DEPTH: usize = 6;
/// Attention heads.
pub const HEADS: usize = 8;
/// Width of one head: the RoPE `dims` argument.
pub const DIM_HEAD: usize = 64;
/// Mel bands, i.e. the band split's and the mask estimator's per-axis count.
pub const NUM_BANDS: usize = 60;
/// Layers in one mask-estimator MLP: `mask_estimator_depth + 1` (torch
/// semantics (a depth of 2 means three Linears).
pub const MASK_MLP_LAYERS: usize = 3;
/// Width of the gathered q/k/v projection: `heads · dim_head`.
pub const DIM_INNER: usize = HEADS * DIM_HEAD; // 512
/// Channels the model takes and gives: stereo, always.
pub const CHANNELS: usize = 2;
/// Real/imaginary parts per complex bin.
pub const COMPLEX_PARTS: usize = 2;
/// ReLU-free hidden width of the mask MLPs and the feed-forward: `4 · dim`.
pub const FF_MULT: usize = 4;
/// Floor on the L2 norm, applied as a `max` on the norm itself.
const L2_EPS: f32 = 1e-12;

// ─────────────────────── mel filterbank (pure) ───────────────────────

const F_SP: f64 = 200.0 / 3.0; // Slaney mel linear-step
const MIN_LOG_MEL: f64 = 1000.0 / F_SP;

fn logstep() -> f64 {
    6.4f64.ln() / 27.0
}

/// Slaney `hz2mel`: linear below 1 kHz, logarithmic above.
fn hz_to_mel(freq: f64) -> f64 {
    if freq >= 1000.0 {
        MIN_LOG_MEL + (freq / 1000.0).ln() / logstep()
    } else {
        freq / F_SP
    }
}

/// Inverse of [`hz_to_mel`].
fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MIN_LOG_MEL {
        1000.0 * (logstep() * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// Which frequency bins each mel band owns, and the index arithmetic that
/// follows from it. Pure: no tensor runtime, no weights.
pub struct BandLayout {
    /// Ascending freq-bin indices per band.
    pub band_freqs: Vec<Vec<usize>>,
    /// `band_freqs[b].len()`.
    pub num_freqs_per_band: Vec<usize>,
    /// Stereo-expanded gathered-row indices, band-major: for each band, for
    /// each of its freqs `f`, the pair `(2f, 2f+1)`. Length `G = 2·Σnf`.
    /// Value = full slot index into the `(2·n_freqs)` rows, `slot = 2f + ch`.
    pub freq_indices_stereo: Vec<i32>,
    /// How many bands cover each freq bin (one entry per bin, each 1 or 2).
    pub num_bands_per_freq: Vec<i32>,
}

/// Feature index of frequency bin `f`, channel `ch`, part `ri` in the
/// interleaved layout this model was trained with: `f·4 + ch·2 + ri`.
///
/// `CHANNELS` and [`COMPLEX_PARTS`] are both 2, so the row stride is 4; the
/// function is written from the stride rather than from the literal so the
/// layout is stated once.
pub fn feature_index(f: usize, ch: usize, ri: usize) -> usize {
    assert!(ch < CHANNELS, "channel {ch} out of range");
    assert!(ri < COMPLEX_PARTS, "part {ri} out of range");
    f * (CHANNELS * COMPLEX_PARTS) + ch * COMPLEX_PARTS + ri
}

/// Row index of `(f, ch)` in the slot-major view used by the band gather:
/// `2f + ch`. A slot occupies features `2·slot` and `2·slot + 1`, which is the
/// same position [`feature_index`] gives for `(f, ch, ri)`.
pub fn slot_index(f: usize, ch: usize) -> usize {
    f * CHANNELS + ch
}

/// Width of one band's feature vector: its bins, times channels, times re/im.
pub fn band_dim_in(num_freqs: usize) -> usize {
    num_freqs * CHANNELS * COMPLEX_PARTS
}

/// Layer shapes of one mask-estimator MLP, `[in, out]` per Linear.
///
/// `dim_in` is the band's feature width. The last layer is `2·dim_in` wide
/// because the GLU folds the two halves together, and "depth 2" contributes the
/// *three* Linears `dim_in ← 1536 ← 1536 ← 2·dim_in` that the checkpoint stores.
pub fn mask_mlp_shapes(dim_in: usize) -> [(usize, usize); MASK_MLP_LAYERS] {
    let hidden = FF_MULT * DIM;
    [(DIM, hidden), (hidden, hidden), (hidden, 2 * dim_in)]
}

/// Contiguous feature ranges (`start`, `len`) of each band inside the gathered
/// feature vector, band-major.
pub fn band_feature_ranges(num_freqs_per_band: &[usize]) -> (Vec<i32>, Vec<i32>) {
    let mut starts = Vec::with_capacity(num_freqs_per_band.len());
    let mut lens = Vec::with_capacity(num_freqs_per_band.len());
    let mut acc = 0usize;
    for &n in num_freqs_per_band {
        starts.push((acc * CHANNELS * COMPLEX_PARTS) as i32);
        lens.push(band_dim_in(n) as i32);
        acc += n;
    }
    (starts, lens)
}

/// Compute the mel band layout: the binary mask of the Slaney mel filterbank
/// (`NUM_BANDS` mels, unnormalized), with DC forced into band 0 and Nyquist
/// into band 59. Bands overlap: every bin belongs to one or two of them.
pub fn band_layout() -> BandLayout {
    let n_mels = NUM_BANDS;
    let n_freqs = stft::N_FREQS as usize;
    let nyquist = crate::audio::SAMPLE_RATE as f64 / 2.0;

    // linspace(0, nyquist, n_freqs)
    let fft_step = nyquist / (n_freqs - 1) as f64;
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| {
            if i == n_freqs - 1 {
                nyquist
            } else {
                fft_step * i as f64
            }
        })
        .collect();

    // linspace(hz_to_mel(0), hz_to_mel(nyquist), n_mels + 2), then mel_to_hz
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(nyquist);
    let mel_step = (max_mel - min_mel) / (n_mels + 1) as f64;
    let mel_freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let m = if i == n_mels + 1 {
                max_mel
            } else {
                min_mel + mel_step * i as f64
            };
            mel_to_hz(m)
        })
        .collect();
    let fdiff: Vec<f64> = mel_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    // Triangle support: weight = max(0, min(-ramps[i]/fdiff[i],
    // ramps[i+2]/fdiff[i+1])) > 0
    let mut band_freqs: Vec<Vec<usize>> = vec![Vec::new(); n_mels];
    for b in 0..n_mels {
        for (j, &ff) in fft_freqs.iter().enumerate() {
            let lower = -(mel_freqs[b] - ff) / fdiff[b];
            let upper = (mel_freqs[b + 2] - ff) / fdiff[b + 1];
            if lower.min(upper) > 0.0 {
                band_freqs[b].push(j);
            }
        }
    }
    // Force coverage at the two ends: the filterbank's first and last triangles
    // are open on one side, and the endpoints are set to 1.0 by the reference.
    if band_freqs[0].first() != Some(&0) {
        band_freqs[0].insert(0, 0);
    }
    let last = n_freqs - 1;
    if band_freqs[n_mels - 1].last() != Some(&last) {
        band_freqs[n_mels - 1].push(last);
    }

    let num_freqs_per_band: Vec<usize> = band_freqs.iter().map(|v| v.len()).collect();

    let mut num_bands_per_freq = vec![0i32; n_freqs];
    for bf in &band_freqs {
        for &f in bf {
            num_bands_per_freq[f] += 1;
        }
    }
    assert!(
        num_bands_per_freq.iter().all(|&c| c > 0),
        "not all frequencies are covered by mel bands"
    );

    // Stereo expansion: slot = 2f + ch, band-major.
    let mut freq_indices_stereo: Vec<i32> = Vec::new();
    for bf in &band_freqs {
        for &f in bf {
            for ch in 0..CHANNELS {
                freq_indices_stereo.push(slot_index(f, ch) as i32);
            }
        }
    }

    BandLayout {
        band_freqs,
        num_freqs_per_band,
        freq_indices_stereo,
        num_bands_per_freq,
    }
}

/// Which of the two alternating transformer axes a block runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Time,
    Freq,
}

impl Axis {
    /// The name component the export uses for this axis.
    pub fn key_part(self) -> &'static str {
        match self {
            Axis::Time => "time_transformer",
            Axis::Freq => "freq_transformer",
        }
    }

    /// Both axes, in the order a block runs them.
    pub fn both() -> [Axis; 2] {
        [Axis::Time, Axis::Freq]
    }
}

// ──────────────────────────── small helpers ────────────────────────

fn mm(a: &Array, b: &Array) -> Array {
    matmul(a, b).expect("matmul")
}
fn adda(a: &Array, b: &Array) -> Array {
    add(a, b).expect("add")
}
fn mula(a: &Array, b: &Array) -> Array {
    multiply(a, b).expect("multiply")
}

/// Host-side statement of the [`L2Norm`] rule, one row at a time.
///
/// Not used by the graph (the device path below is the implementation the
/// model runs), but it is the one place the formula is written in ordinary
/// arithmetic, and `l2norm_device_path_matches_the_formula` holds the device op
/// to it.
pub fn l2_norm_row(x: &[f32], weight: f32, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = norm.max(L2_EPS);
    let scale = (x.len() as f32).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v / denom) * scale * weight;
    }
}

/// Exact (erf-based) GELU, matching PyTorch's `nn.GELU()` default.
///
/// The tanh approximation is *not* used: it agrees to ~1e-3 per element and the
/// mask that comes out of a stack of these is not close enough to the trained
/// model.
pub fn gelu(x: &Array) -> Array {
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);
    let inv_sqrt2 = Array::from_f32(std::f32::consts::FRAC_1_SQRT_2);
    let inner = multiply(x, &inv_sqrt2).expect("gelu: scale");
    let e = erf(&inner).expect("gelu: erf");
    let bracket = add(&one, &e).expect("gelu: bracket");
    multiply(&half, &multiply(x, &bracket).expect("gelu: x*b")).expect("gelu: 0.5*x*b")
}

// ──────────────────────────── layers ───────────────────────────────

/// `x / max(||x||, eps) * sqrt(dim) * weight`: the PyTorch-compatible L2Norm.
///
/// Do **not** replace this with a root-mean-square norm fast path or with a
/// plain "divide by the norm": the first moves where `eps` applies and the
/// second drops the `sqrt(dim)`, which rescales every activation by
/// `1/sqrt(384) ≈ 0.051` and is not recoverable downstream.
pub struct L2Norm {
    pub(crate) weight: Array,
    scale: f32,
}

impl L2Norm {
    pub fn new(weight: Array, dim: usize) -> Self {
        L2Norm {
            weight,
            scale: (dim as f32).sqrt(),
        }
    }

    /// The `sqrt(dim)` multiplier, exposed for the host-side reference.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    fn forward(&self, x: &Array) -> Array {
        let sq = mula(x, x);
        let sum = sum_axis(&sq, -1, true).expect("l2norm: sum");
        let norm = sqrt(&sum).expect("l2norm: sqrt");
        let denom = maximum(&norm, &Array::from_f32(L2_EPS)).expect("l2norm: maximum");
        let scaled = mlx_rs::ops::divide(x, &denom).expect("l2norm: divide");
        let scaled = mula(&scaled, &Array::from_f32(self.scale));
        mula(&scaled, &self.weight)
    }
}

pub struct Attention {
    pub(crate) norm: L2Norm,
    pub(crate) w_qkv: Array,   // [384, 1536] (pre-transposed)
    pub(crate) w_gates: Array, // [384, 8]
    pub(crate) b_gates: Array, // [8]
    pub(crate) w_out: Array,   // [512, 384]
}

impl Attention {
    fn forward(&self, x: &Array) -> Array {
        let n = x.dim(x.ndim() as i32 - 2);
        let h = self.norm.forward(x); // (b, n, 384)

        // (b, n, 1536)
        let qkv = mm(&h, &self.w_qkv);
        let q = qkv.index((.., .., 0..DIM_INNER as i32));
        let k = qkv.index((.., .., DIM_INNER as i32..2 * DIM_INNER as i32));
        let v = qkv.index((.., .., 2 * DIM_INNER as i32..3 * DIM_INNER as i32));

        // (b, n, 8, 64) → (b, 8, n, 64)
        let to_heads = |t: &Array| {
            t.reshape(&[-1, n, HEADS as i32, DIM_HEAD as i32])
                .expect("attn: reshape heads")
                .transpose_axes(&[0, 2, 1, 3])
                .expect("attn: transpose heads")
        };
        let mut q = to_heads(&q);
        let mut k = to_heads(&k);
        let v = to_heads(&v);

        // RoPE: `traditional = true` (interleaved pairs) and base 10000: the
        // repeat_interleave + rotate_half semantics of the reference
        // implementation. Validated by the parity arm, not by anything here.
        q = fast::rope(&q, DIM_HEAD as i32, true, 10000.0f32, 1.0f32, 0i32, None)
            .expect("attn: rope q");
        k = fast::rope(&k, DIM_HEAD as i32, true, 10000.0f32, 1.0f32, 0i32, None)
            .expect("attn: rope k");

        // Manual attention. The fused scaled-dot-product variant is not used:
        // it does not reproduce this arithmetic on real inputs.
        let scale = Array::from_f32(1.0 / (DIM_HEAD as f32).sqrt()); // 1/8
        let k_t = k.transpose_axes(&[0, 1, 3, 2]).expect("attn: k^T");
        let scores = mula(&mm(&q, &k_t), &scale); // (b, 8, n, n)
        let max_s = max_axis(&scores, -1, true).expect("attn: max");
        let scores = mlx_rs::ops::subtract(&scores, &max_s).expect("attn: stable");
        let attn = softmax_axis(&scores, -1, false).expect("attn: softmax");
        let out = mm(&attn, &v); // (b, 8, n, 64)

        // Gates: sigmoid(Linear(x_norm)) → (b, n, 8) → (b, 8, n, 1)
        let gates = sigmoid(&adda(&mm(&h, &self.w_gates), &self.b_gates)).expect("attn: gates");

        let gates = gates
            .transpose_axes(&[0, 2, 1])
            .expect("attn: gates transpose");
        let gates = expand_dims_axes(&gates, &[3]).expect("attn: gates expand");
        let out = mula(&out, &gates);

        // Merge heads: (b, 8, n, 64) → (b, n, 512)
        let out = out
            .transpose_axes(&[0, 2, 1, 3])
            .expect("attn: merge transpose")
            .reshape(&[-1, n, DIM_INNER as i32])
            .expect("attn: merge reshape");
        mm(&out, &self.w_out)
    }
}

pub struct FeedForward {
    pub(crate) norm: L2Norm,
    pub(crate) w1: Array, // [384, 1536]
    pub(crate) b1: Array, // [1536]
    pub(crate) w2: Array, // [1536, 384]
    pub(crate) b2: Array, // [384]
}

impl FeedForward {
    fn forward(&self, x: &Array) -> Array {
        let h = self.norm.forward(x);
        let h = adda(&mm(&h, &self.w1), &self.b1);
        let h = gelu(&h);
        adda(&mm(&h, &self.w2), &self.b2)
    }
}

pub struct TransformerLayer {
    pub(crate) attn: Attention,
    pub(crate) ff: FeedForward,
}

impl TransformerLayer {
    fn forward(&self, x: &Array) -> Array {
        let x = adda(&self.attn.forward(x), x);
        adda(&self.ff.forward(&x), &x)
    }
}

/// Transformer with an L2Norm output norm (`norm_output = true` in the trained
/// checkpoint (the twelve `…_transformer.norm.weight` tensors are used).
pub struct Transformer {
    pub(crate) layers: Vec<TransformerLayer>,
    pub(crate) norm: L2Norm,
}

impl Transformer {
    fn forward(&self, x: &Array) -> Array {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.norm.forward(&x)
    }
}

pub struct Block {
    pub(crate) time: Transformer,
    pub(crate) freq: Transformer,
}

impl Block {
    /// `x`: `(1, frames, bands, dim)`
    fn forward(&self, x: &Array, frames: i32) -> Array {
        // Time transformer: batch over bands → (bands, frames, dim)
        let x_t = x
            .transpose_axes(&[0, 2, 1, 3])
            .expect("block: t transpose")
            .reshape(&[NUM_BANDS as i32, frames, DIM as i32])
            .expect("block: t reshape");
        let x_t = self.time.forward(&x_t);

        // Freq transformer: batch over time → (frames, bands, dim)
        let x_f = x_t
            .transpose_axes(&[1, 0, 2])
            .expect("block: f transpose")
            .reshape(&[frames, NUM_BANDS as i32, DIM as i32])
            .expect("block: f reshape");
        let x_f = self.freq.forward(&x_f);

        // Back to (1, frames, bands, dim): the freq output is already
        // (frames, bands, dim), so a plain reshape prepends the batch axis. A
        // transpose here would swap the time and frequency axes.
        x_f.reshape(&[1, frames, NUM_BANDS as i32, DIM as i32])
            .expect("block: back reshape")
    }
}

pub struct BandSplitModule {
    pub(crate) norm: L2Norm,
    pub(crate) w: Array, // [dim_in, 384]
    pub(crate) b: Array, // [384]
}

impl BandSplitModule {
    fn forward(&self, x: &Array) -> Array {
        let h = self.norm.forward(x);
        adda(&mm(&h, &self.w), &self.b)
    }
}

pub struct BandSplit {
    pub(crate) bands: Vec<BandSplitModule>,
    /// Feature-axis slice start per band, in the **gathered** feature vector
    /// (band-major: band `b` owns `4·nf_b` contiguous features).
    feat_starts: Vec<i32>,
    feat_dims: Vec<i32>,
}

impl BandSplit {
    /// `x_gathered`: `(1, frames, 2G)` → `(1, frames, 60, 384)`
    fn forward(&self, x_gathered: &Array) -> Array {
        let mut outs = Vec::with_capacity(NUM_BANDS);
        for b in 0..NUM_BANDS {
            let s = self.feat_starts[b];
            let d = self.feat_dims[b];
            let slice = x_gathered.index((.., .., s..s + d));
            let o = self.bands[b].forward(&slice);
            outs.push(expand_dims_axes(&o, &[-2]).expect("band_split: expand"));
        }
        concatenate_axis(&outs, -2).expect("band_split: stack")
    }
}

/// Per-band MLP with torch depth semantics (`depth = 2` → three Linears, Tanh
/// between), followed by a GLU: first half × sigmoid(second half).
pub struct MaskMlp {
    pub(crate) w: Vec<Array>, // [in, out] per layer
    pub(crate) b: Vec<Array>,
    pub(crate) dim_in: usize, // band feature dim (4 · n_freqs_in_band)
}

impl MaskMlp {
    fn forward(&self, x: &Array) -> Array {
        let mut h = x.clone();
        for (i, (w, b)) in self.w.iter().zip(self.b.iter()).enumerate() {
            h = adda(&mm(&h, w), b);
            if i + 1 < self.w.len() {
                h = tanh(&h).expect("mlp: tanh");
            }
        }
        let v = h.index((.., .., 0..self.dim_in as i32));
        let g = h.index((.., .., self.dim_in as i32..2 * self.dim_in as i32));
        multiply(&v, &sigmoid(&g).expect("mlp: sigmoid")).expect("mlp: glu")
    }
}

pub struct MaskEstimator {
    pub(crate) mlps: Vec<MaskMlp>,
}

impl MaskEstimator {
    /// `x`: `(1, frames, 60, 384)` → `(1, frames, 2G)` in the gathered-slot
    /// layout (feature = gathered slot `g · 2 + re/im`).
    fn forward(&self, x: &Array) -> Array {
        let mut outs = Vec::with_capacity(NUM_BANDS);
        for b in 0..NUM_BANDS {
            let slice = x
                .index((0..1, 0..x.dim(1), b as i32..b as i32 + 1, 0..DIM as i32))
                .reshape(&[1, x.dim(1), DIM as i32])
                .expect("mask: band reshape");
            outs.push(self.mlps[b].forward(&slice));
        }
        concatenate_axis(&outs, -1).expect("mask: concat")
    }
}

// ──────────────────────────── model ────────────────────────────

pub struct RoFormerModel {
    pub(crate) band_split: BandSplit,
    pub(crate) blocks: Vec<Block>,
    pub(crate) mask_estimator: MaskEstimator,
    /// Feature gather indices `(2G,)`: for each gathered slot `g`, the pair
    /// `(2·slot, 2·slot + 1)` into the `4·n_freqs`-wide feature vector.
    gather_idx: Array,
    /// One-hot scatter matrix `(2·n_freqs, G)`: `M[slot, g] = 1` iff gathered
    /// slot `g` maps to full slot `freq_indices_stereo[g]`.
    scatter_mt: Array,
    /// Overlap denominator `(1, 2·n_freqs, 1, 1)`: `num_bands_per_freq`
    /// repeated per channel, clamped at 1e-8.
    denom: Array,
    gathered_slots: usize,
}

impl RoFormerModel {
    /// Allocate an uninitialized model with the checkpoint's shapes.
    pub fn new() -> Self {
        let layout = band_layout();
        let num_freqs_per_band = layout.num_freqs_per_band.clone();
        let gathered_slots = layout.freq_indices_stereo.len();
        let n_slots_full = CHANNELS * stft::N_FREQS as usize;

        // Feature gather indices: slot `s` occupies features (2s, 2s+1).
        let mut gather_idx_v = Vec::with_capacity(COMPLEX_PARTS * gathered_slots);
        for &s in &layout.freq_indices_stereo {
            let s = s as usize;
            // Slot `s` is `(f, ch)` with `s = 2f + ch`, so its two parts sit at
            // exactly the interleaved positions `feature_index` defines.
            let f = s / CHANNELS;
            let ch = s % CHANNELS;
            for ri in 0..COMPLEX_PARTS {
                gather_idx_v.push(feature_index(f, ch, ri) as i32);
            }
        }
        let gather_idx = Array::from_slice(&gather_idx_v, &[gather_idx_v.len() as i32]);

        // One-hot scatter matrix (2·n_freqs, G).
        let mut mt = vec![0.0f32; n_slots_full * gathered_slots];
        for (g, &s) in layout.freq_indices_stereo.iter().enumerate() {
            mt[s as usize * gathered_slots + g] = 1.0;
        }
        let scatter_mt = Array::from_slice(&mt, &[n_slots_full as i32, gathered_slots as i32]);

        // Denominator: bands-per-frequency repeated per channel. Shape
        // (1, slots, 1, 1) so it broadcasts against `summed`
        // (1, slots, frames, 2): the slot axis is dim 1.
        let mut denom_v = Vec::with_capacity(n_slots_full);
        for &c in &layout.num_bands_per_freq {
            for _ in 0..CHANNELS {
                denom_v.push((c as f32).max(1e-8));
            }
        }
        let denom = Array::from_slice(&denom_v, &[1, n_slots_full as i32, 1, 1]);

        let bands = num_freqs_per_band
            .iter()
            .map(|&n| {
                let dim_in = band_dim_in(n);
                BandSplitModule {
                    norm: L2Norm::new(
                        Array::zeros::<f32>(&[dim_in as i32]).expect("zeros"),
                        dim_in,
                    ),
                    w: Array::zeros::<f32>(&[dim_in as i32, DIM as i32]).expect("zeros"),
                    b: Array::zeros::<f32>(&[DIM as i32]).expect("zeros"),
                }
            })
            .collect();

        let blocks = (0..DEPTH)
            .map(|_| {
                let attn = || Attention {
                    norm: L2Norm::new(Array::zeros::<f32>(&[DIM as i32]).expect("zeros"), DIM),
                    w_qkv: Array::zeros::<f32>(&[DIM as i32, 3 * DIM_INNER as i32]).expect("zeros"),
                    w_gates: Array::zeros::<f32>(&[DIM as i32, HEADS as i32]).expect("zeros"),
                    b_gates: Array::zeros::<f32>(&[HEADS as i32]).expect("zeros"),
                    w_out: Array::zeros::<f32>(&[DIM_INNER as i32, DIM as i32]).expect("zeros"),
                };
                let ff = || FeedForward {
                    norm: L2Norm::new(Array::zeros::<f32>(&[DIM as i32]).expect("zeros"), DIM),
                    w1: Array::zeros::<f32>(&[DIM as i32, (FF_MULT * DIM) as i32]).expect("zeros"),
                    b1: Array::zeros::<f32>(&[(FF_MULT * DIM) as i32]).expect("zeros"),
                    w2: Array::zeros::<f32>(&[(FF_MULT * DIM) as i32, DIM as i32]).expect("zeros"),
                    b2: Array::zeros::<f32>(&[DIM as i32]).expect("zeros"),
                };
                let transformer = || Transformer {
                    layers: vec![TransformerLayer {
                        attn: attn(),
                        ff: ff(),
                    }],
                    norm: L2Norm::new(Array::zeros::<f32>(&[DIM as i32]).expect("zeros"), DIM),
                };
                Block {
                    time: transformer(),
                    freq: transformer(),
                }
            })
            .collect();

        let mlps = num_freqs_per_band
            .iter()
            .map(|&n| {
                let dim_in = band_dim_in(n);
                let shapes = mask_mlp_shapes(dim_in);
                MaskMlp {
                    w: shapes
                        .iter()
                        .map(|&(i, o)| Array::zeros::<f32>(&[i as i32, o as i32]).expect("zeros"))
                        .collect(),
                    b: shapes
                        .iter()
                        .map(|&(_, o)| Array::zeros::<f32>(&[o as i32]).expect("zeros"))
                        .collect(),
                    dim_in,
                }
            })
            .collect();

        // Band-major gathered features: band b owns 4·nf_b contiguous entries.
        let (feat_starts, feat_dims) = band_feature_ranges(&num_freqs_per_band);

        RoFormerModel {
            band_split: BandSplit {
                bands,
                feat_starts,
                feat_dims,
            },
            blocks,
            mask_estimator: MaskEstimator { mlps },
            gather_idx,
            scatter_mt,
            denom,
            gathered_slots,
        }
    }

    /// Gather indices, for tests and for anyone who wants to check the layout
    /// without running a forward.
    pub fn gather_indices(&self) -> Vec<i32> {
        super::ensure_contiguous(&self.gather_idx)
            .as_slice::<i32>()
            .to_vec()
    }

    /// Number of gathered slots `G`.
    pub fn gathered_slots(&self) -> usize {
        self.gathered_slots
    }

    /// Full forward: one raw audio window `(1, 2, T)` → separated vocals
    /// `(1, 2, T)`.
    pub fn forward(&self, audio: &Array) -> Array {
        assert_eq!(audio.ndim(), 3, "audio must be (1, 2, T)");
        assert_eq!(audio.dim(1), CHANNELS as i32, "stereo input required");
        let t_samples = audio.dim(2);
        let frames = stft::n_frames(t_samples);
        let n_freqs = stft::N_FREQS;
        let g = self.gathered_slots as i32;

        // ── STFT per channel → complex (1, N_FREQS, frames) ──
        let mut specs = Vec::with_capacity(CHANNELS);
        for c in 0..CHANNELS as i32 {
            let ch = audio
                .index((0..1, c..c + 1, 0..t_samples))
                .reshape(&[1, t_samples])
                .expect("forward: channel reshape");
            specs.push(stft::stft(&ch));
        }

        // ── Features (1, frames, 4·n_freqs), layout j = f·4 + ch·2 + ri ──
        let re = concatenate_axis(
            &[
                &specs[0].real().expect("features: re0"),
                &specs[1].real().expect("features: re1"),
            ],
            0,
        )
        .expect("features: re concat"); // (2, 1025, frames)
        let im = concatenate_axis(
            &[
                &specs[0].imag().expect("features: im0"),
                &specs[1].imag().expect("features: im1"),
            ],
            0,
        )
        .expect("features: im concat");
        let re = expand_dims_axes(&re, &[3]).expect("features: re expand"); // (2,1025,frames,1)
        let im = expand_dims_axes(&im, &[3]).expect("features: im expand");
        let ri = concatenate_axis(&[&re, &im], -1).expect("features: ri concat"); // (ch,f,t,ri)
        let feat = ri
            .transpose_axes(&[1, 0, 2, 3])
            .expect("features: t1") // (f, ch, t, ri)
            .transpose_axes(&[0, 2, 1, 3])
            .expect("features: t2") // (f, t, ch, ri)
            .reshape(&[n_freqs, frames, 4])
            .expect("features: reshape") // inner 4 = (ch, ri)
            .transpose_axes(&[1, 0, 2])
            .expect("features: t3") // (t, f, 4)
            .reshape(&[1, frames, n_freqs * 4])
            .expect("features: flat");

        // ── Gather mel-band features → band split → transformers → mask ──
        let xg = feat
            .take_axis(&self.gather_idx, 2)
            .expect("forward: gather"); // (1, frames, 2G)
        let x = self.band_split.forward(&xg); // (1, frames, 60, 384)
        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(&x, frames);
        }
        let masks = self.mask_estimator.forward(&x); // (1, frames, 2G)

        // ── Scatter-add the overlapping band masks, divide by bands-per-bin ──
        // (1, frames, G, 2) → (1, G, frames, 2) → (1, G, frames·2)
        let m = masks
            .reshape(&[1, frames, g, 2])
            .expect("masks: reshape")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("masks: transpose")
            .reshape(&[1, g, frames * 2])
            .expect("masks: flatten");
        // matmul (2050, G) @ (1, G, frames·2) → (1, 2050, frames·2): sums the
        // overlapping band masks per full slot.
        let summed = mm(&self.scatter_mt, &m)
            .reshape(&[1, CHANNELS as i32 * n_freqs, frames, 2])
            .expect("masks: scatter reshape");
        let masks = mlx_rs::ops::divide(&summed, &self.denom).expect("masks: denom");

        // ── Complex multiply with the STFT ──
        let sc = concatenate_axis(&[&specs[0], &specs[1]], 0)
            .expect("stft: ch concat") // (2, 1025, frames)
            .transpose_axes(&[1, 0, 2])
            .expect("stft: fch") // (f, ch, t)
            .reshape(&[1, CHANNELS as i32 * n_freqs, frames])
            .expect("stft: flatten"); // (1, 2050, frames) complex

        let m_re = masks.index((0..1, 0..CHANNELS as i32 * n_freqs, 0..frames, 0..1));
        let m_im = masks.index((0..1, 0..CHANNELS as i32 * n_freqs, 0..frames, 1..2));
        let imag_unit = Array::from_complex(complex64::new(0.0, 1.0));
        let m_im_c = multiply(&m_im, &imag_unit).expect("mask: im complex");
        let m_c = add(
            &m_re.as_type::<complex64>().expect("mask: re complex"),
            &m_im_c,
        )
        .expect("mask: complex")
        // (1, 2050, frames, 1) → (1, 2050, frames) to match `sc` for the
        // element-wise complex multiply.
        .reshape(&[1, CHANNELS as i32 * n_freqs, frames])
        .expect("mask: complex reshape");
        let masked = multiply(&sc, &m_c).expect("apply mask"); // (1,2050,frames)

        // ── Split back to channels and iSTFT ──
        let per_ch = masked
            .reshape(&[n_freqs, CHANNELS as i32, frames])
            .expect("istft: reshape") // (f, ch, t)
            .transpose_axes(&[1, 0, 2])
            .expect("istft: transpose"); // (2, f, t)
        let mut channels = Vec::with_capacity(CHANNELS);
        for c in 0..CHANNELS as i32 {
            let spec_c = per_ch
                .index((c..c + 1, 0..n_freqs, 0..frames))
                .reshape(&[1, n_freqs, frames])
                .expect("istft: channel reshape");
            channels.push(stft::istft(&spec_c, t_samples)); // (1, T)
        }
        concatenate_axis(&[&channels[0], &channels[1]], 0).expect("output concat")
    }
}

// ──────────────────────── weight plumbing ────────────────────────
//
// Every setter below is called exactly once per tensor by `weights.rs`, which
// has already checked the shape against the architecture. They take ownership of
// the loaded `Array` rather than copying data, and each asserts the rank it
// expects; a wrong-rank weight would otherwise first show up as a wrong-sounding
// separation.

impl L2Norm {
    /// Per-feature gain, one value per normalized feature.
    pub(crate) fn set_weight(&mut self, weight: Array) {
        assert_eq!(weight.ndim(), 1, "L2Norm weight must be rank 1");
        self.weight = weight;
    }
}

impl Attention {
    pub(crate) fn set_norm(&mut self, weight: Array) {
        self.norm.set_weight(weight);
    }
    /// `[dim, 3 · heads · dim_head]`, q/k/v concatenated in that order.
    pub(crate) fn set_qkv(&mut self, w: Array) {
        assert_rank2(&w, "to_qkv");
        self.w_qkv = w;
    }
    pub(crate) fn set_gates(&mut self, w: Array, b: Array) {
        assert_rank2(&w, "to_gates");
        assert_eq!(b.ndim(), 1, "to_gates bias must be rank 1");
        self.w_gates = w;
        self.b_gates = b;
    }
    pub(crate) fn set_out(&mut self, w: Array) {
        assert_rank2(&w, "to_out");
        self.w_out = w;
    }
}

impl FeedForward {
    pub(crate) fn set_norm(&mut self, weight: Array) {
        self.norm.set_weight(weight);
    }
    /// The `dim → 4·dim` expansion that the exact GELU is applied to.
    pub(crate) fn set_first(&mut self, w: Array, b: Array) {
        assert_rank2(&w, "ff.1");
        self.w1 = w;
        self.b1 = b;
    }
    /// The `4·dim → dim` projection back to model width.
    pub(crate) fn set_second(&mut self, w: Array, b: Array) {
        assert_rank2(&w, "ff.4");
        self.w2 = w;
        self.b2 = b;
    }
}

fn assert_rank2(w: &Array, what: &str) {
    assert_eq!(w.ndim(), 2, "{what} weight must be rank 2 [in, out]");
}

impl Transformer {
    pub(crate) fn attention_mut(&mut self) -> &mut Attention {
        &mut self.layers[0].attn
    }
    pub(crate) fn feed_forward_mut(&mut self) -> &mut FeedForward {
        &mut self.layers[0].ff
    }
    /// The block's output norm: the last normalization in the graph.
    pub(crate) fn set_out_norm(&mut self, weight: Array) {
        self.norm.set_weight(weight);
    }
}

impl Block {
    pub(crate) fn transformer_mut(&mut self, axis: Axis) -> &mut Transformer {
        match axis {
            Axis::Time => &mut self.time,
            Axis::Freq => &mut self.freq,
        }
    }
}

impl BandSplitModule {
    pub(crate) fn set_norm(&mut self, weight: Array) {
        self.norm.set_weight(weight);
    }
    pub(crate) fn set_linear(&mut self, w: Array, b: Array) {
        assert_rank2(&w, "band split");
        self.w = w;
        self.b = b;
    }
}

impl MaskMlp {
    /// One of the `depth + 1` Linears, in order.
    pub(crate) fn set_layer(&mut self, slot: usize, w: Array, b: Array) {
        assert!(slot < MASK_MLP_LAYERS, "mask MLP has no layer {slot}");
        assert_rank2(&w, "mask estimator");
        assert_eq!(b.ndim(), 1, "mask estimator bias must be rank 1");
        self.w[slot] = w;
        self.b[slot] = b;
    }
}

impl RoFormerModel {
    /// The 60 band-split modules, for the loader.
    pub(crate) fn band_split_mut(&mut self) -> &mut [BandSplitModule] {
        &mut self.band_split.bands
    }
    /// The `depth` blocks, for the loader.
    pub(crate) fn blocks_mut(&mut self) -> &mut [Block] {
        &mut self.blocks
    }
    /// The 60 mask-estimator MLPs, for the loader.
    pub(crate) fn mask_estimator_mut(&mut self) -> &mut [MaskMlp] {
        &mut self.mask_estimator.mlps
    }
}

impl Default for RoFormerModel {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────── tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Values of the Slaney mel filterbank's bin counts, from the reference
    /// implementation the checkpoint was trained with. Bands start at 7 bins and
    /// end at 130, and the totals are what every shape in this file derives
    /// from.
    #[test]
    fn band_layout_matches_the_trained_filterbank() {
        let layout = band_layout();
        assert_eq!(
            layout.num_freqs_per_band,
            vec![
                7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 9, 9, 9, 10, 10, 11, 13, 13,
                13, 15, 16, 17, 19, 20, 20, 22, 24, 26, 28, 29, 31, 33, 36, 39, 41, 44, 47, 50, 54,
                57, 61, 66, 71, 76, 80, 86, 93, 99, 105, 113, 122, 130
            ]
        );
        assert_eq!(layout.num_freqs_per_band.len(), NUM_BANDS);
        assert_eq!(layout.num_freqs_per_band.iter().sum::<usize>(), 1979);
        assert_eq!(layout.freq_indices_stereo.len(), 3958);
        // Every bin covered, by at most two bands.
        assert!(layout.num_bands_per_freq.iter().all(|&c| c >= 1));
        assert_eq!(layout.num_bands_per_freq.iter().copied().max(), Some(2));
        assert_eq!(layout.num_bands_per_freq.len(), stft::N_FREQS as usize);
        // Band 0 covers DC, band 59 covers Nyquist.
        assert_eq!(layout.band_freqs[0][0], 0);
        assert_eq!(layout.band_freqs[NUM_BANDS - 1].last(), Some(&1024));
        // Bands are ascending and stereo slots are (2f, 2f+1).
        for bf in &layout.band_freqs {
            assert!(bf.windows(2).all(|w| w[1] > w[0]), "band not ascending");
        }
        assert_eq!(layout.freq_indices_stereo[0], 0);
        assert_eq!(layout.freq_indices_stereo[1], 1);
        assert_eq!(*layout.freq_indices_stereo.last().unwrap(), 2 * 1024 + 1);
    }

    /// The band layout is derived from the sample rate the crate is fixed at, so
    /// a drift in either would change every shape in the model.
    #[test]
    fn band_layout_is_anchored_to_the_sample_rate_and_bin_count() {
        assert_eq!(crate::audio::SAMPLE_RATE, 44_100);
        assert_eq!(stft::N_FREQS, 1025);
        // 60 mels over 0 … 22 050 Hz, Slaney: the linear segment below 1 kHz
        // holds exactly 200/3 Hz per mel, so hz_to_mel and mel_to_hz must be
        // inverses on both sides of the kink.
        for &hz in &[0.0, 100.0, 999.0, 1000.0, 1001.0, 4000.0, 22_050.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!(
                (back - hz).abs() <= hz * 1e-9 + 1e-9,
                "mel round trip at {hz} Hz gave {back}"
            );
        }
        assert_eq!(hz_to_mel(1000.0), MIN_LOG_MEL);
    }

    /// `j = f·4 + ch·2 + ri`, the interleaved layout this model was trained
    /// with, and the reason the gather can address complex stereo features as a
    /// flat vector.
    #[test]
    fn feature_layout_is_the_documented_interleaving() {
        assert_eq!(feature_index(0, 0, 0), 0);
        assert_eq!(feature_index(0, 0, 1), 1);
        assert_eq!(feature_index(0, 1, 0), 2);
        assert_eq!(feature_index(0, 1, 1), 3);
        assert_eq!(feature_index(1, 0, 0), 4);
        assert_eq!(feature_index(1024, 1, 1), 4100 - 1);
        // Directly from the formula, for every position the model can address.
        for f in 0..stft::N_FREQS as usize {
            for ch in 0..CHANNELS {
                for ri in 0..COMPLEX_PARTS {
                    assert_eq!(feature_index(f, ch, ri), f * 4 + ch * 2 + ri);
                    // And the slot view agrees with it: slot `2f + ch` owns
                    // features `2·slot` and `2·slot + 1`.
                    let slot = slot_index(f, ch);
                    assert_eq!(slot, 2 * f + ch);
                    assert_eq!(feature_index(f, ch, 0), 2 * slot);
                    assert_eq!(feature_index(f, ch, 1), 2 * slot + 1);
                }
            }
        }
        // The layout is a bijection onto `4·n_freqs` features.
        let mut seen = vec![false; CHANNELS * stft::N_FREQS as usize * COMPLEX_PARTS];
        assert_eq!(seen.len(), 4100);
        for f in 0..stft::N_FREQS as usize {
            for ch in 0..CHANNELS {
                for ri in 0..COMPLEX_PARTS {
                    let j = feature_index(f, ch, ri);
                    assert!(!seen[j], "feature {j} addressed twice");
                    seen[j] = true;
                }
            }
        }
        assert!(seen.into_iter().all(|s| s), "feature hole");
    }

    /// A band's feature vector is its bins × channels × re/im, and the
    /// band-major ranges have to tile the gathered vector exactly once.
    #[test]
    fn band_ranges_partition_the_gathered_features() {
        let layout = band_layout();
        let (starts, lens) = band_feature_ranges(&layout.num_freqs_per_band);
        assert_eq!(starts.len(), NUM_BANDS);
        assert_eq!(lens.len(), NUM_BANDS);
        assert_eq!(band_dim_in(layout.num_freqs_per_band[0]), 28);
        let mut cursor = 0i64;
        for b in 0..NUM_BANDS {
            assert_eq!(
                band_dim_in(layout.num_freqs_per_band[b]) as i64,
                lens[b] as i64
            );
            assert_eq!(
                starts[b] as i64, cursor,
                "band {b} does not abut the previous"
            );
            cursor += lens[b] as i64;
        }
        // Σ = 2G = 4·Σnf = 7916 features.
        assert_eq!(cursor, 2 * layout.freq_indices_stereo.len() as i64);
        assert_eq!(cursor, 7916);
    }

    /// The gather the forward applies is built from `feature_index`, so the
    /// index array itself is the layout claim. Checked against an independent
    /// recomputation of `2·(2f + ch) + ri`.
    #[test]
    fn gather_indices_encode_the_layout_for_every_band_slot() {
        let layout = band_layout();
        let mut expect = Vec::with_capacity(2 * layout.freq_indices_stereo.len());
        for &slot in &layout.freq_indices_stereo {
            let slot = slot as usize;
            let (f, ch) = (slot / CHANNELS, slot % CHANNELS);
            expect.push((f * 4 + ch * 2) as i32);
            expect.push((f * 4 + ch * 2 + 1) as i32);
        }
        // `gather_indices` needs an allocated model, but not weights and not a
        // forward: the array is built from the constants alone.
        let model = RoFormerModel::new();
        assert_eq!(model.gathered_slots(), 3958);
        assert_eq!(model.gather_indices(), expect);
        assert!(model
            .gather_indices()
            .iter()
            .all(|&j| (0..4100).contains(&j)));
    }

    /// `dim = 384`, `heads·dim_head = 512`, hidden `= 4·dim = 1536`, and a mask
    /// MLP of `depth + 1 = 3` Linears ending in `2·dim_in` for the GLU. These
    /// are the shapes `weights.rs` looks for; a drift here is a load failure
    /// there, and this is the test that catches it without a weight file.
    #[test]
    fn architecture_shapes_follow_from_the_constants() {
        assert_eq!(DIM, 384);
        assert_eq!(DEPTH, 6);
        assert_eq!(HEADS, 8);
        assert_eq!(DIM_HEAD, 64);
        assert_eq!(NUM_BANDS, 60);
        assert_eq!(DIM_INNER, 512);
        assert_eq!(MASK_MLP_LAYERS, 3, "mask_estimator_depth + 1");
        assert_eq!(FF_MULT * DIM, 1536);
        assert_eq!(CHANNELS, 2);
        assert_eq!(COMPLEX_PARTS, 2);

        // Band 0: 7 bins → 28 features; band 59: 130 bins → 520 features.
        let layout = band_layout();
        let first = mask_mlp_shapes(band_dim_in(layout.num_freqs_per_band[0]));
        assert_eq!(first, [(384, 1536), (1536, 1536), (1536, 56)]);
        let last = mask_mlp_shapes(band_dim_in(layout.num_freqs_per_band[NUM_BANDS - 1]));
        assert_eq!(last, [(384, 1536), (1536, 1536), (1536, 1040)]);
        for &n in &layout.num_freqs_per_band {
            let s = mask_mlp_shapes(band_dim_in(n));
            assert_eq!(s[0], (DIM, FF_MULT * DIM));
            assert_eq!(s[1], (FF_MULT * DIM, FF_MULT * DIM));
            assert_eq!(s[2].0, FF_MULT * DIM);
            assert_eq!(s[2].1, 2 * band_dim_in(n), "GLU needs both halves");
        }
    }

    /// A model allocated from the constants alone must have the shapes the
    /// checkpoint's 672 tensors fill, no weights involved.
    #[test]
    fn allocated_model_has_the_checkpoints_shapes() {
        let model = RoFormerModel::new();
        assert_eq!(model.blocks.len(), DEPTH);
        for block in &model.blocks {
            assert_eq!(block.time.layers.len(), 1);
            assert_eq!(block.freq.layers.len(), 1);
            assert_eq!(
                block.time.layers[0].attn.w_qkv.shape(),
                &[DIM as i32, 3 * DIM_INNER as i32]
            );
            assert_eq!(
                block.time.layers[0].attn.w_gates.shape(),
                &[DIM as i32, HEADS as i32]
            );
            assert_eq!(
                block.time.layers[0].attn.w_out.shape(),
                &[DIM_INNER as i32, DIM as i32]
            );
            assert_eq!(
                block.freq.layers[0].ff.w1.shape(),
                &[DIM as i32, (FF_MULT * DIM) as i32]
            );
            assert_eq!(
                block.freq.layers[0].ff.w2.shape(),
                &[(FF_MULT * DIM) as i32, DIM as i32]
            );
        }
        assert_eq!(model.band_split.bands.len(), NUM_BANDS);
        assert_eq!(model.mask_estimator.mlps.len(), NUM_BANDS);
        let layout = band_layout();
        for b in 0..NUM_BANDS {
            let dim_in = band_dim_in(layout.num_freqs_per_band[b]);
            assert_eq!(
                model.band_split.bands[b].norm.weight.shape(),
                &[dim_in as i32]
            );
            assert_eq!(
                model.band_split.bands[b].w.shape(),
                &[dim_in as i32, DIM as i32]
            );
            let mlp = &model.mask_estimator.mlps[b];
            assert_eq!(mlp.w.len(), MASK_MLP_LAYERS);
            assert_eq!(
                mlp.w[MASK_MLP_LAYERS - 1].shape(),
                &[(FF_MULT * DIM) as i32, (2 * dim_in) as i32]
            );
            assert_eq!(mlp.b[MASK_MLP_LAYERS - 1].shape(), &[2 * dim_in as i32]);
        }
    }

    /// The convention, as arithmetic: a unit-L2 row of `[3, 4]` scaled by
    /// `sqrt(2)`.
    ///
    /// Dropping the `sqrt(dim)`, which is what "just divide by the norm" does:
    /// gives `[0.6, 0.8]`, a factor `1/sqrt(2)` away on this row and
    /// `1/sqrt(384)` on every real activation.
    #[test]
    fn l2norm_keeps_the_sqrt_dim_factor() {
        let mut out = [0.0f32; 2];
        l2_norm_row(&[3.0, 4.0], 1.0, &mut out);
        let s2 = std::f32::consts::SQRT_2;
        assert!((out[0] - 0.6 * s2).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.8 * s2).abs() < 1e-6, "{out:?}");
        assert!(
            (out[0] - 0.6).abs() > 0.2,
            "the sqrt(dim) factor is missing: {out:?} is the unit-L2 answer"
        );

        // `dim = 384`, the width the model actually uses: a row that already has
        // unit norm comes out with norm sqrt(384), i.e. the norm is deliberately
        // *not* preserved.
        let row = vec![1.0f32 / (DIM as f32).sqrt(); DIM];
        let mut out = vec![0.0f32; DIM];
        l2_norm_row(&row, 1.0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5, "{:?}", out[0]);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let want = (DIM as f32).sqrt();
        assert!(
            (norm - want).abs() < 1e-2,
            "norm {norm}, expected sqrt(dim) = {want}"
        );

        // The weight is a per-feature multiplier, applied last.
        let mut out = [0.0f32; 2];
        l2_norm_row(&[3.0, 4.0], 2.0, &mut out);
        assert!((out[0] - 0.6 * s2 * 2.0).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.8 * s2 * 2.0).abs() < 1e-6, "{out:?}");

        // An all-zero row stays zero instead of becoming NaN: the epsilon is a
        // floor on the norm, not a term inside a square root.
        let mut out = [0.0f32; 4];
        l2_norm_row(&[0.0; 4], 1.0, &mut out);
        assert!(out.iter().all(|v| *v == 0.0), "{out:?}");

        // Where the floor is the whole story: a row of denormals.
        // `max(||x||, 1e-12)` clamps the denominator, while an epsilon *inside*
        // the mean-square root would divide by ~1e-6 instead: six orders of
        // magnitude apart. This is the case a root-mean-square fast path gets
        // wrong even though it agrees on well-scaled rows.
        let mut out = [0.0f32; 4];
        l2_norm_row(&[1e-20; 4], 1.0, &mut out);
        assert!(out.iter().all(|&v| (v - 2e-8).abs() < 1e-9), "{out:?}");
        let rms_style = 1e-20f32 / (1e-40f32 + 1e-12).sqrt();
        assert!(
            (out[0] / rms_style).abs() > 1e4,
            "the epsilon's placement no longer matters: {out:?} vs {rms_style}"
        );
    }

    /// Same rule on the device, against the host reference, including the two
    /// rows the fast paths get wrong.
    #[test]
    fn l2norm_device_path_matches_the_formula() {
        let rows: Vec<[f32; 4]> = vec![
            [3.0, 4.0, 0.0, -2.0],
            [0.0, 0.0, 0.0, 0.0],
            [1e-20, 1e-20, 1e-20, 1e-20],
        ];
        let weight: [f32; 4] = [1.0, -0.5, 2.0, 0.25];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let x = Array::from_slice(&flat, &[1, rows.len() as i32, 4]);
        let w = Array::from_slice(&weight, &[4]);
        let norm = L2Norm::new(w, 4);
        assert!((norm.scale() - 2.0).abs() < 1e-6);

        let out = norm.forward(&x);
        let out = super::super::ensure_contiguous(&out);
        let got = out.as_slice::<f32>();
        assert_eq!(got.len(), flat.len());
        for (i, row) in rows.iter().enumerate() {
            let mut base = [0.0f32; 4];
            l2_norm_row(row, 1.0, &mut base);
            for j in 0..4 {
                let want = base[j] * weight[j];
                let have = got[i * 4 + j];
                assert!(
                    (have - want).abs() <= 1e-5f32.max(want.abs() * 1e-4),
                    "row {i} feature {j}: device {have}, host reference {want}"
                );
            }
        }
    }

    /// GELU as a formula, on the host: the exact-erf value at a few points, and
    /// how far the tanh approximation is from each of them.
    #[test]
    fn gelu_is_the_exact_erf_form() {
        assert_eq!(gelu_ref(0.0), 0.0);
        for &(x, want) in &[
            (1.0f64, 0.841_344_746_068_542_9),
            (-1.0, -0.158_655_253_931_457_1),
            (2.0, 1.954_499_736_103_641_6),
            (-2.0, -0.045_500_263_896_358_4),
            (0.5, 0.345_731_230_637_006_5),
            (-0.5, -0.154_268_769_362_993_5),
        ] {
            let exact = gelu_ref(x);
            assert!(
                (exact - want).abs() < 1e-6,
                "gelu({x}) = {exact}, reference {want}"
            );
            let approx = gelu_tanh(x);
            assert!(
                (exact - approx).abs() > 5e-6,
                "gelu({x}): the tanh approximation is indistinguishable ({approx}), \
                 which means this test stopped pinning the erf form"
            );
        }
        // The two limits any correct GELU has.
        assert!(gelu_ref(-6.0).abs() < 1e-8);
        assert!((gelu_ref(6.0) - 6.0).abs() < 1e-8);
    }

    /// The op the model runs, on the same points. The tensor library computes
    /// `erf` itself; this is the arm that would fail if the model silently
    /// switched to a cheaper approximation.
    #[test]
    fn gelu_device_path_matches_erf() {
        let xs = [0.0f32, 0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];
        let x = Array::from_slice(&xs, &[1, xs.len() as i32]);
        let y = gelu(&x);
        let y = super::super::ensure_contiguous(&y);
        let got = y.as_slice::<f32>();
        for (i, &v) in xs.iter().enumerate() {
            let want = gelu_ref(v as f64) as f32;
            assert!(
                (got[i] - want).abs() < 2e-5,
                "gelu({v}) on device = {}, host reference {want}",
                got[i]
            );
        }
    }

    /// `depth + 1` Linears with Tanh *between* them only: the last projection is
    /// raw, because the GLU that follows it is the nonlinearity. Asserted as a
    /// shape/behaviour property of the module the loader fills.
    #[test]
    fn mask_mlp_has_depth_plus_one_linears_and_a_glu() {
        let dim_in = band_dim_in(7);
        let shapes = mask_mlp_shapes(dim_in);
        assert_eq!(shapes.len(), MASK_MLP_LAYERS);
        assert_eq!(MASK_MLP_LAYERS, 2 + 1, "mask_estimator_depth = 2");
        // The GLU halves the width: 2·dim_in in, dim_in out.
        assert_eq!(shapes[MASK_MLP_LAYERS - 1].1, 2 * dim_in);

        // Behaviour on zeros: with zero weights every layer outputs its bias, so
        // a bias of 1 in both halves gives 1 · sigmoid(1) per feature, i.e. the
        // two halves are `v` and `g`, in that order, and `v` is not gated by a
        // Tanh applied after the last Linear.
        let mlp = MaskMlp {
            w: shapes
                .iter()
                .map(|&(i, o)| Array::zeros::<f32>(&[i as i32, o as i32]).expect("zeros"))
                .collect(),
            b: shapes
                .iter()
                .map(|&(_, o)| Array::from_slice(&vec![1.0f32; o], &[o as i32]))
                .collect(),
            dim_in,
        };
        let x = Array::zeros::<f32>(&[1, 1, DIM as i32]).expect("zeros");
        let out = mlp.forward(&x);
        assert_eq!(out.shape(), &[1, 1, dim_in as i32]);
        let out = super::super::ensure_contiguous(&out);
        let want = 1.0f32 * (1.0 / (1.0 + (-1.0f32).exp()));
        for &v in out.as_slice::<f32>() {
            assert!((v - want).abs() < 1e-5, "glu(1,1) = {want}, got {v}");
        }
    }

    /// Slow, deliberately boring reference for `erf`: the Abramowitz–Stegun
    /// 7.1.26 form, whose stated error is 1.5e-7 absolute. It is here to pin the
    /// *shape* of the activation (exact erf, not the tanh surrogate) at the 1e-5
    /// level, not to be the reference itself.
    fn erf_ref(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        sign * (1.0 - poly * (-x * x).exp())
    }

    fn gelu_ref(x: f64) -> f64 {
        0.5 * x * (1.0 + erf_ref(x / std::f32::consts::SQRT_2 as f64))
    }

    fn gelu_tanh(x: f64) -> f64 {
        let c = (2.0 / std::f64::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    }
}
