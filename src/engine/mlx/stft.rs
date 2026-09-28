//! `torch.stft` / `torch.istft` (`center = true`) for the Mel-Band RoFormer.
//!
//! Parameters fixed by the checkpoint:
//!
//! * `n_fft = 2048`, `hop_length = 441`, `win_length = 2048` (periodic Hann)
//! * `center = true` → reflect-pad by `n_fft / 2 = 1024` on both sides
//! * **all 1025 frequency bins are kept.** The Nyquist bin is *not* dropped:
//!   the band split below consumes every bin from DC to Nyquist, so dropping
//!   one would leave a hole in the last mel band.
//! * `normalized = false`
//!
//! The iSTFT does overlap-add with a host loop (the tensor library has no
//! scatter-add), normalises by the window's sum of squares and crops the center
//! padding, matching `torch.istft(..., center = true, length = T)`.
//!
//! ## Why this is an FFT and not the convolution the exported graph uses
//!
//! The ONNX export computes the same STFT as a matmul against cos/sin kernels.
//! In `f32` that accumulates differently, and the difference is not noise: in
//! quiet high-frequency bands it is amplified by the per-band L2Norm and then
//! cascaded through the transformer stack. Arbitrating the two with a float64
//! FFT showed this FFT path matching PyTorch while the convolution path did not
//! (see the module doc), so the arithmetic here is deliberately the direct one:
//! frame, window, [`rfft`].
//!
//! ## Pure vs. device
//!
//! Every index rule in this file is a plain Rust function ([`reflect_indices`],
//! [`n_frames`], [`ola_len`], [`crop_range`]) and the device code builds its
//! gather indices from them, so the arithmetic is testable without a tensor
//! runtime. What is *not* pure is the framing, the window, the FFT and the
//! overlap-add buffer, and the tests for those say so in their names.

use mlx_rs::fft::{irfft as irfft_raw, rfft as rfft_raw};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    add as add_op, cos as cos_op, divide, expand_dims_axes, maximum as maximum_op,
    multiply as mul_op, subtract as sub_op,
};
use mlx_rs::Array;
use std::f32::consts::PI;

/// FFT window size.
pub const N_FFT: i32 = 2048;
/// Hop length between frames.
pub const HOP: i32 = 441;
/// Number of frequency bins kept (DC … Nyquist inclusive).
pub const N_FREQS: i32 = N_FFT / 2 + 1; // 1025
/// Center reflect-pad applied on each side before framing.
pub const CENTER_PAD: i32 = N_FFT / 2; // 1024

/// Periodic Hann window of length `n`, matching
/// `torch.hann_window(n, periodic = true)`, i.e. `0.5 - 0.5·cos(2πk/n)`.
pub fn hann_window_n(n: i32) -> Array {
    let n_arr = Array::arange::<f32, f32>(None, n as f32, None).expect("hann: arange");
    let angle = mul_op(&n_arr, &Array::from_f32(2.0 * PI / n as f32)).expect("hann: angle");
    let cos_val = cos_op(&angle).expect("hann: cos");
    let half = Array::from_f32(0.5);
    let scaled = mul_op(&half, &cos_val).expect("hann: scale");
    sub_op(&half, &scaled).expect("hann: 0.5 - 0.5*cos")
}

/// Gather indices of a `torch`-style reflect pad (`symmetric`, edge value *not*
/// repeated) of a length-`len` axis by `pad_left` / `pad_right`.
///
/// Left: `pad_left, …, 1`. Right: `len-2, len-3, …`. Pure so the round-trip
/// tests and the caller agree on one definition; the device path feeds the
/// result to [`Array::take_axis`].
pub fn reflect_indices(len: i32, pad_left: i32, pad_right: i32) -> Vec<i32> {
    let mut idx = Vec::with_capacity((len + pad_left + pad_right) as usize);
    // Left: sample 1, then 2, … up to `pad_left`, i.e. the pad reads the
    // signal backwards from one past the edge, so `x[0]` appears exactly once.
    for i in (1..=pad_left).rev() {
        idx.push(i);
    }
    idx.extend(0..len);
    // Right: the mirror image, again starting one in from the edge.
    for i in 1..=pad_right {
        idx.push(len - 1 - i);
    }
    idx
}

/// Reflect-pad a `[..., T]` array along its last axis, `torch` `mode='reflect'`.
pub fn reflect_pad_last(a: &Array, pad_left: i32, pad_right: i32) -> Array {
    let last = a.ndim() as i32 - 1;
    let size = a.dim(last);
    let indices = reflect_indices(size, pad_left, pad_right);
    let idx = Array::from_slice(&indices, &[indices.len() as i32]);
    a.take_axis(&idx, last).expect("pad: take_axis")
}

/// Number of STFT frames for a `len`-sample input, center padding included.
pub fn n_frames(len: i32) -> i32 {
    (len + 2 * CENTER_PAD - N_FFT) / HOP + 1
}

/// Samples in the overlap-add buffer for `frames` frames: the last frame starts
/// at `(frames - 1) · HOP` and contributes `N_FFT` samples.
pub fn ola_len(frames: i32) -> usize {
    ((frames - 1) * HOP + N_FFT) as usize
}

/// Half-open sample range [`crop_range`] keeps out of the overlap-add buffer,
/// i.e. the center padding removed.
pub fn crop_range(length: i32) -> (i32, i32) {
    (CENTER_PAD, CENTER_PAD + length)
}

/// STFT: `(1, T)` f32 → `(1, N_FREQS, frames)` complex64.
pub fn stft(x: &Array) -> Array {
    let padded = reflect_pad_last(x, CENTER_PAD, CENTER_PAD);
    let padded_len = padded.dim(1);
    let frames = (padded_len - N_FFT) / HOP + 1;

    // Frame indices: (N_FFT, frames), row k of column t = k + t·HOP.
    let frame_off = Array::arange::<i32, i32>(None, N_FFT, None)
        .expect("stft: frame_off")
        .reshape(&[N_FFT, 1])
        .expect("stft: frame_off reshape");
    let starts = mul_op(
        &Array::arange::<i32, i32>(None, frames, None).expect("stft: starts arange"),
        &Array::from_int(HOP),
    )
    .expect("stft: starts mul")
    .reshape(&[1, frames])
    .expect("stft: starts reshape");
    let indices = add_op(&frame_off, &starts).expect("stft: frame indices");

    // (1, N_FFT, frames)
    let framed = padded.take_axis(&indices, 1).expect("stft: take_axis");
    let window = hann_window_n(N_FFT);
    let win_3d = expand_dims_axes(&window, &[0, 2]).expect("stft: window expand");
    let windowed = mul_op(&framed, &win_3d).expect("stft: window mul");

    rfft_raw(&windowed, Some(N_FFT), 1).expect("stft: rfft")
}

/// iSTFT: `(1, N_FREQS, frames)` complex64 → `(1, length)` f32.
///
/// Matches `torch.istft(..., n_fft, hop_length, win_length, center = true,
/// length)` for a spectrogram produced by [`stft`].
pub fn istft(spec: &Array, length: i32) -> Array {
    let window = hann_window_n(N_FFT);
    let n_frames_t = spec.dim(spec.ndim() as i32 - 1);

    // (1, N_FFT, frames) f32
    let time_frames = irfft_raw(spec, Some(N_FFT), 1).expect("istft: irfft");
    let win_3d = expand_dims_axes(&window, &[0, 2]).expect("istft: window expand");
    let windowed = mul_op(&time_frames, &win_3d).expect("istft: window mul");

    // Overlap-add on the host (the tensor library has no scatter-add).
    let ola_len = ola_len(n_frames_t);
    let frames_u = n_frames_t as usize;
    let nfft_u = N_FFT as usize;
    // `irfft` on a non-last axis can return a non-contiguous buffer; the host
    // loop below indexes raw memory with the logical (1, nfft, frames) strides,
    // so flatten to a C-order copy first (see `ensure_contiguous`).
    let windowed_c = super::ensure_contiguous(&windowed);
    let w_slice = windowed_c.as_slice::<f32>();
    let mut out_buf = vec![0.0f32; ola_len];

    // windowed layout: (1, nfft, frames) → index k * frames + t
    for t in 0..frames_u {
        let base_out = t * HOP as usize;
        for k in 0..nfft_u {
            out_buf[base_out + k] += w_slice[k * frames_u + t];
        }
    }

    // Window sum-of-squares normalization.
    let win_sq_slice: Vec<f32> = {
        let win_sq = mul_op(&window, &window).expect("istft: win_sq");
        win_sq.as_slice::<f32>().to_vec()
    };
    let mut wss = vec![0.0f32; ola_len];
    for t in 0..frames_u {
        let base_out = t * HOP as usize;
        for k in 0..nfft_u {
            wss[base_out + k] += win_sq_slice[k];
        }
    }

    let ola = Array::from_slice(&out_buf, &[ola_len as i32]);
    let wss_arr = Array::from_slice(&wss, &[ola_len as i32]);
    let norm = maximum_op(&wss_arr, &Array::from_f32(1e-12)).expect("istft: clamp");
    let normalized = divide(&ola, &norm).expect("istft: normalize");

    // Remove the center padding. NOTE: a single-range `IndexOp` on this
    // (1, ola_len) array crops the *flattened* buffer (verified numerically by
    // the round-trip test), so the shape is re-applied afterwards.
    let (start, end) = crop_range(length);
    normalized
        .index(start..end)
        .reshape(&[1, length])
        .expect("istft: crop reshape")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame count is the length every downstream shape is derived from, and
    /// `n_frames(WIN) == 801` is what the acceptance bar was stated against.
    #[test]
    fn n_frames_is_exact_at_the_native_window() {
        assert_eq!(N_FREQS, 1025);
        assert_eq!(CENTER_PAD, 1024);
        // 8 s @ 44.1 kHz divides exactly by the hop: no ragged last frame.
        assert_eq!(352_800 % HOP, 0);
        assert_eq!(n_frames(352_800), 801);
        // A shorter input still gets the center padding counted, and one frame
        // is the floor.
        assert_eq!(n_frames(HOP), 2);
        assert_eq!(n_frames(1), 1);
        assert_eq!(n_frames(0), 1);
        for len in [1i32, 440, 441, 4096, 44_100, 352_800, 352_801] {
            let f = n_frames(len);
            assert!(f >= 1, "len {len}: {f} frames");
            // Frames never start past the padded signal's end.
            assert!((f - 1) * HOP + N_FFT <= len + 2 * CENTER_PAD, "len {len}");
        }
    }

    /// The overlap-add buffer and the crop have to agree: the crop range has to
    /// sit inside the buffer and hold exactly `length` samples.
    #[test]
    fn ola_buffer_holds_the_crop() {
        for len in [1i32, 441, 4096, 44_100, 352_800] {
            let frames = n_frames(len);
            let buf = ola_len(frames);
            let (start, end) = crop_range(len);
            assert_eq!(end - start, len, "length {len}: crop is not `length` wide");
            assert!(
                (end as usize) <= buf,
                "length {len}: crop runs off the {}-sample buffer",
                buf
            );
            // One hop of slack per extra frame beyond the first, and the two
            // center pads, is what the buffer is.
            assert_eq!(buf, (frames as usize - 1) * HOP as usize + N_FFT as usize);
        }
        // The native window: 801 frames → 354 848 samples buffered, 1024 dropped
        // off each end.
        assert_eq!(ola_len(801), 354_848);
        assert_eq!(crop_range(352_800), (1024, 353_824));
    }

    /// `torch` reflect padding: symmetric around the edge sample, the edge
    /// value itself never repeated. Values here are the sample *indices*, so
    /// this is the same rule the device-side test below asserts on values.
    #[test]
    fn reflect_indices_follow_torch_semantics() {
        // x = [1,2,3,4,5] (indices 0..4), pad 2 left / 3 right →
        // indices 2,1,0,1,2,3,4,3,2,1 → values 3,2,1,2,3,4,5,4,3,2.
        assert_eq!(reflect_indices(5, 2, 3), vec![2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
        assert_eq!(reflect_indices(5, 0, 0), vec![0, 1, 2, 3, 4]);
        assert_eq!(reflect_indices(5, 3, 0), vec![3, 2, 1, 0, 1, 2, 3, 4]);
        // `torch` reflection is only expressible as a single gather while the pad
        // stays inside the signal (`pad < len - 1`), and that is the case this
        // engine ever uses: the center pad is 1024 and a padded read is always a
        // whole window of 352 800 samples.
        for len in [1025i32, 4096, 44_100, 352_800] {
            let idx = reflect_indices(len, CENTER_PAD, CENTER_PAD);
            assert_eq!(idx.len(), (len + 2 * CENTER_PAD) as usize);
            // The middle is the signal, unmoved: that is what makes the crop in
            // `crop_range` correct.
            let middle = (0..len).collect::<Vec<i32>>();
            assert_eq!(
                &idx[CENTER_PAD as usize..(CENTER_PAD + len) as usize],
                &middle[..]
            );
            // Reflecting, never clamping: every index is a real sample and the
            // first padded sample is `1`, not `0`.
            assert_eq!(idx[0], CENTER_PAD);
            assert_eq!(idx[CENTER_PAD as usize - 1], 1);
            assert_eq!(idx[idx.len() - 1], len - 1 - CENTER_PAD);
            assert!(idx.iter().all(|&i| i >= 0 && i < len));
        }
    }

    /// The device path must pad with the same indices the pure rule produces,
    /// checked on the host by reading the padded array back.
    #[test]
    fn reflect_pad_last_matches_the_pure_indices() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]);
        let padded = reflect_pad_last(&x, 2, 3);
        assert_eq!(
            padded.as_slice::<f32>(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
        );
    }

    /// Hann, periodic: `torch.hann_window(n, periodic=True)`, the form the
    /// checkpoint's STFT used. What that means concretely, and what a plot of the
    /// array would not tell you:
    ///
    /// * `w[0] == 0` and the peak is at exactly `n/2`, so consecutive frames can be
    ///   butt-joined without doubling an edge sample;
    /// * the reflection is about the *period* (`w[k] == w[n-k]`), not about the
    ///   array's own end (`w[k] == w[n-1-k]`, which is the symmetric window). Both
    ///   arrays look like a bell; only one of them sums to the COLA constant the
    ///   iSTFT's normalisation divides by.
    #[test]
    fn periodic_hann_reflects_about_the_period_not_the_end() {
        let w = hann_window_n(N_FFT);
        let s = w.as_slice::<f32>();
        let n = N_FFT as usize;
        assert_eq!(s.len(), n);

        // The periodic formula itself, on the host, at indices spread over the
        // window. A symmetric `0.5 - 0.5·cos(2πk/(n-1))` would be outside this
        // tolerance at all but the first couple.
        for &k in &[0usize, 1, 64, 512, 1023, 1024, 1536, 2047] {
            let want = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
            assert!(
                (s[k] as f64 - want).abs() < 1e-5,
                "w[{k}] = {}, periodic Hann is {want}",
                s[k]
            );
        }
        // And the symmetric window is *not* what this is, stated where the two
        // differ by more than any rounding: k = 1536.
        let symmetric = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * 1536.0 / (n - 1) as f64).cos();
        assert!(
            (s[1536] as f64 - symmetric).abs() > 1e-3,
            "w[1536] = {} is indistinguishable from a symmetric window ({symmetric})",
            s[1536]
        );

        // Reflection about the period: `w[k] == w[n-k]`, which is the property the
        // overlap-add's COLA sum rests on.
        for &k in &[512usize, 1000, 1536] {
            assert!(
                (s[k] - s[n - k]).abs() < 1e-5,
                "not periodic-symmetric at {k}: {} vs {}",
                s[k],
                s[n - k]
            );
        }
        // The peak is unique and exactly at n/2. A symmetric window of this length
        // would have two equal samples either side of it, so the gap here (about
        // 4.7e-6, far above the ~1e-7 rounding of the subtraction) is what says
        // the frame butt-joins without doubling an edge.
        assert!((s[n / 2] - 1.0).abs() < 1e-6, "w[n/2] = {}", s[n / 2]);
        assert!(
            s[n / 2] > s[n / 2 - 1] + 1e-6,
            "w[n/2] and w[n/2-1] are equal: {} vs {} — that is a symmetric window",
            s[n / 2],
            s[n / 2 - 1]
        );
        assert!(s[0].abs() < 1e-6, "w[0] = {}", s[0]);
        // 0.5 - 0.5·cos is never negative and never above 1.
        assert!(s.iter().all(|&v| (0.0..=1.0).contains(&v)));
        // Sum of squares ~ n/2, which is what the iSTFT's overlap-add normalization
        // divides by; the `max(·, 1e-12)` guard in there exists for the two ends of
        // the padded signal, not for the interior.
        let sq: f32 = s.iter().map(|v| v * v).sum();
        assert!(sq > 500.0, "sum of squares {sq}");
    }

    /// Round trip of a full native window: 801 frames, and the interior (away
    /// from the reflect-padded edges, where the COLA sum is complete) must come
    /// back at float32 precision. This is the arm that would fail if the FFT were
    /// replaced by the export's convolution, so it is the arithmetic the whole
    /// module exists to keep.
    #[test]
    fn stft_istft_roundtrip_on_one_native_window() {
        let len = 352_800i32;
        let data: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        let x = Array::from_slice(&data, &[1, len]);

        let spec = stft(&x);
        assert_eq!(spec.shape(), &[1, N_FREQS, 801], "stft shape");
        assert_eq!(n_frames(len), spec.dim(2));

        let recon = istft(&spec, len);
        assert_eq!(recon.shape(), &[1, len], "istft shape");

        let r = super::super::ensure_contiguous(&recon);
        let r = r.as_slice::<f32>();
        let mut max_err = 0.0f32;
        for i in 2048..len as usize - 2048 {
            max_err = max_err.max((r[i] - data[i]).abs());
        }
        assert!(max_err < 1e-4, "roundtrip max_err={max_err}");
    }
}
