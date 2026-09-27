//! Mel-Band RoFormer vocal separation on MLX (Apple Silicon, `feature = "mlx"`).
//!
//! A from-scratch Rust implementation of the architecture: its own STFT, its own
//! mel-band split, alternating time/frequency transformer blocks, its own mask
//! estimator. There is no exported graph being executed here — which is the point,
//! and the reason this arm runs the checkpoint's native window (8 s, 352 800
//! samples at 44.1 kHz) with nothing cut or re-exported. The I/O path around it is
//! the other half of the story: one window resident, both stems streamed to disk,
//! a checkpoint at every window boundary.
//!
//! ## Acceptance: against a PyTorch FFT reference, not against the ONNX arm
//!
//! The bar this port is accepted at is the trained model's own output, computed by
//! PyTorch with an FFT-based STFT. It must not be quietly swapped for "does the MLX
//! arm agree with the ONNX arm", because that comparison cancels out the very error
//! this engine exists to avoid:
//!
//! The shipped ONNX export computes its STFT as a convolution against cos/sin
//! kernels. In float32 that accumulates differently, and in quiet high-frequency
//! bands the difference is not noise — it is amplified by the per-band L2Norm and
//! then cascaded through the transformer stack. Arbitrating the two against a
//! float64 FFT showed this FFT path matching PyTorch at correlation 0.9999997 while
//! the exported graph deviated from the model it came from (correlation ≈ 0.70). So
//! agreement with the ONNX graph's waveform would be evidence of nothing: it would
//! say only that two implementations agree with each other, which after this
//! measurement is the suspicious case. Comparing against PyTorch's own reference is
//! the only test that can distinguish "my arithmetic is the trained model's" from
//! "my arithmetic reproduces the export's error".
//!
//! The gate that runs that comparison is
//! [`one_window_matches_the_torch_fft_reference`](separate::MlxEngine) in
//! `separate.rs`. It needs three things this repository does not carry — the weight
//! file, a real recording, and a reference dump — so it reads them from
//! `RR_MLX_WEIGHTS`, `RR_TEST_WAV` and `RR_TORCH_REF` and prints `[SKIP]` when any
//! is unset. The arms that need only weights — streaming equivalence, and
//! resume-after-a-kill bit identity — are gated the same way and additionally
//! `#[ignore]`d, because mapping the file costs ~1 GB of resident memory and
//! several minutes of forwards; `cargo test --features mlx --lib -- --ignored
//! --test-threads=1` is what runs them.
//!
//! ## Numerical conventions that must be preserved
//!
//! Each of these was found by being wrong without it. The tests named beside them
//! are in this module; an asterisk means the convention is only documented, because
//! nothing short of a forward against the trained model can see it.
//!
//! * L2Norm is `x / max(||x||, 1e-12) * sqrt(dim) * weight` — **not** a
//!   root-mean-square norm and not a plain "divide by the norm". The second drops
//!   the `sqrt(dim)`, which rescales every activation by `1/sqrt(384)`; the first
//!   moves where `eps` applies, which differs by six orders of magnitude on a row of
//!   denormals. `l2norm_keeps_the_sqrt_dim_factor`,
//!   `l2norm_device_path_matches_the_formula`.
//! * GELU is the exact erf form, `0.5x(1 + erf(x/√2))`. The tanh approximation is
//!   within ~1.5e-4 per element and the stack amplifies it rather than averaging it
//!   away. `gelu_is_the_exact_erf_form`, `gelu_device_path_matches_erf`.
//! * RoPE is `fast::rope(traditional = true, base = 10000, dims = 64)`, which is the
//!   `repeat_interleave` plus pairwise `rotate_half` semantics of the reference
//!   implementation. `traditional = false` or another base trains nothing and
//!   separates garbage. \*
//! * Attention is a manual matmul plus softmax with the `1/sqrt(dim_head)` scale
//!   applied before the max subtraction. The fused scaled-dot-product variant
//!   diverges in practice, and only a parity run against the model shows it. \*
//! * Every transformer block ends with an L2Norm output norm, so all twelve
//!   `layers_{i}.{time,freq}_transformer.norm.weight` tensors are consumed. There is
//!   **no** model-level final norm: the reference implementation bypassed it and the
//!   exported graph has none either, so adding one here would be inventing a layer.
//!   `weight_file_keys_are_exactly_the_ones_the_loader_expects`, `key_naming_is_exact`.
//! * The mask estimator MLP has `depth + 1` Linears — torch semantics — i.e. 384 →
//!   1536 → 1536 → `2 · dim_in` with Tanh *between* them and a GLU folding the last
//!   pair of halves. `mask_mlp_has_depth_plus_one_linears_and_a_glu`,
//!   `mask_mlp_layer_indices_are_the_even_slots`.
//! * Feature layout is `(f, channel, re/im)` interleaved: feature index
//!   `j = f * 4 + ch * 2 + ri` for stereo complex. `feature_layout_is_the_documented_interleaving`,
//!   `gather_indices_encode_the_layout_for_every_band_slot`.
//!
//! ## Architecture, fixed by the checkpoint
//!
//! `dim = 384`, `depth = 6`, `time_transformer_depth = freq_transformer_depth = 1`,
//! `heads = 8`, `dim_head = 64`, `num_bands = 60`, stereo in and out,
//! `num_stems = 1` (the background stem is the residual `mix − vocals`),
//! `mask_estimator_depth = 2`, `n_fft = 2048`, `hop = 441`, `win = 2048`, and one
//! 352 800-sample window with a 110 250-sample overlap crossfaded linearly. The band
//! split's 60 mel bands, their bin counts and their overlap are derived from the
//! Slaney filterbank in [`model::band_layout`], not stored.
//!
//! ## Building this feature is not the same as building the crate
//!
//! Read this before trying `--features mlx`. Measured on the machine this port was
//! written on, against the `mlx-rs` 0.25 dependency the manifest declares:
//!
//! * `cargo check --no-default-features` and `cargo test --no-default-features` —
//!   the default, portable build. This module is not compiled, and neither is its
//!   test list. Green, and that is the state a clean clone starts in.
//! * `cargo check --no-default-features --features mlx` — **fails**, and it fails
//!   before any of this crate's code is reached: `mlx-rs` depends on `mlx-sys`, whose
//!   build script runs CMake over the C API and whose `CMakeLists.txt` fetches the
//!   MLX C++ library from source (an upstream `FetchContent` of the MLX repository at
//!   the matching tag). So the command needs a C++ toolchain, CMake, network access
//!   to clone and build MLX itself, and then a long compile. On this machine the
//!   attempt ended in `error: failed to run custom build command for mlx-sys v0.2.0`
//!   with a CMake "no download info given for `mlx-populate`" complaint, because a
//!   locally patched copy of the crate in the registry cache pointed the fetch at a
//!   private source directory belonging to another project; a clean clone instead
//!   reaches the upstream path and tries to clone MLX. Both are the same finding:
//!   **this feature does not build from a clean checkout without provisioning
//!   MLX.**
//! * What a user has to install, in short: macOS on Apple Silicon, CMake, a clang
//!   that can target Metal, and either (a) let the build fetch and compile MLX, or
//!   (b) an existing MLX install — the dylib and its headers — with the binding crate
//!   configured to link it. Route (b) is what the application this was ported from
//!   does, and it is a vendored arrangement that cannot be expressed from inside a
//!   published crate without changing its manifest, which is outside what this module
//!   owns.
//!
//! What *was* verified, and how: this module was type-checked, and every test in it
//! run, in a scratch copy of the crate whose manifest pointed `mlx-sys` at an MLX
//! install that already existed on the machine — the harness is not something this
//! repository provides, and its results are reported beside the default build rather
//! than folded into it. The feature is never quietly disabled to make a build look
//! green: with `--features mlx` on an unprovisioned machine the command fails in the
//! C++ build, and that failure is the honest answer.
//!
//! ## Module map
//!
//! * [`model`] — the graph: band split, alternating transformer, mask estimator, and
//!   the pure layout/band arithmetic the forward is built from.
//! * [`stft`] — framing, periodic Hann, `rfft` / `irfft`, host overlap-add.
//! * [`weights`] — the 672-tensor `.safetensors` contract: key naming, the
//!   `[out, in] → [in, out]` transpose, shape checks.
//! * [`separate`] — [`MlxEngine`], the window loop, crossfade, checkpoints, resume.

pub mod model;
pub mod separate;
pub mod stft;
pub mod weights;

use mlx_rs::Array;

pub use separate::{resolve_overlap, resolve_window, MlxEngine, OVERLAP, WIN};

/// Flatten `a` into a row-major (C-contiguous) 1-D buffer and make sure it is
/// computed, so the host can read it.
///
/// MLX ops may return arrays whose buffer layout differs from their logical
/// shape — `rfft`/`irfft` on a non-last axis keep the FFT kernel's transposed
/// layout, for one. Device-side ops handle strides transparently, but a
/// host-side `as_slice` reads raw memory in C-order and silently misinterprets a
/// strided buffer. Call this (and use the returned array) before any host read.
///
/// The result is 1-D: callers index the raw slice with the strides of `a`'s
/// logical shape, as the overlap-add loop in [`stft`] and the output read in
/// [`separate`] do. Reshaping back to the original rank must *not* be added:
/// `reshape(reshape(x, -1), s)` fuses back into a view of the strided buffer,
/// which defeats the copy this exists to force.
///
/// Reshaping a non-row-contiguous array cannot be expressed as a view, so the
/// reshape below materializes a genuine C-order copy; for an array that is
/// already contiguous it is a cheap view and the `eval` merely computes it. A
/// primitive elementwise op such as `x * 1.0` is *not* a substitute: its output
/// can inherit the source's strided buffer through donation and still read
/// scrambled on the host.
pub(crate) fn ensure_contiguous(a: &Array) -> Array {
    let flat = a
        .reshape(&[-1])
        .expect("ensure_contiguous: flatten to C order");
    // Materialize now, so the caller's `as_slice` sees the normalized buffer
    // rather than a lazy graph node.
    mlx_rs::transforms::eval([&flat]).expect("ensure_contiguous: eval");
    flat
}
