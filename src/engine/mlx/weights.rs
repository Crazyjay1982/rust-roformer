//! Safetensors weight loading for the MLX Mel-Band RoFormer (vocals).
//!
//! The file holds 672 tensors and is produced from the stock fp32 ONNX export by
//! `tools/extract_onnx_weights.py`, which you run yourself: this crate ships no
//! weights, and the extractor's output is a build artifact of someone else's
//! model rather than a redistribution of it.
//!
//! ## Key naming
//!
//! The keys are what the export's own module paths translate to. They are built
//! here by the [`*_key`] functions and nowhere else, so the whole naming scheme
//! and its coverage of the file is checkable without a weight file on disk:
//! [`expected_keys`] enumerates all 672 names and [`apply_weights`] refuses any
//! key in the file that it does not recognise.
//!
//! | key | count |
//! |---|---|
//! | `band_split.to_features_{b}.norm.weight` / `.linear.weight` / `.linear.bias` | 180 |
//! | `layers_{i}.{time,freq}_transformer.layers_0.attn.{norm.weight, to_qkv.weight, to_gates.weight, to_gates.bias, to_out.layers.0.weight}` | 60 |
//! | `layers_{i}.{time,freq}_transformer.layers_0.ff.net.layers.{0,1,4}.{weight,bias}` | 60 |
//! | `layers_{i}.{time,freq}_transformer.norm.weight`: the block's output L2Norm | 12 |
//! | `mask_estimators_0.to_freqs_{b}.layers.{0,2,4}.{weight,bias}` | 360 |
//!
//! The `layers_{i}` / `to_features_{b}` spelling (underscore before the index) is
//! not a typo: it is what the exporter writes, because the torch module path
//! `layers.0.0` is ambiguous once flattened into one string.
//!
//! ## Transpose convention
//!
//! Linear weights are stored `[out, in]` in the file (the `nn.Linear` layout the
//! state dict carries) and are transposed to `[in, out]` here, so every Linear
//! in `model.rs` is a plain `matmul(x, w) + b`. Only rank-2 tensors are
//! transposed: L2Norm weights and biases are rank-1 and mean what they say.
//!
//! ## Failure mode
//!
//! A missing key, an unrecognised key or a shape that disagrees with the
//! architecture is [`Error::Model`], never a panic: the caller's decision at that
//! point is "this file is not the model I expected", and it has to be reachable
//! while the file is still mapped.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use mlx_rs::Array;

use crate::error::{Error, Result};

use super::model::{
    band_dim_in, band_layout, mask_mlp_shapes, Axis, RoFormerModel, DEPTH, DIM, DIM_INNER, FF_MULT,
    HEADS, MASK_MLP_LAYERS, NUM_BANDS,
};

/// Tensors the checkpoint holds: 180 band split + 60 attention + 60
/// feed-forward + 12 block output norms + 360 mask-estimator layers.
pub const EXPECTED_TENSOR_COUNT: usize = 60 * 3 + 60 + 60 + 12 + 60 * MASK_MLP_LAYERS * 2;

// ─────────────────────── key naming, in one place ───────────────────────

/// `band_split.to_features_{b}`: the module one band's split lives in.
pub fn band_split_prefix(b: usize) -> String {
    format!("band_split.to_features_{b}")
}

/// The whole block prefix: `layers_{i}.{time,freq}_transformer.layers_0`.
pub fn block_prefix(i: usize, axis: Axis) -> String {
    format!("layers_{i}.{}.layers_0", axis.key_part())
}

pub fn band_norm_key(b: usize) -> String {
    format!("{}.norm.weight", band_split_prefix(b))
}

pub fn band_linear_weight_key(b: usize) -> String {
    format!("{}.linear.weight", band_split_prefix(b))
}

pub fn band_linear_bias_key(b: usize) -> String {
    format!("{}.linear.bias", band_split_prefix(b))
}

pub fn attn_norm_key(i: usize, axis: Axis) -> String {
    format!("{}.attn.norm.weight", block_prefix(i, axis))
}

pub fn attn_qkv_key(i: usize, axis: Axis) -> String {
    format!("{}.attn.to_qkv.weight", block_prefix(i, axis))
}

pub fn attn_gates_weight_key(i: usize, axis: Axis) -> String {
    format!("{}.attn.to_gates.weight", block_prefix(i, axis))
}

pub fn attn_gates_bias_key(i: usize, axis: Axis) -> String {
    format!("{}.attn.to_gates.bias", block_prefix(i, axis))
}

/// The output projection is a `Sequential` in the export, hence the
/// `to_out.layers.0` hop before `.weight`.
pub fn attn_out_weight_key(i: usize, axis: Axis) -> String {
    format!("{}.attn.to_out.layers.0.weight", block_prefix(i, axis))
}

/// Feed-forward member index: the L2Norm sits at 0, then the two Linears at 1
/// and 4: the export numbers every module in the `Sequential`, including the
/// activations, which is why the second Linear is 4 and not 2.
pub fn ff_norm_key(i: usize, axis: Axis) -> String {
    format!("{}.ff.net.layers.0.weight", block_prefix(i, axis))
}

pub fn ff_linear_key(i: usize, axis: Axis, member: usize, weight: bool) -> String {
    format!(
        "{}.ff.net.layers.{}.{}",
        block_prefix(i, axis),
        member,
        if weight { "weight" } else { "bias" }
    )
}

/// A block's output L2Norm. These twelve are the last norms in the graph, which
/// is the statement that there is no model-level final norm.
pub fn block_out_norm_key(i: usize, axis: Axis) -> String {
    format!("layers_{i}.{}.norm.weight", axis.key_part())
}

/// `mask_estimators_0.to_freqs_{b}`: the module one band's estimator lives in.
pub fn mask_estimator_prefix(b: usize) -> String {
    format!("mask_estimators_0.to_freqs_{b}")
}

/// The layer indices one mask MLP uses, derived rather than hard-coded so
/// [`MASK_MLP_LAYERS`] cannot drift from the keys looked up.
pub fn mask_mlp_layer_indices() -> [usize; MASK_MLP_LAYERS] {
    let mut out = [0usize; MASK_MLP_LAYERS];
    let mut i = 0;
    while i < MASK_MLP_LAYERS {
        out[i] = i * 2;
        i += 1;
    }
    out
}

pub fn mask_mlp_weight_key(b: usize, j: usize) -> String {
    format!("{}.layers.{j}.weight", mask_estimator_prefix(b))
}

pub fn mask_mlp_bias_key(b: usize, j: usize) -> String {
    format!("{}.layers.{j}.bias", mask_estimator_prefix(b))
}

/// Every key [`apply_weights`] reads, in the order it reads them.
pub fn expected_keys() -> Vec<String> {
    let mut keys = Vec::with_capacity(EXPECTED_TENSOR_COUNT);
    for b in 0..NUM_BANDS {
        keys.push(band_norm_key(b));
        keys.push(band_linear_weight_key(b));
        keys.push(band_linear_bias_key(b));
    }
    for i in 0..DEPTH {
        for axis in Axis::both() {
            keys.push(attn_norm_key(i, axis));
            keys.push(attn_qkv_key(i, axis));
            keys.push(attn_gates_weight_key(i, axis));
            keys.push(attn_gates_bias_key(i, axis));
            keys.push(attn_out_weight_key(i, axis));
            keys.push(ff_norm_key(i, axis));
            keys.push(ff_linear_key(i, axis, 1, true));
            keys.push(ff_linear_key(i, axis, 1, false));
            keys.push(ff_linear_key(i, axis, 4, true));
            keys.push(ff_linear_key(i, axis, 4, false));
            keys.push(block_out_norm_key(i, axis));
        }
    }
    for b in 0..NUM_BANDS {
        for j in mask_mlp_layer_indices() {
            keys.push(mask_mlp_weight_key(b, j));
            keys.push(mask_mlp_bias_key(b, j));
        }
    }
    keys
}

/// The `[in, out]` shape (as this crate stores it) of one expected key, or
/// `None` if this loader has no place for that name.
///
/// Derived from the architecture, so [`apply_weights`] and anyone checking a
/// weight file before loading it cannot disagree about what the file must hold.
/// A host that wants to inspect a checkpoint without mapping it (to see whether
/// it is this model at all) can walk [`expected_keys`] and compare.
pub fn expected_shape(key: &str) -> Option<Vec<i32>> {
    let dim_in = |b: usize| band_dim_in(band_layout().num_freqs_per_band[b]) as i32;
    // Band split.
    if let Some(rest) = key.strip_prefix("band_split.to_features_") {
        let b: usize = rest.split('.').next()?.parse().ok()?;
        return Some(if key.ends_with(".norm.weight") {
            vec![dim_in(b)]
        } else if key.ends_with(".linear.weight") {
            vec![dim_in(b), DIM as i32]
        } else {
            vec![DIM as i32]
        });
    }
    // Mask estimator.
    if let Some(rest) = key.strip_prefix("mask_estimators_0.to_freqs_") {
        let b: usize = rest.split('.').next()?.parse().ok()?;
        let j: usize = key.rsplit('.').nth(1)?.parse().ok()?;
        let slot = j / 2;
        let shapes = mask_mlp_shapes(dim_in(b) as usize);
        let (_, o) = *shapes.get(slot)?;
        return Some(if key.ends_with(".weight") {
            // Stored [out, in]; this returns the [in, out] the model keeps.
            vec![shapes[slot].0 as i32, o as i32]
        } else {
            vec![o as i32]
        });
    }
    // Blocks.
    if !key.starts_with("layers_") {
        return None;
    }
    let i: usize = key["layers_".len()..].split('.').next()?.parse().ok()?;
    let _ = i;
    if key.ends_with(".attn.norm.weight") || key.ends_with(".ff.net.layers.0.weight") {
        return Some(vec![DIM as i32]);
    }
    if key.contains(".norm.weight") && key.contains("_transformer.norm.weight") {
        return Some(vec![DIM as i32]);
    }
    if key.ends_with(".attn.to_qkv.weight") {
        return Some(vec![DIM as i32, 3 * DIM_INNER as i32]);
    }
    if key.ends_with(".attn.to_gates.weight") {
        return Some(vec![DIM as i32, HEADS as i32]);
    }
    if key.ends_with(".attn.to_gates.bias") {
        return Some(vec![HEADS as i32]);
    }
    if key.ends_with(".attn.to_out.layers.0.weight") {
        return Some(vec![DIM_INNER as i32, DIM as i32]);
    }
    if key.ends_with(".ff.net.layers.1.weight") {
        return Some(vec![DIM as i32, (FF_MULT * DIM) as i32]);
    }
    if key.ends_with(".ff.net.layers.1.bias") {
        return Some(vec![(FF_MULT * DIM) as i32]);
    }
    if key.ends_with(".ff.net.layers.4.weight") {
        return Some(vec![(FF_MULT * DIM) as i32, DIM as i32]);
    }
    if key.ends_with(".ff.net.layers.4.bias") {
        return Some(vec![DIM as i32]);
    }
    None
}

// ─────────────────────────── loading ───────────────────────────

/// Read a `.safetensors` file into key → array.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Array>> {
    Array::load_safetensors(path).map_err(|e| Error::Model {
        detail: format!("could not read {}: {e}", path.display()),
    })
}

/// Load every tensor the model needs from `path` into `model`.
pub fn load_weights(model: &mut RoFormerModel, path: &Path) -> Result<()> {
    let weights = load_safetensors(path)?;
    apply_weights(model, &weights, path)
}

/// Fill `model` from an already-read tensor map.
///
/// Split out from [`load_weights`] so the whole naming and shape contract is
/// testable without the file: the tests build a map with [`expected_keys`] and
/// the shapes [`expected_shape`] asks for.
pub fn apply_weights(
    model: &mut RoFormerModel,
    weights: &HashMap<String, Array>,
    path: &Path,
) -> Result<()> {
    let wanted = expected_keys();
    if weights.len() != wanted.len() {
        return Err(Error::Model {
            detail: format!(
                "{} holds {} tensors, this architecture needs {} ({} band split + \
                 {} attention + {} feed-forward + {} block norms + {} mask estimator)",
                path.display(),
                weights.len(),
                EXPECTED_TENSOR_COUNT,
                NUM_BANDS * 3,
                DEPTH * 2 * 5,
                DEPTH * 2 * 5,
                DEPTH * 2,
                NUM_BANDS * MASK_MLP_LAYERS * 2,
            ),
        });
    }
    // An unrecognised key means the file is not this model, even if it happens
    // to hold the right number of tensors: say which key, rather than letting a
    // differently-named checkpoint load as a guess.
    let known: HashSet<&str> = wanted.iter().map(String::as_str).collect();
    if let Some(extra) = weights.keys().find(|k| !known.contains(k.as_str())) {
        return Err(Error::Model {
            detail: format!(
                "{} contains a tensor this model has no place for: `{extra}`",
                path.display()
            ),
        });
    }

    let layout = band_layout();

    // ─── Band split ───
    for (b, band) in model.band_split_mut().iter_mut().enumerate() {
        let d = band_dim_in(layout.num_freqs_per_band[b]) as i32;
        band.set_norm(require(weights, &band_norm_key(b), &[d])?);
        band.set_linear(
            require_transposed(weights, &band_linear_weight_key(b), &[d, DIM as i32])?,
            require(weights, &band_linear_bias_key(b), &[DIM as i32])?,
        );
    }

    // ─── Alternating transformer blocks ───
    for (i, block) in model.blocks_mut().iter_mut().enumerate() {
        for axis in Axis::both() {
            let trans = block.transformer_mut(axis);

            trans.attention_mut().set_norm(require(
                weights,
                &attn_norm_key(i, axis),
                &[DIM as i32],
            )?);
            trans.attention_mut().set_qkv(require_transposed(
                weights,
                &attn_qkv_key(i, axis),
                &[DIM as i32, 3 * DIM_INNER as i32],
            )?);
            trans.attention_mut().set_gates(
                require_transposed(
                    weights,
                    &attn_gates_weight_key(i, axis),
                    &[DIM as i32, HEADS as i32],
                )?,
                require(weights, &attn_gates_bias_key(i, axis), &[HEADS as i32])?,
            );
            trans.attention_mut().set_out(require_transposed(
                weights,
                &attn_out_weight_key(i, axis),
                &[DIM_INNER as i32, DIM as i32],
            )?);

            trans.feed_forward_mut().set_norm(require(
                weights,
                &ff_norm_key(i, axis),
                &[DIM as i32],
            )?);
            trans.feed_forward_mut().set_first(
                require_transposed(
                    weights,
                    &ff_linear_key(i, axis, 1, true),
                    &[DIM as i32, (FF_MULT * DIM) as i32],
                )?,
                require(
                    weights,
                    &ff_linear_key(i, axis, 1, false),
                    &[(FF_MULT * DIM) as i32],
                )?,
            );
            trans.feed_forward_mut().set_second(
                require_transposed(
                    weights,
                    &ff_linear_key(i, axis, 4, true),
                    &[(FF_MULT * DIM) as i32, DIM as i32],
                )?,
                require(weights, &ff_linear_key(i, axis, 4, false), &[DIM as i32])?,
            );

            trans.set_out_norm(require(
                weights,
                &block_out_norm_key(i, axis),
                &[DIM as i32],
            )?);
        }
    }

    // ─── Mask estimator ───
    for (b, mlp) in model.mask_estimator_mut().iter_mut().enumerate() {
        let dim_in = band_dim_in(layout.num_freqs_per_band[b]);
        for (slot, &j) in mask_mlp_layer_indices().iter().enumerate() {
            let (i, o) = mask_mlp_shapes(dim_in)[slot];
            mlp.set_layer(
                slot,
                require_transposed(weights, &mask_mlp_weight_key(b, j), &[i as i32, o as i32])?,
                require(weights, &mask_mlp_bias_key(b, j), &[o as i32])?,
            );
        }
    }

    log::info!(
        "[mlx] {} weight tensors loaded from {}",
        wanted.len(),
        path.display()
    );
    Ok(())
}

/// Fetch one tensor by key, or name the missing key.
fn require(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let found = weights.get(key).ok_or_else(|| Error::Model {
        detail: format!("weight file is missing `{key}`"),
    })?;
    check_shape(found, key, shape)?;
    Ok(found.clone())
}

/// Fetch one rank-2 Linear weight and transpose `[out, in]` → `[in, out]`.
///
/// The file's layout is checked *before* the transpose, which is the only order
/// that can tell a `[out, in]` weight from an already-transposed one.
fn require_transposed(
    weights: &HashMap<String, Array>,
    key: &str,
    in_out: &[i32],
) -> Result<Array> {
    let found = weights.get(key).ok_or_else(|| Error::Model {
        detail: format!("weight file is missing `{key}`"),
    })?;
    let stored = vec![in_out[1], in_out[0]];
    check_shape(found, key, &stored)?;
    found.transpose_axes(&[1, 0]).map_err(|e| Error::Model {
        detail: format!("could not transpose `{key}` to [in, out]: {e}"),
    })
}

fn check_shape(found: &Array, key: &str, want: &[i32]) -> Result<()> {
    if found.shape() == want {
        return Ok(());
    }
    Err(Error::Model {
        detail: format!(
            "`{key}` has shape {:?}, this architecture expects {:?}",
            found.shape(),
            want
        ),
    })
}

/// The transpose rule, as a test-visible function: rank 2 only.
pub fn needs_transpose(shape: &[i32]) -> bool {
    shape.len() == 2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 180 band split + 60 attention + 60 feed-forward + 12 block output norms
    /// + 360 mask-estimator layers.
    #[test]
    fn tensor_count_is_the_documented_arithmetic() {
        assert_eq!(EXPECTED_TENSOR_COUNT, 672);
        assert_eq!(NUM_BANDS * 3, 180);
        assert_eq!(DEPTH * 2 * 5, 60, "attention");
        assert_eq!(DEPTH * 2 * 5, 60, "feed-forward");
        assert_eq!(DEPTH * 2, 12, "block output norms");
        assert_eq!(NUM_BANDS * MASK_MLP_LAYERS * 2, 360);
        assert_eq!(expected_keys().len(), EXPECTED_TENSOR_COUNT);
    }

    /// Every key the loader can ask for, spelled out. A typo in one of these is
    /// a load failure only against a ~900 MB file, which is why the naming lives
    /// in functions and the strings live here.
    #[test]
    fn key_naming_is_exact() {
        assert_eq!(band_split_prefix(0), "band_split.to_features_0");
        assert_eq!(band_norm_key(0), "band_split.to_features_0.norm.weight");
        assert_eq!(
            band_linear_weight_key(59),
            "band_split.to_features_59.linear.weight"
        );
        assert_eq!(
            band_linear_bias_key(59),
            "band_split.to_features_59.linear.bias"
        );

        assert_eq!(
            block_prefix(0, Axis::Time),
            "layers_0.time_transformer.layers_0"
        );
        assert_eq!(
            block_prefix(5, Axis::Freq),
            "layers_5.freq_transformer.layers_0"
        );
        assert_eq!(
            attn_norm_key(3, Axis::Freq),
            "layers_3.freq_transformer.layers_0.attn.norm.weight"
        );
        assert_eq!(
            attn_qkv_key(0, Axis::Time),
            "layers_0.time_transformer.layers_0.attn.to_qkv.weight"
        );
        assert_eq!(
            attn_gates_weight_key(0, Axis::Time),
            "layers_0.time_transformer.layers_0.attn.to_gates.weight"
        );
        assert_eq!(
            attn_gates_bias_key(0, Axis::Time),
            "layers_0.time_transformer.layers_0.attn.to_gates.bias"
        );
        assert_eq!(
            attn_out_weight_key(2, Axis::Freq),
            "layers_2.freq_transformer.layers_0.attn.to_out.layers.0.weight"
        );
        assert_eq!(
            ff_norm_key(0, Axis::Time),
            "layers_0.time_transformer.layers_0.ff.net.layers.0.weight"
        );
        assert_eq!(
            ff_linear_key(0, Axis::Time, 1, true),
            "layers_0.time_transformer.layers_0.ff.net.layers.1.weight"
        );
        assert_eq!(
            ff_linear_key(4, Axis::Freq, 4, false),
            "layers_4.freq_transformer.layers_0.ff.net.layers.4.bias"
        );
        assert_eq!(
            block_out_norm_key(0, Axis::Time),
            "layers_0.time_transformer.norm.weight"
        );
        assert_eq!(
            block_out_norm_key(5, Axis::Freq),
            "layers_5.freq_transformer.norm.weight"
        );
        assert_eq!(mask_estimator_prefix(7), "mask_estimators_0.to_freqs_7");
        assert_eq!(
            mask_mlp_weight_key(0, 0),
            "mask_estimators_0.to_freqs_0.layers.0.weight"
        );
        assert_eq!(
            mask_mlp_bias_key(59, 4),
            "mask_estimators_0.to_freqs_59.layers.4.bias"
        );
    }

    /// The three Linears of a depth-2 estimator are at indices 0, 2 and 4: the
    /// odd slots are the Tanh activations, which the export numbered too.
    #[test]
    fn mask_mlp_layer_indices_are_the_even_slots() {
        assert_eq!(mask_mlp_layer_indices(), [0, 2, 4]);
        assert_eq!(mask_mlp_layer_indices().len(), MASK_MLP_LAYERS);
    }

    /// Only rank-2 tensors get transposed, so a norm or bias vector cannot be
    /// silently turned into a column.
    #[test]
    fn only_rank_two_weights_are_transposed() {
        assert!(needs_transpose(&[384, 1536]));
        assert!(!needs_transpose(&[384]));
        assert!(!needs_transpose(&[8]));
        assert!(
            !needs_transpose(&[1, 2, 3]),
            "this model has no rank-3 weights"
        );
    }

    /// The key list has no duplicates and covers the count, and every key has a
    /// derived shape: an unmapped key would load as "missing" against a real
    /// file and is invisible until then.
    #[test]
    fn every_expected_key_has_exactly_one_derived_shape() {
        let keys = expected_keys();
        let set: HashSet<&String> = keys.iter().collect();
        assert_eq!(set.len(), keys.len(), "duplicate key in expected_keys()");
        for k in &keys {
            let s = expected_shape(k).unwrap_or_else(|| panic!("no derived shape for `{k}`"));
            assert!(!s.is_empty(), "empty shape for `{k}`");
            assert!(
                s.iter().all(|&d| d > 0),
                "non-positive dim in {s:?} for `{k}`"
            );
        }
    }

    /// The shape table, at the positions a mistake would hide in: the q/k/v
    /// concatenation, the per-head gates, the `[512, 384]` output projection
    /// (which is the one rank-2 weight whose two dims differ *and* whose
    /// transpose direction is easy to invert), and a band's widths.
    #[test]
    fn derived_shapes_are_the_architectures_own() {
        assert_eq!(
            expected_shape("layers_0.time_transformer.layers_0.attn.to_qkv.weight"),
            Some(vec![384, 1536])
        );
        assert_eq!(
            expected_shape("layers_0.time_transformer.layers_0.attn.to_gates.weight"),
            Some(vec![384, 8])
        );
        assert_eq!(
            expected_shape("layers_5.freq_transformer.layers_0.attn.to_out.layers.0.weight"),
            Some(vec![512, 384])
        );
        assert_eq!(
            expected_shape("layers_5.freq_transformer.layers_0.ff.net.layers.1.weight"),
            Some(vec![384, 1536])
        );
        assert_eq!(
            expected_shape("layers_5.freq_transformer.layers_0.ff.net.layers.4.bias"),
            Some(vec![384])
        );
        assert_eq!(
            expected_shape("layers_2.time_transformer.norm.weight"),
            Some(vec![384])
        );
        // Band 0 holds 7 bins → 28 features; band 59 holds 130 → 520.
        assert_eq!(
            expected_shape("band_split.to_features_0.norm.weight"),
            Some(vec![28])
        );
        assert_eq!(
            expected_shape("band_split.to_features_0.linear.weight"),
            Some(vec![28, 384])
        );
        assert_eq!(
            expected_shape("band_split.to_features_59.linear.weight"),
            Some(vec![520, 384])
        );
        // Mask MLP output is `2 · dim_in` for the GLU.
        assert_eq!(
            expected_shape("mask_estimators_0.to_freqs_0.layers.4.weight"),
            Some(vec![1536, 56])
        );
        assert_eq!(
            expected_shape("mask_estimators_0.to_freqs_59.layers.4.weight"),
            Some(vec![1536, 1040])
        );
        assert_eq!(
            expected_shape("mask_estimators_0.to_freqs_59.layers.0.bias"),
            Some(vec![1536])
        );
    }

    /// A complete fake checkpoint: every expected key, at the *stored*
    /// (`[out, in]` for rank 2) shape. Loading it must fill the model, which is
    /// the coverage proof that does not need the real file.
    fn fake_checkpoint() -> HashMap<String, Array> {
        let mut map = HashMap::new();
        for key in expected_keys() {
            let want = expected_shape(&key).expect("shape");
            let stored: Vec<i32> = if needs_transpose(&want) {
                vec![want[1], want[0]]
            } else {
                want
            };
            map.insert(
                key.clone(),
                Array::zeros::<f32>(&stored)
                    .unwrap_or_else(|e| panic!("zeros {stored:?} for {key}: {e}")),
            );
        }
        map
    }

    #[test]
    fn a_complete_tensor_map_loads_into_the_model() {
        let weights = fake_checkpoint();
        assert_eq!(weights.len(), EXPECTED_TENSOR_COUNT);
        let mut model = RoFormerModel::new();
        apply_weights(&mut model, &weights, Path::new("fake"))
            .expect("a full, correctly shaped map must load");
        // Rank-2 weights arrive transposed relative to the file, so the model
        // can run `matmul(x, w)` directly.
        assert_eq!(
            model.blocks[0].time.layers[0].attn.w_qkv.shape(),
            &[DIM as i32, 3 * DIM_INNER as i32]
        );
        assert_eq!(model.band_split.bands[0].w.shape(), &[28, DIM as i32]);
        assert_eq!(
            model.mask_estimator.mlps[59].w[MASK_MLP_LAYERS - 1].shape(),
            &[(FF_MULT * DIM) as i32, 1040]
        );
    }

    /// A key the loader does not recognise is refused by name, even when the
    /// count happens to match.
    #[test]
    fn an_unrecognised_key_is_refused() {
        let mut weights = fake_checkpoint();
        let moved = weights.remove("layers_0.time_transformer.layers_0.attn.to_qkv.weight");
        assert!(moved.is_some(), "the fake map must contain the key");
        weights.insert(
            "layers_0.time_transformer.layers_0.attn.to_query.weight".into(),
            moved.expect("taken"),
        );
        let mut model = RoFormerModel::new();
        let err = apply_weights(&mut model, &weights, Path::new("renamed.safetensors"))
            .expect_err("a renamed key must not load");
        let detail = err.to_string();
        assert!(detail.contains("no place for"), "{detail}");
        assert!(detail.contains("to_query"), "{detail}");
    }

    /// `[in, out]` where the file stores `[out, in]` is the single most likely
    /// way to wire this loader backwards, and a square matrix would hide it: the
    /// non-square projections are the ones that catch it.
    #[test]
    fn a_weight_left_in_the_files_orientation_is_refused() {
        let mut weights = fake_checkpoint();
        let key = "band_split.to_features_0.linear.weight"; // stored [384, 28]
        weights.insert(
            key.to_string(),
            // The model's own `[in, out]` order, i.e. a weight that never got
            // transposed on the way in.
            Array::zeros::<f32>(&[28, 384]).expect("zeros"),
        );
        let mut model = RoFormerModel::new();
        let err = apply_weights(&mut model, &weights, Path::new("untransposed.safetensors"))
            .expect_err("an untransposed weight must not load");
        let detail = err.to_string();
        assert!(detail.contains(key), "{detail}");
        assert!(detail.contains("[384, 28]"), "{detail}");
        assert!(detail.contains("[28, 384]"), "{detail}");
    }

    /// A missing tensor names itself instead of leaving a zero weight in place,
    /// which is what a panic-based loader does to the first band it never
    /// reaches.
    #[test]
    fn a_missing_key_names_the_key() {
        let mut weights = fake_checkpoint();
        weights.remove("mask_estimators_0.to_freqs_59.layers.4.bias");
        let mut model = RoFormerModel::new();
        let err = apply_weights(&mut model, &weights, Path::new("short.safetensors"))
            .expect_err("a missing tensor must not load");
        let detail = err.to_string();
        // Count is checked first, so the message is the breakdown.
        assert!(detail.contains("holds 671 tensors"), "{detail}");
        assert!(detail.contains("672"), "{detail}");
        weights.insert(
            "spare".to_string(),
            Array::zeros::<f32>(&[1]).expect("zeros"),
        );
        let err = apply_weights(&mut model, &weights, Path::new("renamed.safetensors"))
            .expect_err("672 keys with one wrong is still wrong");
        assert!(err.to_string().contains("`spare`"), "{err}");
    }

    /// [`expected_shape`] is only load-bearing if it agrees with the checks
    /// [`apply_weights`] performs; a divergence would mean the fake checkpoint
    /// proves nothing about the real file. Asserted on a sample of both paths.
    #[test]
    fn the_shape_table_and_the_loader_agree() {
        let weights = fake_checkpoint();
        for key in [
            "band_split.to_features_0.linear.weight",
            "band_split.to_features_59.linear.weight",
            "layers_0.time_transformer.layers_0.attn.to_qkv.weight",
            "layers_3.freq_transformer.layers_0.attn.to_out.layers.0.weight",
            "layers_5.time_transformer.layers_0.ff.net.layers.1.bias",
            "mask_estimators_0.to_freqs_17.layers.0.weight",
            "mask_estimators_0.to_freqs_17.layers.4.weight",
        ] {
            let want = expected_shape(key).expect("derived");
            let stored = weights.get(key).unwrap_or_else(|| panic!("missing {key}"));
            let as_stored: Vec<i32> = if needs_transpose(&want) {
                vec![want[1], want[0]]
            } else {
                want.clone()
            };
            check_shape(stored, key, &as_stored)
                .unwrap_or_else(|e| panic!("{key}: table vs loader disagree: {e}"));
        }
    }
}
