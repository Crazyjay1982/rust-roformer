#!/usr/bin/env python3
"""Turn the stock fp32 ONNX export into the safetensors file the MLX engine loads.

WHAT THIS DOES
    ``rust-roformer``'s MLX engine (``src/engine/mlx/``) implements the Mel-Band
    RoFormer architecture itself and runs it on weights read from one
    ``.safetensors`` file of 672 tensors. This crate ships no weights: you point
    this script at an fp32 ONNX export of the vocal-separation checkpoint that you
    downloaded yourself, and it re-keys that file's initializers into the layout
    the Rust loader expects. The output is a derivative of a file you already
    have, not a redistribution of anything -- the terms that cover the checkpoint
    (see ``docs/LICENSES.md``) cover what this writes, and it stays out of the
    repository.

    ``torch.onnx`` legacy tracing kept the module path inside each node name
    (e.g. ``/model/layers.0.0/layers.0.0/to_qkv/MatMul``) while renaming weight
    initializers to ``onnx::MatMul_XXXX``. So the MatMul weights are mapped by
    their node name and the norm gammas and biases by their initializer name; both
    land in the flat key scheme documented at the top of
    ``src/engine/mlx/weights.rs``.

    Linear weights are written ``[out, in]`` -- the ``nn.Linear`` layout a torch
    state dict carries. The Rust loader transposes them to ``[in, out]`` once, at
    load time, so the graph runs plain ``matmul(x, w) + b``. Keep that
    convention: the loader checks each shape in *this* file's orientation and
    refuses a file that has already been turned.

    The export's embedded STFT buffers (``stft.*`` / ``istft.*`` initializers) are
    skipped deliberately. This engine computes its own STFT with an FFT, which is
    the arithmetic the checkpoint was trained against; the export evaluates the
    same transform as a convolution and drifts from the model in quiet
    high-frequency bands, so copying those tables over would import the drift.

USAGE
    python3 tools/extract_onnx_weights.py IN.onnx OUT.safetensors

    Needs ``onnx``, ``numpy`` and ``mlx`` (for ``save_safetensors``) already
    installed. No network access, and no torch: the export is the input.

EXIT CODES
    0 wrote the file; 1 the export is not the graph this mapping describes;
    2 wrong command line.
"""

import os
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper

# What the architecture fixes, restated here only so the file written out can be
# checked against it before it is ever loaded.
DIM = 384
DIM_INNER = 512
HEADS = 8
EXPECTED_TENSOR_COUNT = 672


def mlx_module_key(path: str) -> str:
    """traced node module path -> MLX key prefix.

    Node names double the nested prefix (``band_split.to_features.0.to_features.0.1``
    is the Linear inside ``band_split.to_features.0``), because the tracer appends
    the child's own path to an already-qualified parent.
    """
    # band_split.to_features.{b}.to_features.{b}.1
    mt = re.fullmatch(r"band_split\.to_features\.(\d+)\.to_features\.\1\.1", path)
    if mt:
        return f"band_split.to_features_{mt.group(1)}.linear"
    # layers.{i}.{t}.layers.0.0.{to_qkv|to_gates|to_out.to_out.0}
    mt = re.fullmatch(
        r"layers\.(\d+)\.([01])\.layers\.0\.0\.(to_qkv|to_gates|to_out\.to_out\.0)", path
    )
    if mt:
        i, t, sub = mt.groups()
        tt = "time_transformer" if t == "0" else "freq_transformer"
        return f"layers_{i}.{tt}.layers_0.attn.{sub}"
    # layers.{i}.{t}.layers.0.1.net.net.{j}
    mt = re.fullmatch(r"layers\.(\d+)\.([01])\.layers\.0\.1\.net\.net\.(\d+)", path)
    if mt:
        i, t, j = mt.groups()
        tt = "time_transformer" if t == "0" else "freq_transformer"
        return f"layers_{i}.{tt}.layers_0.ff.net.layers.{j}"
    # mask_estimators.{s}.to_freqs.{b}.to_freqs.{b}.0.to_freqs.{b}.0.{j}
    mt = re.fullmatch(
        r"mask_estimators\.(\d+)\.to_freqs\.(\d+)\.to_freqs\.\2\.0\.to_freqs\.\2\.0\.(\d+)", path
    )
    if mt:
        s, b, j = mt.groups()
        return f"mask_estimators_{s}.to_freqs_{b}.layers.{j}"
    raise ValueError(f"unhandled module path: {path}")


def mlx_named_key(name: str) -> str:
    """torch initializer name -> MLX weight key."""
    if not name.startswith("model."):
        raise ValueError(f"initializer outside the model: {name}")
    k = name[len("model."):]
    # The export's norm parameter is `gamma`; this graph calls it `weight`.
    k = k.replace(".gamma", ".weight")
    mt = re.fullmatch(r"band_split\.to_features\.(\d+)\.([01])\.(.+)", k)
    if mt:
        b, sub, rest = mt.groups()
        return f"band_split.to_features_{b}.{'norm' if sub == '0' else 'linear'}.{rest}"
    mt = re.fullmatch(r"layers\.(\d+)\.([01])\.layers\.(\d+)\.([01])\.(.+)", k)
    if mt:
        i, t, kk, sub, rest = mt.groups()
        tt = "time_transformer" if t == "0" else "freq_transformer"
        s = "attn" if sub == "0" else "ff"
        # Re-number a `Sequential`'s members into `layers.{n}` so the key stays
        # readable once flattened: `to_out.0` -> `to_out.layers.0`.
        rest = re.sub(r"^net\.(\d+)\.", r"net.layers.\1.", rest)
        rest = re.sub(r"^to_out\.(\d+)\.", r"to_out.layers.\1.", rest)
        rest = re.sub(r"\.net\.(\d+)\.", r".net.layers.\1.", rest)
        rest = re.sub(r"\.to_out\.(\d+)\.", r".to_out.layers.\1.", rest)
        return f"layers_{i}.{tt}.layers_{kk}.{s}.{rest}"
    mt = re.fullmatch(r"layers\.(\d+)\.([01])\.norm\.(.+)", k)
    if mt:
        i, t, rest = mt.groups()
        tt = "time_transformer" if t == "0" else "freq_transformer"
        return f"layers_{i}.{tt}.norm.{rest}"
    mt = re.fullmatch(r"mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.(\d+)\.(.+)", k)
    if mt:
        s, b, j, rest = mt.groups()
        return f"mask_estimators_{s}.to_freqs_{b}.layers.{j}.{rest}"
    k = re.sub(r"\.net\.(\d+)\.", r".net.layers.\1.", k)
    k = re.sub(r"\.to_out\.(\d+)\.", r".to_out.layers.\1.", k)
    return k


def main(onnx_path: str, out_path: str) -> int:
    if not os.path.isfile(onnx_path):
        print(f"FAIL: no such ONNX file: {onnx_path}", file=sys.stderr)
        return 1
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    inits = {it.name: numpy_helper.to_array(it) for it in graph.initializer}

    out = {}
    # Named initializers: the norms and biases. The embedded STFT/FFT buffers are
    # skipped -- see the docstring.
    for name, arr in inits.items():
        if name.startswith(("onnx::", "stft.", "istft.")):
            continue
        key = mlx_named_key(name)
        if key in out:
            print(f"FAIL: duplicate key {key} from initializer {name}", file=sys.stderr)
            return 1
        out[key] = arr.astype(np.float32)

    # Unnamed MatMul weights, mapped through their node names.
    n_mm = 0
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        weight_input = node.input[1]
        if not weight_input.startswith("onnx::"):
            continue
        # node name: /model/<module.path>/MatMul
        parts = node.name.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "model" or parts[-1] != "MatMul":
            print(f"FAIL: unexpected MatMul node name {node.name}", file=sys.stderr)
            return 1
        key = mlx_module_key(".".join(parts[1:-1]))
        # The attention sub-modules carry their own suffix; a plain Linear gets
        # `.weight`.
        if key.endswith((".attn.to_qkv", ".attn.to_gates")):
            key += ".weight"
        elif key.endswith(".attn.to_out.to_out.0"):
            key = key[: -len(".to_out.to_out.0")] + ".to_out.layers.0.weight"
        else:
            key += ".weight"
        if key in out:
            print(f"FAIL: duplicate key {key} from node {node.name}", file=sys.stderr)
            return 1
        arr = inits[weight_input].astype(np.float32)
        # The export folds `nn.Linear` into a MatMul with the weight stored
        # `[in, out]` (applied as `x @ W`); torch's state dict -- and this file's
        # convention -- is `[out, in]`.
        out[key] = arr.T.copy()
        n_mm += 1

    print(f"matmul weights mapped: {n_mm}; total tensors: {len(out)}")
    if len(out) != EXPECTED_TENSOR_COUNT:
        print(
            f"FAIL: mapped {len(out)} tensors, the MLX engine expects "
            f"{EXPECTED_TENSOR_COUNT} (180 band split + 60 attention + 60 "
            f"feed-forward + 12 block norms + 360 mask estimator)",
            file=sys.stderr,
        )
        return 1

    # Shape sanity on the three projections whose orientation a wrong transpose
    # would otherwise hide, because the loader can only tell `[out, in]` from
    # `[in, out]` where the two differ.
    checks = {
        "attn.to_qkv.weight": (3 * DIM_INNER, DIM),
        "attn.to_out.layers.0.weight": (DIM, DIM_INNER),
        "attn.to_gates.weight": (HEADS, DIM),
    }
    for key, arr in out.items():
        for suffix, want in checks.items():
            if key.endswith(suffix) and tuple(arr.shape) != want:
                print(f"FAIL: {key} is {arr.shape}, expected {want}", file=sys.stderr)
                return 1

    import mlx.core as mx

    mx.save_safetensors(out_path, {k: mx.array(v) for k, v in out.items()})
    print(f"saved {len(out)} tensors -> {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: extract_onnx_weights.py IN.onnx OUT.safetensors", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
