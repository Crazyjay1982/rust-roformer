#!/usr/bin/env python3
"""Window-reduction graph surgery for the Mel-Band RoFormer vocal-separation ONNX.

WHAT THIS DOES
    The shipped model bakes its input window length (in samples) into a handful
    of int64 graph constants plus the I/O tensor dims. This script rewrites
    those to a new window WITHOUT touching a single weight byte.

    Four int64 ``Constant`` nodes carry the window, and are discovered here by
    *value* rather than by hard-coded name:

        /Constant          int64[3] = [2, 1, <win>]  -> /Reshape      (input fold)
        /istft/Constant_23 int64[]   = <win>          -> /istft/Sub_1  (pad amount)
        /istft/Constant_29 int64[1]  = <win>          -> /istft/Slice_4 end (trim)
        /Constant_35       int64[1]  = <win>          -> /Concat_8     (output shape)

    plus ``graph.input`` / ``graph.output`` last dims.

THE ONE REAL CONSTRAINT -- ``istft.window_sum_inv``
    iSTFT here is a ConvTranspose (``kernel_shape=[2048]``, ``strides=[441]``)
    whose output is normalised element-wise by the initializer
    ``istft.window_sum_inv``, a reciprocal overlap-add window-power table. Its
    length is ``<baked_max_window> + n_fft`` -- 354848 = 352800 + 2048 in the
    stock 8 s model -- and it is sliced at RUNTIME to the ConvTranspose output
    length, which is why shrinking the window needs no change to it.

    * SHRINKING is exact. Reusing a prefix of the longer table is provably
      correct: frame ``fr`` covers samples ``[fr*hop, fr*hop + n_fft)``, so the
      first sample position where a discarded frame could contribute is
      ``n_frames_target * hop``, which is ``hop`` samples PAST the point where
      ``/istft/Slice_4`` trims. Every retained sample therefore reads the same
      value a native export at that window would produce. Verified on the
      factory pair: the 8 s and 4 s files have byte-identical initializers.

    * GROWING beyond the baked capacity is NOT possible by re-pointing. The
      table is physically too short, and would have to be regenerated. This
      script refuses unless ``--force`` is given.

USAGE
    reduce_window.py IN.onnx OUT.onnx 176400
    reduce_window.py IN.onnx OUT.onnx 176400 --expect shipped_4s.onnx
    reduce_window.py IN.onnx OUT.onnx 352800 --verify     # round-trip proof
    reduce_window.py IN.onnx --report

    Needs only ``onnx`` (+ numpy). No torch, no network. Loads the whole
    protobuf in memory (~3x file size RSS), so an 8 s fp32 graph wants ~3 GB.

SERIALIZATION
    Rewrites via ``onnx.save``. Re-serialising this model family is
    byte-faithful: load+save of the 8 s int8 graph reproduces its SHA-256
    exactly, so the output differs from the input ONLY at the fields listed
    above. That is what makes ``--expect`` / ``--verify`` meaningful.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import sys
import tempfile

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # pragma: no cover
    sys.exit("error: the `onnx` package is required (pip install onnx)")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _log(msg=""):
    print(msg, flush=True)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dims_of(vi):
    """Concrete dims of a ValueInfoProto; symbolic dims come back as str."""
    out = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append(None)
    return out


def set_dim(vi, idx, value):
    vi.type.tensor_type.shape.dim[idx].dim_value = int(value)


def tensor_int64_values(tp):
    """Return (list_of_ints, is_mutable) for an INT64 TensorProto."""
    if tp.data_type != TensorProto.INT64:
        return None
    arr = numpy_helper.to_array(tp)
    return arr


def find_conv_geometry(graph):
    """Recover (n_fft, hop) from the conv-based STFT/iSTFT.

    The STFT is two Conv nodes with ``kernel_shape=[n_fft]`` and
    ``strides=[hop]``; iSTFT is one ConvTranspose with the same geometry.
    Returns (n_fft, hop) or (None, None) if not found.
    """
    geom = []
    for n in graph.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            ks, st = None, None
            for a in n.attribute:
                if a.name == "kernel_shape" and list(a.ints):
                    ks = list(a.ints)[0]
                if a.name == "strides" and list(a.ints):
                    st = list(a.ints)[0]
            # only the 1-D time-domain convs look like this
            if ks and st and len(n.input) >= 2 and ks > 1:
                geom.append((ks, st, n.op_type, n.name))
    if not geom:
        return None, None
    n_fft = geom[0][0]
    hop = geom[0][1]
    return n_fft, hop


# --------------------------------------------------------------------------- #
# core surgery
# --------------------------------------------------------------------------- #

class Surgery:
    """Everything we find / change, for reporting and verification."""

    def __init__(self):
        self.constants = []      # (node_name, consumer_op, elem_indices, old, new)
        self.io = []             # ("input"/"output", name, axis, old, new)
        self.value_info = []     # (name, axis, old, new)
        self.stripped_value_info = 0

    def summary(self):
        lines = []
        for name, cons, idxs, old, new in self.constants:
            lines.append("  Constant %-22s -> %-6s elems %s : %s -> %s"
                         % (name, cons, idxs, old, new))
        for kind, name, ax, old, new in self.io:
            lines.append("  %-8s %-14s axis %d : %s -> %s" % (kind, name, ax, old, new))
        if self.value_info:
            lines.append("  value_info dims patched: %d" % len(self.value_info))
        if self.stripped_value_info:
            lines.append("  value_info records stripped: %d" % self.stripped_value_info)
        return "\n".join(lines)


def detect_window(model):
    """The window = the concrete last dim of the graph input."""
    g = model.graph
    if not g.input:
        raise SystemExit("error: graph has no inputs")
    d = dims_of(g.input[0])
    if not d or not isinstance(d[-1], int):
        raise SystemExit("error: graph input last dim is not concrete; this model "
                         "has dynamic length and needs no surgery")
    return d[-1]


def run_surgery(model, cur_win, target, shape_meta="strip", strict=True):
    """Patch ``model`` in place from cur_win -> target. Returns a Surgery."""
    s = Surgery()
    g = model.graph

    # ---- 0. geometry + capability -------------------------------------------
    n_fft, hop = find_conv_geometry(g)
    wsi = next((i for i in g.initializer if "window_sum_inv" in i.name), None)
    if wsi is not None:
        baked = list(wsi.dims)[0] - (n_fft or 0)
        s.baked_max_window = baked          # type: ignore[attr-defined]
        s.window_sum_inv_name = wsi.name    # type: ignore[attr-defined]
        if target > baked and strict:
            raise SystemExit(
                "error: cannot grow the window past %d samples.\n"
                "  %s holds %d values = %d (max window) + %d (n_fft); it is sliced\n"
                "  at runtime, so shrinking reuses a prefix but growing needs the\n"
                "  table RECOMPUTED, which this tool does not do.\n"
                "  Use --force to patch anyway (output will be numerically wrong)."
                % (baked, wsi.name, list(wsi.dims)[0], baked, n_fft or 0))
    if hop and strict and target % hop:
        raise SystemExit(
            "error: target window %d is not a multiple of the STFT hop %d;\n"
            "  frame count would be fractional and the iSTFT trim would desync.\n"
            "  (valid windows: %d, %d, %d ...)"
            % (target, hop, (target // hop) * hop, (target // hop + 1) * hop,
               (target // hop + 2) * hop))

    # ---- 1. the four int64 window constants, discovered by value -----------
    producers = {o: n for n in g.node for o in n.output}
    n_patched = 0
    for n in g.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.name != "value" or a.t.data_type != TensorProto.INT64:
                continue
            arr = numpy_helper.to_array(a.t)
            if arr.size == 0:
                continue
            orig_shape = tuple(a.t.dims)
            flat = arr.reshape(-1).copy()          # explicit copy: never rely on
            idxs = [i for i, v in enumerate(flat) if int(v) == cur_win]
            if not idxs:
                continue
            for i in idxs:
                flat[i] = target
            # Preserve the tensor's OWN name exactly. These Constant tensors carry
            # an empty name in the factory files; filling it in from the node
            # name would add 65 bytes to this model and break byte-exactness.
            a.t.CopyFrom(numpy_helper.from_array(flat.reshape(orig_shape),
                                                 a.t.name or None))
            cons = []
            for out in n.output:
                for m in g.node:
                    if out in m.input:
                        cons.append(m.op_type)
            s.constants.append((n.name or "<unnamed>", ",".join(sorted(set(cons))) or "-",
                                idxs, cur_win, target))
            n_patched += 1

    # ---- 2. graph input / output dims --------------------------------------
    for kind, seq in (("input", g.input), ("output", g.output)):
        for vi in seq:
            d = dims_of(vi)
            for ax, v in enumerate(d):
                if isinstance(v, int) and v == cur_win:
                    set_dim(vi, ax, target)
                    s.io.append((kind, vi.name, ax, cur_win, target))

    # ---- 3. cached shape metadata (value_info) ------------------------------
    if shape_meta == "strip":
        s.stripped_value_info = len(g.value_info)
        del g.value_info[:]
    elif shape_meta == "patch":
        for vi in g.value_info:
            d = dims_of(vi)
            for ax, v in enumerate(d):
                if isinstance(v, int) and v == cur_win:
                    set_dim(vi, ax, target)
                    s.value_info.append((vi.name, ax, cur_win, target))
    # shape_meta == "keep": leave stale, only useful for experiments

    s.n_constants = n_patched
    return s


def byte_diff(path_a, path_b, max_runs=64):
    """Byte-level diff of two files: differing count + first runs."""
    sa, sb = os.path.getsize(path_a), os.path.getsize(path_b)
    if sa != sb:
        return {"same_length": False, "size_a": sa, "size_b": sb, "size_delta": sa - sb,
                "differing_bytes": None, "runs": []}
    runs = []
    n = 0
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        CH = 1 << 22
        off = 0
        cur = None
        while True:
            ba, bb = fa.read(CH), fb.read(CH)
            if not ba:
                break
            for i in range(len(ba)):
                if ba[i] != bb[i]:
                    n += 1
                    if cur and cur[0] + cur[1] == off + i:
                        cur = (cur[0], cur[1] + 1)
                    else:
                        if cur:
                            runs.append(cur)
                        cur = (off + i, 1)
            off += len(ba)
    if cur:
        runs.append(cur)
    runs = [tuple(r) for r in runs]
    return {"same_length": True, "size_a": sa, "size_b": sb, "size_delta": 0,
            "differing_bytes": n, "runs": runs, "runs_shown": runs[:max_runs]}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def do_report(path):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    win = detect_window(m)
    n_fft, hop = find_conv_geometry(g)
    _log("file              %s" % path)
    _log("size              %d bytes" % os.path.getsize(path))
    _log("nodes             %d" % len(g.node))
    _log("initializers      %d" % len(g.initializer))
    _log("value_info recs   %d" % len(g.value_info))
    _log("graph input       %s %s" % (g.input[0].name, dims_of(g.input[0])))
    _log("graph output      %s %s" % (g.output[0].name, dims_of(g.output[0])))
    _log("detected window   %d samples  (%.3f s @ 44.1 kHz)" % (win, win / 44100))
    _log("stft geometry     n_fft=%s hop=%s" % (n_fft, hop))
    _log("window/hop        %s" % (win / hop if hop else "?"))
    for i in g.initializer:
        if "window_sum_inv" in i.name:
            baked = list(i.dims)[0] - (n_fft or 0)
            _log("%-16s %s len=%d -> baked max window %d (%.3f s)"
                 % ("window table", i.name, list(i.dims)[0], baked, baked / 44100))
    hits = []
    for n in g.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.name == "value" and a.t.data_type == TensorProto.INT64:
                arr = numpy_helper.to_array(a.t).ravel()
                if arr.size and (arr == win).any():
                    cons = sorted({m2.op_type for o in n.output
                                   for m2 in g.node if o in m2.input})
                    hits.append((n.name, cons, list(a.t.dims)))
    _log("window constants  %d found" % len(hits))
    for h in hits:
        _log("   %-22s -> %-8s dims=%s" % h)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rewrite the baked input window of a Mel-Band RoFormer ONNX "
                    "model without touching any weight.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="source .onnx")
    ap.add_argument("output", nargs="?", help="destination .onnx")
    ap.add_argument("window", nargs="?", type=int,
                    help="target window in samples (e.g. 176400 = 4 s @ 44.1 kHz)")
    ap.add_argument("--report", action="store_true",
                    help="describe the model's window baking and exit")
    ap.add_argument("--shape-metadata", choices=("strip", "patch", "keep"),
                    default="strip",
                    help="what to do with cached value_info shapes. 'strip' matches "
                         "the factory 4 s artifact byte-for-byte; 'patch' keeps "
                         "metadata consistent with the new window (default: strip)")
    ap.add_argument("--force", action="store_true",
                    help="allow growing past the baked window_sum_inv capacity "
                         "(the normalisation table would have to be recomputed)")
    ap.add_argument("--expect", metavar="PATH",
                    help="after writing, compare the result to PATH byte-for-byte")
    ap.add_argument("--verify", action="store_true",
                    help="round-trip: patch to TARGET, then back to the original "
                         "window, and compare with INPUT byte-for-byte")
    ap.add_argument("--check", action="store_true",
                    help="run onnx.checker on the result (slower on big graphs)")
    ap.add_argument("--keep-temp", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.report:
        do_report(a.input)
        return 0
    if a.output is None or a.window is None:
        ap.error("need OUTPUT and WINDOW (or --report)")

    strict = not a.force
    _log("loading %s ..." % a.input)
    model = onnx.load(a.input, load_external_data=False)
    cur = detect_window(model)
    _log("detected source window: %d samples (%.3f s @ 44.1 kHz)" % (cur, cur / 44100))
    if cur == a.window:
        _log("target equals source window; nothing to do")
        onnx.save(model, a.output, save_as_external_data=False)
        return 0

    s = run_surgery(model, cur, a.window, a.shape_metadata, strict)
    _log("surgery:\n%s" % s.summary())
    if s.n_constants == 0:
        raise SystemExit("error: no int64 constant equal to the window was found -- "
                         "this model does not bake its window, or the detection is wrong")
    if a.check:
        onnx.checker.check_model(model)
        _log("onnx.checker: OK")
    _log("writing %s ..." % a.output)
    onnx.save(model, a.output, save_as_external_data=False)
    _log("  %d bytes (source %d, delta %+d)"
         % (os.path.getsize(a.output), os.path.getsize(a.input),
            os.path.getsize(a.output) - os.path.getsize(a.input)))

    rc = 0

    if a.expect:
        d = byte_diff(a.output, a.expect)
        _log("\n--expect comparison vs %s" % a.expect)
        if not d["same_length"]:
            _log("  DIFFERENT LENGTH: %d vs %d (delta %+d)"
                 % (d["size_a"], d["size_b"], d["size_delta"]))
        elif d["differing_bytes"] == 0:
            _log("  BYTE-IDENTICAL")
        else:
            _log("  %d differing bytes in %d run(s): %s"
                 % (d["differing_bytes"], len(d["runs"]), d["runs_shown"]))
        _log("  sha256 out %s" % sha256_file(a.output))
        _log("  sha256 exp %s" % sha256_file(a.expect))
        rc = 0 if d["same_length"] and d["differing_bytes"] == 0 else 1

    if a.verify:
        _log("\n--verify round trip: %d -> %d -> %d" % (cur, a.window, cur))
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False,
                                          dir=os.path.dirname(os.path.abspath(a.output))
                                          or ".")
        tmp.close()
        try:
            back = onnx.load(a.output, load_external_data=False)
            cur2 = detect_window(back)
            if cur2 != a.window:
                raise SystemExit("error: forward patch did not take (window reads %d)" % cur2)
            sm = a.shape_metadata if a.shape_metadata != "strip" else "strip"
            run_surgery(back, cur2, cur, sm, strict)
            onnx.save(back, tmp.name, save_as_external_data=False)
            d = byte_diff(a.input, tmp.name)
            if not d["same_length"]:
                _log("  NOT bit-for-bit: %d vs %d bytes (delta %+d)"
                     % (d["size_a"], d["size_b"], d["size_delta"]))
                rc = 1
            elif d["differing_bytes"] == 0:
                _log("  ROUND-TRIP BYTE-IDENTICAL to %s" % a.input)
            else:
                _log("  %d differing byte(s) in %d run(s)"
                     % (d["differing_bytes"], len(d["runs"])))
                _log("  runs: %s" % d["runs_shown"])
                rc = 1
            _log("  sha256 source      %s" % sha256_file(a.input))
            _log("  sha256 round-trip  %s" % sha256_file(tmp.name))
        finally:
            if not a.keep_temp:
                os.unlink(tmp.name)
    return rc


if __name__ == "__main__":
    sys.exit(main())
