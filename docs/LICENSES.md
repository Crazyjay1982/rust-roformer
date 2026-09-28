# Licences and provenance

Read this before you redistribute anything. It is short, and the last section is
the part people usually skip and later wish they had not.

## 1. The code in this repository

Apache License 2.0 — see [`LICENSE`](../LICENSE). This covers `src/`,
`tools/`, `examples/`, `docs/` and the tests. It does **not** cover model
weights, and this repository ships none: no `.onnx`, no `.safetensors`, no
audio. `.gitignore` keeps them out on purpose, and `tools/` exists so you can
produce what you need from a checkpoint you downloaded yourself.

## 2. The model this code runs

The architecture, the checkpoint, and the ONNX export are three different
distributions with three different rights-holders. What their authors declare,
verified against the public APIs on 2026-09-27 rather than copied from a README:

| What | Where | Declared licence | Checked |
| --- | --- | --- | --- |
| Architecture + training code | `ZFTurbo/Music-Source-Separation-Training` (GitHub), arXiv:2310.01809 | MIT (repository licence field) | 2026-09-27, 1,562 stars, last push 2026-09-26 |
| Vocals checkpoint | `KimberleyJSN/melbandroformer` (Hugging Face) | `license:mit` card tag | 2026-09-27, card last modified 2026-04-22 |
| ONNX export used by the `onnx` engine | `smank/mel-band-roformer-vocals-onnx` (Hugging Face) | `license:mit` card tag | 2026-09-27, card last modified 2026-07-02 |

The export is a single file, which makes integrity checking unambiguous:
`melband_roformer_vocals.onnx`, **953,292,899 bytes**, git-lfs oid
`64a4f3bee48fbe7d971b23875adc924ed004c3533f49672592641dddc0f6f561`. If your copy
is not that size and hash, you are running something else, and every number in
`docs/benchmarks.md` stops applying to it. The copy this code was developed and
measured against hashes to exactly that value — so "the reference model is the
stock upstream export, unmodified" is a checkable statement here, not a
convenience claim.

`tools/reduce_window.py` edits that file's shape constants so a forward pass runs
on a shorter window; `tools/extract_onnx_weights.py` reads it and writes the
`.safetensors` the `mlx` engine loads. Both are offline transformations of a file
you already hold. Neither is a licence to redistribute the result: a derivative of
a weight file carries the weight file's terms, not this repository's Apache-2.0.

## 3. Runtime libraries

The `onnx` engine links ONNX Runtime through the `ort` crate; the `mlx` engine
links Apple's MLX through `mlx-rs`. Each has its own licence terms, and the set of
transitive crates changes over time, so do not trust a list written in a markdown
file — regenerate it against the lockfile you actually built with:

```sh
cargo install cargo-license   # once
cargo license | tee THIRD_PARTY_CRATES.txt
```

## 4. The part that is not solved by a licence tag

Every card above says MIT, and MIT is about the artefacts those authors published.
None of them states what the vocals checkpoint was trained on. A model whose
outputs are useful for separating recorded music is, by construction, a model that
saw recorded music, and that is a question about the *weights* which no licence tag
in the table above answers.

So, stated plainly:

* This repository makes no claim that training-data rights are cleared.
* If you ship a product that runs these weights, the question is yours to answer
  with your own counsel — not because anyone told you it is fine, and not because
  a Hugging Face card field says `mit`.
* What the code here does give you is reproducibility of the parts that are ours:
  the streaming I/O, the memory gate, and the arithmetic of the model
  implementation, which you can check against PyTorch without ever publishing a
  weight file.
