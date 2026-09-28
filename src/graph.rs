//! Find and rewrite the input window baked into an exported graph, in memory.
//!
//! # Why this exists
//!
//! A Mel-Band RoFormer export fixes its window length in *shape*, not in a
//! setting: the value appears as a handful of `int64` `Constant` nodes and as the
//! last dimension of the graph's input and output. ONNX Runtime has no knob for
//! it. That left two options: ask every user to run an offline script over their
//! downloaded model, or apply the same edit to a buffer at load time and hand the
//! bytes to `ort`'s `Session::builder().commit_from_memory(..)`. This module is the
//! second one, so the stock export stays the file on disk and the window becomes
//! a load-time parameter.
//!
//! # What is safe, and what is not
//!
//! * **Weights are never touched.** The edit rewrites `Constant` payloads and
//!   tensor dimensions only. If a window value is found inside an *initializer* (a
//!   weight), this module refuses outright, because at that point we would be
//!   editing the model's parameters and not its shape.
//! * **Shrinking is exact.** `istft.window_sum_inv` is an overlap-add
//!   normalisation table of length `capacity + n_fft`, and the graph slices it at
//!   runtime to whatever the ConvTranspose produced. Reusing a prefix of that
//!   table is provably identical to a native export at the shorter window: the
//!   first sample position where a dropped frame could contribute is exactly one
//!   hop past the point where the graph trims.
//! * **Growing is not possible here.** Beyond the baked capacity the table is
//!   physically shorter than the graph needs, and `Slice` clamps silently rather
//!   than erroring. [`patch_window`] refuses past `capacity`, and says so.
//! * **Length matters.** A constant's payload is fixed-width little-endian, so
//!   replacing its value cannot change the file's length. A `dim_value` is a
//!   protobuf *varint*, and varints grow at 2^7, 2^14, 2^21 …, so a window
//!   beyond 127 … 16,383 would need more bytes. Rewriting a varint in place with
//!   a different byte length would corrupt everything after it, so a
//!   non-length-neutral dimension is a refusal, not a re-serialisation.
//!
//! # What we do not claim
//!
//! The exported graph also carries cached shape-inference results
//! (`value_info`). They describe the *old* window, and we leave them alone:
//! deleting them changes the message length and would require re-serialising the
//! whole model. Whether the runtime accepts it was measured here: see
//! `engine::onnx`'s module note, which records the observed behaviour of building
//! a session from patched bytes.

use crate::error::{Error, Result};

/// `n_fft` of the STFT/iSTFT inside these graphs. The normalisation table is
/// `capacity + N_FFT` long, so this is what converts a table length into the
/// largest window that graph can serve.
pub const N_FFT: usize = 2048;
/// iSTFT hop, in samples. A window that is not a multiple of this cannot be
/// tiled by the overlap-add grid, and the graph would not fail loudly; it would
/// produce a short output.
pub const HOP: usize = 441;

const FIELD_NODE: u64 = 1;
const FIELD_INITIALIZER: u64 = 5;
const FIELD_INPUT: u64 = 11;
const FIELD_OUTPUT: u64 = 12;

// NodeProto
const NODE_NAME: u64 = 3;
const NODE_OP_TYPE: u64 = 4;
const NODE_ATTRIBUTE: u64 = 5;
// AttributeProto
const ATTR_NAME: u64 = 1;
const ATTR_INTS: u64 = 8;
/// `AttributeProto.t`: the tensor payload of a `Constant`'s `value` attribute.
///
/// Field **5**. Field 14 is `tp` (a `TypeProto`, deprecated since opset 16), which
/// was the number here before, and reading `tp` as a tensor is how every real
/// export ended up looking like a graph that does not bake its window: a decoded
/// `Constant` never matched, so `inspect` found no site to rewrite and refused.
/// Verified against the model's own bytes and against `onnx 1.18`'s descriptor:
/// `name = 1, f = 2, i = 3, s = 4, t = 5, g = 6, floats = 7, ints = 8,`.
const ATTR_TENSOR: u64 = 5;
// TensorProto
const TENSOR_DIMS: u64 = 1;
const TENSOR_DATA_TYPE: u64 = 2;
const TENSOR_INT64_DATA: u64 = 7;
const TENSOR_NAME: u64 = 8;
const TENSOR_RAW_DATA: u64 = 9;
const INT64: i64 = 7;
// ValueInfoProto / TypeProto / TensorShapeProto
const VI_TYPE: u64 = 2;
const TYPE_TENSOR: u64 = 1;
const TT_SHAPE: u64 = 2;
const SHAPE_DIM: u64 = 1;
const DIM_VALUE: u64 = 1;
// ModelProto / GraphProto
const MODEL_GRAPH: u64 = 7;
const GRAPH_NAME: u64 = 2;

/// One byte range we would rewrite, with the value found there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    /// Byte offset into the model buffer.
    pub offset: usize,
    /// Length of the encoded value at `offset`.
    pub len: usize,
    /// What it is, for the log and for tests.
    pub kind: SiteKind,
    /// Node or tensor name, when the enclosing message had one.
    pub path: String,
    /// The value currently encoded there.
    pub value: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteKind {
    /// A slot inside a `Constant` tensor's `raw_data` / `int64_data`.
    Constant,
    /// A `dim_value` varint in the graph's input or output shape.
    Dimension,
}

/// Everything [`inspect`] learned about a graph, before any decision.
#[derive(Debug, Clone)]
pub struct GraphReport {
    /// Window the graph declares on its input's last dimension.
    pub declared_window: i64,
    /// Sites carrying that value.
    pub sites: Vec<Site>,
    /// Largest window this graph's normalisation table can serve.
    pub capacity: Option<usize>,
    /// Cached shape-inference records that will disagree after a patch.
    pub value_info_records: usize,
    /// Number of nodes and initializers walked.
    pub nodes: usize,
    pub initializers: usize,
}

impl GraphReport {
    /// Sites we would actually rewrite.
    pub fn constants(&self) -> impl Iterator<Item = &Site> {
        self.sites.iter().filter(|s| s.kind == SiteKind::Constant)
    }

    pub fn dimensions(&self) -> impl Iterator<Item = &Site> {
        self.sites.iter().filter(|s| s.kind == SiteKind::Dimension)
    }
}

/// Refusals, all of them "we did not touch a byte".
fn refuse(why: &str) -> Error {
    Error::Model {
        detail: format!("{why} — refusing without modifying anything"),
    }
}

/// A `(start, end)` byte range inside the buffer being scanned.
type Span = (usize, usize);
/// One protobuf field: `(field number, wire type, span)`.
type Field = (u64, u8, Span);

/// Walk one message level, calling `f` for every field with its absolute byte
/// range. Unknown wire types are skipped; a truncated field is an error rather
/// than a silent stop.
fn fields_of(buf: &[u8], range: Span, out: &mut Vec<Field>) -> Result<()> {
    let (mut i, end) = range;
    while i < end {
        let (tag, next) = read_varint(buf, i, end)?;
        i = next;
        let field = tag >> 3;
        let wire = (tag & 0x7) as u8;
        let span = match wire {
            0 => {
                let (_, after) = read_varint(buf, i, end)?;
                (i, after)
            }
            1 => (i, i.saturating_add(8)),
            2 => {
                let (len, after) = read_varint(buf, i, end)?;
                let start = after;
                let stop = start
                    .checked_add(len as usize)
                    .ok_or_else(|| refuse("length prefix overflows"))?;
                if stop > end {
                    return Err(refuse("length-delimited field runs past its parent"));
                }
                (start, stop)
            }
            5 => (i, i.saturating_add(4)),
            _ => return Err(refuse(&format!("unsupported wire type {wire}"))),
        };
        out.push((field, wire, (span.0, span.1)));
        i = span.1;
    }
    Ok(())
}

fn read_varint(buf: &[u8], from: usize, end: usize) -> Result<(u64, usize)> {
    let mut shift = 0u32;
    let mut value = 0u64;
    let mut i = from;
    loop {
        if i >= end {
            return Err(refuse("truncated varint"));
        }
        let byte = buf[i];
        value |= ((byte & 0x7f) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return Ok((value, i));
        }
        shift += 7;
        if shift >= 64 {
            return Err(refuse("varint too long"));
        }
    }
}

/// `zigzag`-free protobuf int64: negative values are encoded as 10-byte unsigned
/// varints. Only non-negative window lengths matter here, so a value that does
/// not fit in `i64` is reported as out of range rather than wrapped.
fn varint_to_i64(buf: &[u8], range: (usize, usize)) -> Result<i64> {
    let (raw, _) = read_varint(buf, range.0, range.1)?;
    if raw > i64::MAX as u64 {
        return Err(refuse("a varint field exceeded i64 range"));
    }
    Ok(raw as i64)
}

fn string_of(buf: &[u8], range: (usize, usize)) -> String {
    String::from_utf8_lossy(&buf[range.0..range.1]).into_owned()
}

/// Collect every int64 in a TensorProto, with absolute byte ranges. Handles both
/// the packed (length-delimited) and unpacked (repeated varint) encodings, in
/// `int64_data` and in `raw_data`.
fn int64_slots(buf: &[u8], tensor: (usize, usize)) -> Result<Vec<(i64, usize, usize)>> {
    let mut fields = Vec::new();
    fields_of(buf, tensor, &mut fields)?;
    let mut slots = Vec::new();
    for (field, wire, span) in fields {
        if field != TENSOR_INT64_DATA && field != TENSOR_RAW_DATA {
            continue;
        }
        if field == TENSOR_RAW_DATA {
            // Fixed-width little-endian: 8 bytes per int64.
            let len = span.1 - span.0;
            if len % 8 != 0 {
                return Err(refuse("raw_data length is not a multiple of 8"));
            }
            for word in 0..(len / 8) {
                let at = span.0 + word * 8;
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&buf[at..at + 8]);
                slots.push((i64::from_le_bytes(bytes), at, 8));
            }
            continue;
        }
        match wire {
            2 => {
                // packed: a run of varints
                let mut i = span.0;
                while i < span.1 {
                    let (v, after) = read_varint(buf, i, span.1)?;
                    if v > i64::MAX as u64 {
                        return Err(refuse("packed int64 exceeded i64 range"));
                    }
                    slots.push((v as i64, i, after - i));
                    i = after;
                }
            }
            0 => {
                let v = varint_to_i64(buf, span)?;
                slots.push((v, span.0, span.1 - span.0));
            }
            _ => return Err(refuse("int64_data with an unexpected wire type")),
        }
    }
    Ok(slots)
}

/// True if a TensorProto declares itself INT64.
fn is_int64_tensor(buf: &[u8], tensor: (usize, usize)) -> Result<bool> {
    let mut fields = Vec::new();
    fields_of(buf, tensor, &mut fields)?;
    for (field, wire, span) in fields {
        if field == TENSOR_DATA_TYPE {
            if wire != 0 {
                return Err(refuse("TensorProto.data_type is not a varint"));
            }
            return Ok(varint_to_i64(buf, span)? == INT64);
        }
    }
    Ok(false)
}

fn tensor_name(buf: &[u8], tensor: (usize, usize)) -> Result<String> {
    let mut fields = Vec::new();
    fields_of(buf, tensor, &mut fields)?;
    for (field, _, span) in fields {
        if field == TENSOR_NAME {
            return Ok(string_of(buf, span));
        }
    }
    Ok(String::new())
}

/// Last `dim_value` of a ValueInfoProto, with its byte range.
fn last_dim(buf: &[u8], vi: (usize, usize)) -> Result<Option<(i64, usize, usize)>> {
    let mut fields = Vec::new();
    fields_of(buf, vi, &mut fields)?;
    for (field, _, span) in fields {
        if field != VI_TYPE {
            continue;
        }
        let mut tf = Vec::new();
        fields_of(buf, span, &mut tf)?;
        for (field, _, tspan) in tf {
            if field != TYPE_TENSOR {
                continue;
            }
            let mut sf = Vec::new();
            fields_of(buf, tspan, &mut sf)?;
            for (field, _, sspan) in sf {
                if field != TT_SHAPE {
                    continue;
                }
                let mut dims = Vec::new();
                fields_of(buf, sspan, &mut dims)?;
                let mut last = None;
                for (field, _, dspan) in dims {
                    if field != SHAPE_DIM {
                        continue;
                    }
                    let mut df = Vec::new();
                    fields_of(buf, dspan, &mut df)?;
                    for (field, wire, vspan) in df {
                        if field == DIM_VALUE && wire == 0 {
                            last = Some((varint_to_i64(buf, vspan)?, vspan.0, vspan.1 - vspan.0));
                        }
                    }
                }
                return Ok(last);
            }
        }
    }
    Ok(None)
}

/// Walk the graph: find the declared window, every site carrying it, the table
/// capacity, and any weight that would make patching unsafe.
pub fn inspect(buf: &[u8]) -> Result<GraphReport> {
    let mut model_fields = Vec::new();
    fields_of(buf, (0, buf.len()), &mut model_fields)?;
    let graph = model_fields
        .iter()
        .find(|(f, _, _)| *f == MODEL_GRAPH)
        .map(|(_, _, s)| *s)
        .ok_or_else(|| refuse("ModelProto has no graph"))?;

    let mut gfields = Vec::new();
    fields_of(buf, graph, &mut gfields)?;

    let declared = match gfields
        .iter()
        .find(|(f, _, _)| *f == FIELD_INPUT)
        .map(|(_, _, s)| *s)
        .and_then(|vi| last_dim(buf, vi).ok().flatten())
    {
        Some((v, _, _)) => v,
        None => return Err(refuse("graph input has no static last dimension")),
    };

    let mut report = GraphReport {
        declared_window: declared,
        sites: Vec::new(),
        capacity: None,
        value_info_records: 0,
        nodes: 0,
        initializers: 0,
    };

    for &(field, _, span) in gfields.iter() {
        match field {
            FIELD_NODE => {
                report.nodes += 1;
                let (name, op, attrs) = read_node(buf, span)?;
                if op != "Constant" {
                    continue;
                }
                for attr in attrs {
                    let (an, value) = read_constant_attr(buf, attr)?;
                    let Some(t) = value else { continue };
                    if !is_int64_tensor(buf, t)? {
                        continue;
                    }
                    let label = if name.is_empty() {
                        attr_label(&an)
                    } else {
                        name.clone()
                    };
                    for (v, at, len) in int64_slots(buf, t)? {
                        if v == declared {
                            report.sites.push(Site {
                                offset: at,
                                len,
                                kind: SiteKind::Constant,
                                path: label.clone(),
                                value: v,
                            });
                        }
                    }
                }
            }
            FIELD_INITIALIZER => {
                report.initializers += 1;
                // The normalisation table is a FLOAT tensor, so this check runs
                // before the int64 filter below, not after it.
                let name = tensor_name(buf, span)?;
                if name.contains("window_sum_inv") {
                    report.capacity = table_capacity(buf, span)?;
                }
                if !is_int64_tensor(buf, span)? {
                    continue;
                }
                for (v, _at, _) in int64_slots(buf, span)? {
                    if v == declared {
                        return Err(refuse(&format!(
                            "initializer '{name}' carries the window value {declared} as data; \
                             patching here would edit weights, not shape"
                        )));
                    }
                }
            }
            FIELD_INPUT | FIELD_OUTPUT => {
                let io = if field == FIELD_INPUT {
                    "input"
                } else {
                    "output"
                };
                if let Some((v, at, len)) = last_dim(buf, span)? {
                    if v == declared {
                        report.sites.push(Site {
                            offset: at,
                            len,
                            kind: SiteKind::Dimension,
                            path: io.to_string(),
                            value: v,
                        });
                    } else {
                        return Err(refuse(&format!(
                            "graph {io} declares {v}, which is not the input window {declared}"
                        )));
                    }
                }
            }
            13 => report.value_info_records += 1,
            _ => {}
        }
    }

    if report.sites.iter().all(|s| s.kind != SiteKind::Constant) {
        return Err(refuse(&format!(
            "no Constant carries the window {declared}; this graph does not bake its window \
             the way a Mel-Band RoFormer export does"
        )));
    }
    Ok(report)
}

fn attr_label(name: &str) -> String {
    if name.is_empty() {
        "Constant.value".to_string()
    } else {
        name.to_string()
    }
}

fn read_node(buf: &[u8], node: Span) -> Result<(String, String, Vec<Span>)> {
    let mut fields = Vec::new();
    fields_of(buf, node, &mut fields)?;
    let mut name = String::new();
    let mut op = String::new();
    let mut attrs = Vec::new();
    for (f, _, span) in fields {
        match f {
            NODE_NAME => name = string_of(buf, span),
            NODE_OP_TYPE => op = string_of(buf, span),
            NODE_ATTRIBUTE => attrs.push(span),
            _ => {}
        }
    }
    Ok((name, op, attrs))
}

/// `(attribute name, its tensor if it has one)`.
fn read_constant_attr(
    buf: &[u8],
    attr: (usize, usize),
) -> Result<(String, Option<(usize, usize)>)> {
    let mut fields = Vec::new();
    fields_of(buf, attr, &mut fields)?;
    let mut name = String::new();
    let mut tensor = None;
    for (f, _, span) in fields {
        match f {
            ATTR_NAME => name = string_of(buf, span),
            ATTR_TENSOR => tensor = Some(span),
            // `ints` holds repeated int64 directly on the attribute; the exports
            // we measured put the value in a tensor instead, but refuse silently
            // missing it would be worse than reporting it.
            ATTR_INTS => {
                return Err(refuse(
                    "a Constant used AttributeProto.ints rather than a tensor value; \
                     this locator does not know how to rewrite that shape of export",
                ));
            }
            _ => {}
        }
    }
    Ok((name, tensor))
}

fn table_capacity(buf: &[u8], tensor: (usize, usize)) -> Result<Option<usize>> {
    let mut fields = Vec::new();
    fields_of(buf, tensor, &mut fields)?;
    for (f, wire, span) in fields {
        if f != TENSOR_DIMS {
            continue;
        }
        // dims is repeated int64: packed or unpacked.
        let slots = match wire {
            2 => {
                let mut v = Vec::new();
                let mut i = span.0;
                while i < span.1 {
                    let (n, after) = read_varint(buf, i, span.1)?;
                    v.push(n as usize);
                    i = after;
                }
                v
            }
            0 => vec![read_varint(buf, span.0, span.1)?.0 as usize],
            _ => return Ok(None),
        };
        let len = slots.first().copied().unwrap_or(0);
        return Ok(len.checked_sub(N_FFT));
    }
    Ok(None)
}

/// Rewrite the baked window in place. `buf` must be the whole model file; it is
/// modified in place only if every check passes.
///
/// Returns the sites that were rewritten. Refusals come with a reason and with
/// `buf` untouched, which is the property callers rely on: a failed patch must
/// not leave a half-patched buffer that would still load and quietly compute the
/// wrong thing.
pub fn patch_window(buf: &mut [u8], target: usize) -> Result<GraphReport> {
    if target == 0 || !target.is_multiple_of(HOP) {
        return Err(Error::Model {
            detail: format!("window {target} is not a positive multiple of the hop {HOP}"),
        });
    }
    let report = inspect(buf)?;
    let from = report.declared_window as usize;
    if target == from {
        return Ok(report);
    }
    if target > from {
        let cap = report.capacity.unwrap_or(0);
        return Err(Error::Model {
            detail: format!(
                "growing the window from {from} to {target} needs the iSTFT normalisation table \
                 regenerated (baked capacity {cap} samples); only shrinking is a shape edit"
            ),
        });
    }
    if let Some(cap) = report.capacity {
        if target > cap {
            return Err(Error::Model {
                detail: format!("window {target} exceeds the baked capacity {cap}"),
            });
        }
    }
    // Varints must stay the same width, or every offset after them shifts.
    for site in report
        .sites
        .iter()
        .filter(|s| s.kind == SiteKind::Dimension)
    {
        let need = varint_len(target as u64);
        if need != site.len {
            return Err(Error::Model {
                detail: format!(
                    "{} dimension encodes {from} in {} byte(s) but {target} needs {need}; \
                     use tools/reduce_window.py, which re-serialises the graph",
                    site.path, site.len
                ),
            });
        }
    }

    for site in &report.sites {
        match site.kind {
            SiteKind::Constant => {
                buf[site.offset..site.offset + 8].copy_from_slice(&(target as i64).to_le_bytes());
            }
            SiteKind::Dimension => {
                let encoded = varint_bytes(target as u64);
                buf[site.offset..site.offset + site.len].copy_from_slice(&encoded);
            }
        }
    }
    let mut after = inspect(buf)?;
    after.capacity = report.capacity;
    if after.declared_window != target as i64 {
        return Err(Error::Model {
            detail: format!(
                "patch reported success but the graph still declares {}",
                after.declared_window
            ),
        });
    }
    Ok(after)
}

fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

fn varint_bytes(mut v: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

/// The graph's own name, when it has one; used in logs so a user can tell which
/// file they are looking at.
pub fn graph_name(buf: &[u8]) -> Option<String> {
    let mut model_fields = Vec::new();
    fields_of(buf, (0, buf.len()), &mut model_fields).ok()?;
    let graph = model_fields.iter().find(|(f, _, _)| *f == MODEL_GRAPH)?.2;
    let mut gfields = Vec::new();
    fields_of(buf, graph, &mut gfields).ok()?;
    gfields
        .iter()
        .find(|(f, _, _)| *f == GRAPH_NAME)
        .map(|(_, _, s)| string_of(buf, *s))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a tiny encoder, so the tests describe bytes rather than trust a parser

    fn varint(mut v: u64) -> Vec<u8> {
        let mut o = Vec::new();
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                o.push(b);
                return o;
            }
            o.push(b | 0x80);
        }
    }

    fn key(field: u64, wire: u8) -> Vec<u8> {
        varint((field << 3) | wire as u64)
    }

    fn ld(field: u64, payload: Vec<u8>) -> Vec<u8> {
        let mut o = key(field, 2);
        o.extend(varint(payload.len() as u64));
        o.extend(payload);
        o
    }

    fn vi(field: u64, v: u64) -> Vec<u8> {
        let mut o = key(field, 0);
        o.extend(varint(v));
        o
    }

    /// TensorProto: INT64, one dim = len, raw_data = 8 bytes per value.
    fn int64_tensor(name: &str, values: &[i64]) -> Vec<u8> {
        // dims is a packed repeated int64: one varint per element, length-prefixed.
        let dims = varint(values.len() as u64);
        let mut raw = Vec::new();
        for v in values {
            raw.extend(v.to_le_bytes());
        }
        let mut o = ld(TENSOR_DIMS, dims);
        o.extend(vi(TENSOR_DATA_TYPE, INT64 as u64));
        if !name.is_empty() {
            o.extend(ld(TENSOR_NAME, name.as_bytes().to_vec()));
        }
        o.extend(ld(TENSOR_RAW_DATA, raw));
        o
    }

    fn f32_tensor(name: &str, words: usize) -> Vec<u8> {
        let mut o = ld(TENSOR_DIMS, varint(words as u64));
        o.extend(vi(TENSOR_DATA_TYPE, 1)); // FLOAT
        o.extend(ld(TENSOR_NAME, name.as_bytes().to_vec()));
        o.extend(ld(TENSOR_RAW_DATA, vec![0u8; words * 4]));
        o
    }

    fn constant(name: &str, values: &[i64]) -> Vec<u8> {
        let attr = ld(ATTR_NAME, b"value".to_vec());
        let mut attr = attr;
        attr.extend(ld(ATTR_TENSOR, int64_tensor("", values)));
        let mut o = ld(NODE_NAME, name.as_bytes().to_vec());
        o.extend(ld(NODE_OP_TYPE, b"Constant".to_vec()));
        o.extend(ld(NODE_ATTRIBUTE, attr));
        o
    }

    fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for d in dims {
            let dim = vi(DIM_VALUE, *d as u64);
            shape.extend(ld(SHAPE_DIM, dim));
        }
        let tt = {
            let mut o = vi(1, 1); // elem_type FLOAT
            o.extend(ld(TT_SHAPE, shape));
            o
        };
        let ty = ld(TYPE_TENSOR, tt);
        let mut o = ld(1, name.as_bytes().to_vec());
        o.extend(ld(VI_TYPE, ty));
        o
    }

    fn graph(
        input: Vec<u8>,
        nodes: Vec<Vec<u8>>,
        inits: Vec<Vec<u8>>,
        extra: Vec<Vec<u8>>,
    ) -> Vec<u8> {
        let mut o = Vec::new();
        for n in nodes {
            o.extend(ld(FIELD_NODE, n));
        }
        for i in inits {
            o.extend(ld(FIELD_INITIALIZER, i));
        }
        o.extend(ld(FIELD_INPUT, input));
        for x in extra {
            o.extend(x);
        }
        o
    }

    fn model(graph_payload: Vec<u8>) -> Vec<u8> {
        let mut o = vi(1, 9); // ir_version
        o.extend(ld(MODEL_GRAPH, graph_payload));
        o
    }

    const W: usize = 352_800;

    fn fixture() -> Vec<u8> {
        let nodes = vec![
            constant("/Constant", &[2, 1, W as i64]),
            constant("/istft/Constant_23", &[W as i64]),
            constant("/istft/Constant_29", &[W as i64]),
            constant("/Constant_35", &[W as i64]),
            constant("/stft/Constant_4", &[2048]),
        ];
        let inits = vec![
            int64_tensor("band_start_end", &[0, 12, 64]),
            f32_tensor("istft.window_sum_inv", W + N_FFT),
            f32_tensor("stft.cos_kernel", 1024),
        ];
        let g = graph(
            value_info("mix", &[1, 2, W as i64]),
            nodes,
            inits,
            vec![
                ld(FIELD_OUTPUT, value_info("sources", &[1, 1, 2, W as i64])),
                ld(13, value_info("/Reshape_10", &[1, 2, W as i64])),
            ],
        );
        model(g)
    }

    #[test]
    fn inspect_finds_the_declared_window_and_the_four_constants() {
        let buf = fixture();
        let r = inspect(&buf).expect("inspect");
        assert_eq!(r.declared_window, W as i64);
        assert_eq!(r.nodes, 5, "five Constant nodes in the fixture");
        assert_eq!(r.initializers, 3);
        assert_eq!(r.value_info_records, 1);
        assert_eq!(r.constants().count(), 4, "{:?}", r.sites);
        // /Constant holds [2,1,W]: exactly one slot of it equals the window.
        let paths: Vec<&str> = r.constants().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "/Constant",
                "/istft/Constant_23",
                "/istft/Constant_29",
                "/Constant_35"
            ]
        );
        assert_eq!(r.dimensions().count(), 2, "input + output");
        assert_eq!(r.capacity, Some(W));
    }

    #[test]
    fn patching_to_a_shorter_window_rewrites_only_shape_sites() {
        let mut buf = fixture();
        let before = buf.clone();
        let r = patch_window(&mut buf, 176_400).expect("patch");
        assert_eq!(r.declared_window, 176_400);
        let mut changed = 0;
        for (i, (a, b)) in before.iter().zip(buf.iter()).enumerate() {
            if a != b {
                changed += 1;
                assert!(
                    r.sites
                        .iter()
                        .any(|s| i >= s.offset && i < s.offset + s.len),
                    "byte {i} changed but is not a recorded site"
                );
            }
        }
        assert_eq!(
            buf.len(),
            before.len(),
            "a length-neutral patch must stay length-neutral"
        );
        // Six sites carry the window: four constants and two dimensions. The exact
        // byte count is not asserted (an 8-byte little-endian word differs only in
        // its low bytes), but every moved byte must belong to one of them.
        assert_eq!(r.sites.len(), 6, "{:?}", r.sites);
        assert!(
            changed > 0 && changed <= 6 * 8,
            "implausible number of moved bytes: {changed}"
        );
        // The normalisation table and every other constant are untouched.
        let r2 = inspect(&buf).expect("re-inspect");
        assert!(r2.constants().all(|s| s.value == 176_400));
    }

    #[test]
    fn growing_is_refused_and_changes_nothing() {
        let mut buf = fixture();
        let before = buf.clone();
        let e = patch_window(&mut buf, 705_600).expect_err("grow must refuse");
        assert!(e.to_string().contains("regenerated"), "{e}");
        assert_eq!(buf, before, "a refusal must not leave a patched buffer");
    }

    #[test]
    fn a_window_inside_a_weight_is_a_hard_no() {
        let mut nodes = vec![constant("/Constant", &[W as i64])];
        nodes.push(constant("/other", &[4]));
        let inits = vec![int64_tensor("something_suspicious", &[0, W as i64])];
        let g = graph(
            value_info("mix", &[1, 2, W as i64]),
            nodes,
            inits,
            vec![ld(
                FIELD_OUTPUT,
                value_info("sources", &[1, 1, 2, W as i64]),
            )],
        );
        let buf = model(g);
        let e = inspect(&buf).expect_err("must refuse");
        assert!(e.to_string().contains("would edit weights"), "{e}");
    }

    #[test]
    fn a_graph_that_does_not_bake_its_window_is_not_touched() {
        let g = graph(
            value_info("mix", &[1, 2, W as i64]),
            vec![constant("/unrelated", &[7, 8, 9])],
            vec![],
            vec![ld(
                FIELD_OUTPUT,
                value_info("sources", &[1, 1, 2, W as i64]),
            )],
        );
        let buf = model(g);
        let e = inspect(&buf).expect_err("no window-carrying constant");
        assert!(e.to_string().contains("does not bake its window"), "{e}");
    }

    #[test]
    fn non_hop_aligned_and_zero_windows_are_refused_before_parsing() {
        let mut buf = fixture();
        assert!(patch_window(&mut buf, 100_000)
            .unwrap_err()
            .to_string()
            .contains("multiple"));
        assert!(patch_window(&mut buf, 0)
            .unwrap_err()
            .to_string()
            .contains("multiple"));
    }

    #[test]
    fn a_dimension_that_would_need_more_bytes_defers_to_the_offline_tool() {
        // A dimension varint that would change width cannot be rewritten in place:
        // every offset after it shifts. 441 samples is hop-aligned and encodes in
        // 2 bytes where 352_800 takes 3, so this is the case that must refuse:
        // and name the offline tool that re-serialises instead.
        let g = graph(
            value_info("mix", &[1, 2, W as i64]),
            vec![constant("/Constant", &[W as i64])],
            vec![f32_tensor("istft.window_sum_inv", W + N_FFT)],
            vec![ld(
                FIELD_OUTPUT,
                value_info("sources", &[1, 1, 2, W as i64]),
            )],
        );
        let mut buf = model(g);
        let e = patch_window(&mut buf, 441).expect_err("varint width differs");
        assert!(e.to_string().contains("reduce_window.py"), "{e}");
    }

    /// The field number that decided whether any real export could be patched at
    /// all. `AttributeProto.t` is 5; 14 is `tp`, a `TypeProto`. A fixture written
    /// with the wrong number passes against the wrong parser, so this test names
    /// both and asserts which one the graph actually speaks.
    #[test]
    fn the_constant_tensor_attribute_is_field_five_not_fourteen() {
        let mut attr = ld(ATTR_NAME, b"value".to_vec());
        attr.extend(ld(ATTR_TENSOR, int64_tensor("", &[W as i64])));
        attr.extend(vi(20, 4)); // AttributeProto.type = TENSOR
        let mut node = ld(NODE_NAME, b"/Constant".to_vec());
        node.extend(ld(NODE_OP_TYPE, b"Constant".to_vec()));
        node.extend(ld(NODE_ATTRIBUTE, attr));
        let buf = model(graph(
            value_info("mix", &[1, 2, W as i64]),
            vec![node],
            vec![],
            vec![ld(
                FIELD_OUTPUT,
                value_info("sources", &[1, 1, 2, W as i64]),
            )],
        ));
        let r = inspect(&buf).expect("a Constant value at field 5 is a window site");
        assert_eq!(r.constants().count(), 1, "{:?}", r.sites);
        assert_eq!(r.constants().next().unwrap().value, W as i64);
        assert_eq!(ATTR_TENSOR, 5, "AttributeProto.t");

        // The same bytes with the tensor parked at 14 (`tp`) are a graph this
        // locator must *not* recognize: nothing there carries the window.
        let mut attr = ld(ATTR_NAME, b"value".to_vec());
        attr.extend(ld(14, int64_tensor("", &[W as i64])));
        let mut node = ld(NODE_NAME, b"/Constant".to_vec());
        node.extend(ld(NODE_OP_TYPE, b"Constant".to_vec()));
        node.extend(ld(NODE_ATTRIBUTE, attr));
        let buf = model(graph(
            value_info("mix", &[1, 2, W as i64]),
            vec![node],
            vec![],
            vec![ld(
                FIELD_OUTPUT,
                value_info("sources", &[1, 1, 2, W as i64]),
            )],
        ));
        let e = inspect(&buf).expect_err("field 14 is a TypeProto, not a tensor");
        assert!(e.to_string().contains("does not bake its window"), "{e}");
    }

    #[test]
    fn varint_widths_agree_with_the_patch_rule() {
        assert_eq!(varint_len(352_800), 3);
        assert_eq!(varint_len(176_400), 3);
        assert_eq!(varint_len(441), 2);
        assert_eq!(varint_len(127), 1);
        assert_eq!(varint_len(128), 2);
        // 176_400 = 0b10_1011_0001_0001_0000 → 7-bit groups 10 / 98 / 16.
        assert_eq!(varint_bytes(176_400), vec![0x90, 0xe2, 0x0a]);
        // Round-trip, so a wrong literal cannot pass by agreeing with itself.
        let enc = varint_bytes(176_400);
        assert_eq!(read_varint(&enc, 0, enc.len()).unwrap().0, 176_400);
    }

    #[test]
    fn a_truncated_or_malformed_message_is_an_error_not_a_panic() {
        let buf = fixture();
        // Chop inside the graph: every nested range must stay within its parent.
        let cut = buf.len() - 5;
        assert!(inspect(&buf[..cut]).is_err());
        assert!(inspect(&[]).is_err());
        assert!(inspect(&[0xff, 0xff, 0xff]).is_err());
    }
}
