//! Streaming input: decode a track in bounded blocks, resample it in bounded
//! blocks.
//!
//! The separation loop itself reads the mix one window at a time (that is
//! [`WavSource`], which lives next to the writer it pairs with and is re-exported
//! here because the input side of this crate is meant to be found in one place).
//! Everything below is for the *other* shape of read a caller needs: the whole
//! track once, mono, usually at a rate far below the file's own — a level
//! meter, a voice-activity pre-pass, a checksum of what was actually decoded.
//!
//! Two allocations are what a naive version of that read costs, and both scale
//! with the track:
//!
//! 1. the decoded row at the *source* rate. 44.1 kHz stereo as `f32` is
//!    44_100 · 2 · 4 = 352_800 bytes per second, i.e. ~1.27 GB per hour of
//!    material, and the mono fold needs a second row to write the result into;
//! 2. the resampler's own history. `rubato::SincFixedIn::new(ratio, 2.0, _,
//!    chunk_size, channels)` allocates a `chunk_size + 2·sinc_len` ring buffer
//!    *per channel*, and the one-shot calling style passes the whole waveform
//!    length as `chunk_size` — so the source-rate row exists twice for the
//!    duration of the resample.
//!
//! [`stream_mono`] removes (1) by folding each interleaved block to mono as it
//! is read, and [`StreamingSinc`] removes (2) by giving the filter a block of
//! [`READ_CHUNK_FRAMES`] frames instead of one of the length of the track: the
//! history buffer then costs 65_536 + 256 floats, whatever the file is.
//! [`read_mono`] is the pair wired together, and it is what the engines'
//! pre-flight reads use.
//!
//! What streaming the resample costs is stated in [`resample_chunked`], and it
//! is not nothing: at a ratio `f64` cannot represent exactly the chunked row is
//! not bit-identical to the one-shot row. That difference is documented,
//! measured and pinned by a test rather than buried.

use std::io::{Read, Seek};
use std::path::Path;

use hound::{SampleFormat, WavReader};
use rubato::Resampler;

use crate::error::{Error, Result};

/// Sample rate the Mel-Band RoFormer checkpoints are fixed at.
pub const SAMPLE_RATE: u32 = 44_100;

/// Frames per block on the way from disk to the fold, and from the fold to the
/// resampler.
///
/// 64 k source frames is 256 KB of `f32`, three orders of magnitude below the
/// row it replaces, and still large enough that no per-block cost shows up in
/// the wall clock: the filter's fixed overhead (`sinc_len` of history per
/// block, plus a call that touches a whole block at once) amortises to
/// nothing at this size, while a block small enough to matter in the profile
/// would also be small enough to make the seek-restore pattern of a real file
/// the dominant cost.
pub const READ_CHUNK_FRAMES: usize = 65_536;

/// The windowed, stereo, 44.1 kHz-normalised reader.
///
/// Defined in [`crate::stream`] beside [`crate::stream::StemWriter`] because the
/// two are one pass over the same track; re-exported so `audio::` is the single
/// place to look for reading.
pub use crate::stream::WavSource;

/// The `SincFixedIn` filter both resamplers here are built with, in one place so
/// a comparison between the two cannot accidentally compare two different
/// filters.
fn sinc_params() -> rubato::SincInterpolationParameters {
    rubato::SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.913,
        interpolation: rubato::SincInterpolationType::Cubic,
        oversampling_factor: 128,
        window: rubato::WindowFunction::Hann2,
    }
}

/// What a WAV header claims, checked enough to be pumped.
struct Header {
    channels: usize,
    sample_rate: u32,
    frames: usize,
}

/// Read and validate the header only.
///
/// The three guards are the ones that keep a corrupt header from becoming a
/// panic or a silent zero-length result further down: a channel count of zero
/// divides the frame count and indexes the fold, a sample rate of zero makes the
/// resample ratio zero, and an unsupported depth would otherwise be reinterpreted
/// as whatever type happens to compile.
fn read_header<R: Read + Seek>(reader: &WavReader<R>, path: &Path) -> Result<Header> {
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(wav(path, "zero channels"));
    }
    if spec.sample_rate == 0 {
        return Err(wav(path, "sample rate is 0 Hz"));
    }
    let ok = match spec.sample_format {
        SampleFormat::Int => matches!(spec.bits_per_sample, 16 | 24 | 32),
        SampleFormat::Float => spec.bits_per_sample == 32,
    };
    if !ok {
        return Err(wav(
            path,
            format!(
                "unsupported sample format: {} bit {} PCM (this reader takes 16-, 24- and 32-bit \
                 integers and 32-bit floats)",
                spec.bits_per_sample,
                match spec.sample_format {
                    SampleFormat::Int => "integer",
                    SampleFormat::Float => "float",
                }
            ),
        ));
    }
    // `len()` counts values across all channels, so dividing by the channel count
    // gives frames — and it is available before a single sample is read, which is
    // what lets every buffer here be allocated once instead of grown.
    Ok(Header {
        channels,
        sample_rate: spec.sample_rate,
        frames: reader.len() as usize / channels,
    })
}

fn wav(path: &Path, detail: impl Into<String>) -> Error {
    Error::Wav {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

/// Decode a WAV to mono at *its own* sample rate, `chunk` frames at a time, and
/// hand each block to `sink`.
///
/// Nothing accumulates: peak cost is one block plus whatever the caller keeps.
/// A caller that needs the whole row uses [`read_mono`]; a caller that needs a
/// number out of the track (peak level, a checksum) needs this.
///
/// The mono fold is the loader's own arithmetic, expression for expression,
/// including `chunks_exact`'s habit of dropping a truncated trailing frame. That
/// is not an oversight: a different rounding here moves any downstream boundary
/// detection that reads the row, so it is pinned by
/// `every_channel_count_and_depth_folds_to_the_same_mono` rather than improved.
pub fn stream_mono<F>(path: &Path, chunk: usize, mut sink: F) -> Result<()>
where
    F: FnMut(&[f32]) -> Result<()>,
{
    let reader = WavReader::open(path).map_err(|e| wav(path, format!("open: {e}")))?;
    let header = read_header(&reader, path)?;
    let mut reader = reader;
    // Monomorphised per sample type, like the reference decode it replaces: a
    // boxed iterator would put a per-sample virtual call on the side that does
    // the bulk of the work.
    let spec = reader.spec();
    match spec.bits_per_sample {
        16 => pump_blocks(
            path,
            reader
                .samples::<i16>()
                .map(|r| r.map(|s| s as f32 / 32768.0)),
            header.channels,
            chunk,
            &mut sink,
        ),
        24 => pump_blocks(
            path,
            reader
                .samples::<i32>()
                .map(|r| r.map(|s| s as f32 / 8_388_608.0)),
            header.channels,
            chunk,
            &mut sink,
        ),
        32 if spec.sample_format == SampleFormat::Float => pump_blocks(
            path,
            reader.samples::<f32>(),
            header.channels,
            chunk,
            &mut sink,
        ),
        32 => pump_blocks(
            path,
            reader
                .samples::<i32>()
                .map(|r| r.map(|s| s as f32 / 2_147_483_648.0)),
            header.channels,
            chunk,
            &mut sink,
        ),
        // Guarded by `read_header`; kept so a widened guard cannot send an
        // unsupported depth through the closest-looking arm instead.
        other => Err(wav(path, format!("unsupported bit depth {other}"))),
    }
}

/// Hand `sink` the mono fold `chunk` frames at a time.
fn pump_blocks<I, F>(
    path: &Path,
    mut samples: I,
    channels: usize,
    chunk: usize,
    sink: &mut F,
) -> Result<()>
where
    I: Iterator<Item = std::result::Result<f32, hound::Error>>,
    F: FnMut(&[f32]) -> Result<()> + ?Sized,
{
    let mut row: Vec<f32> = Vec::with_capacity(chunk);
    loop {
        row.clear();
        // Every block is `chunk × channels` values, so it begins and ends at
        // channel parity 0 exactly as one pass over the whole file would: the
        // frames this hands on are the frames a single pass would have handed on.
        downmix_interleaved(
            path,
            samples.by_ref().take(chunk * channels),
            channels,
            &mut row,
        )?;
        if row.is_empty() {
            return Ok(());
        }
        sink(&row)?;
    }
}

/// Average an interleaved sample stream down to mono, one frame at a time.
fn downmix_interleaved<I>(path: &Path, it: I, channels: usize, mono: &mut Vec<f32>) -> Result<()>
where
    I: Iterator<Item = std::result::Result<f32, hound::Error>>,
{
    let mut pair_left = 0.0f32;
    let mut acc = 0.0f32;
    let mut filled = 0usize;
    for v in it {
        let v = v.map_err(|e| wav(path, format!("sample stream: {e}")))?;
        filled += 1;
        match channels {
            1 => mono.push(v),
            2 => {
                if filled == 1 {
                    pair_left = v;
                } else {
                    mono.push((pair_left + v) * 0.5);
                    filled = 0;
                }
            }
            _ => {
                acc += v;
                if filled == channels {
                    mono.push(acc / channels as f32);
                    acc = 0.0;
                    filled = 0;
                }
            }
        }
    }
    Ok(())
}

/// Decode a WAV to mono float32 at `target_sr`, resampling in blocks if needed.
///
/// This is the whole-track form of [`stream_mono`] and the only reader here that
/// returns a row, so its output does scale with the track — at 16 kHz mono that
/// is 64 bytes per millisecond, which is the price of asking for the whole thing
/// rather than a bug in the pump.
///
/// A file shorter than one block is resampled *as* one block, which is the
/// one-shot shape and therefore bit-identical to [`resample_mono`]. Past one
/// block, the bounded tail in [`StreamingSinc::push`] applies and the row carries
/// the documented last-bit differences.
pub fn read_mono(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
    read_mono_in_chunks(path, target_sr, READ_CHUNK_FRAMES)
}

/// [`read_mono`] with an explicit block size.
///
/// The public entry point uses [`READ_CHUNK_FRAMES`]; this exists so a test can
/// force a ten-thousand-sample file through several blocks instead of one, which
/// is the only way to exercise the tail behaviour without writing a hundred
/// megabytes of fixture.
pub fn read_mono_in_chunks(path: &Path, target_sr: u32, chunk: usize) -> Result<Vec<f32>> {
    let reader = WavReader::open(path).map_err(|e| wav(path, format!("open: {e}")))?;
    let header = read_header(&reader, path)?;

    // A file shorter than one block is resampled as a single block: `push` on a
    // short piece returns the filter's response to its own zero-pad, which on a
    // 0.2 s clip is a row far longer than the clip. Handing it over as one piece
    // is the one-shot's own shape, so the row is bit-identical to
    // `resample_mono` on the same fold — and nothing is given up, because at this
    // size no source-rate row ever existed to stream around.
    let piece = if header.frames == 0 {
        chunk
    } else {
        chunk.min(header.frames)
    };

    if header.sample_rate == target_sr {
        // No resample: hand back the fold unchanged, so only the block is bounded.
        let mut out: Vec<f32> = Vec::with_capacity(header.frames);
        stream_mono(path, piece, |row| {
            out.extend_from_slice(row);
            Ok(())
        })?;
        return Ok(out);
    }

    let mut sinc = StreamingSinc::new(header.sample_rate, target_sr, piece, header.frames)?;
    stream_mono(path, piece, |row| sinc.push(row))?;
    Ok(sinc.into_output())
}

/// rubato's windowed sinc driven by a caller that hands it frames as they
/// arrive, instead of one piece the length of the track.
///
/// Handing [`resample_mono`] a whole track makes
/// `SincFixedIn::new(ratio, 2.0, _, waveform.len(), 1)` allocate a
/// `chunk_size + 2·sinc_len` history buffer — a second copy of the source-rate
/// row, on top of the row itself. This type gives the filter
/// [`READ_CHUNK_FRAMES`] frames at a time, so the history buffer has a fixed size
/// and the source row never exists at all.
pub struct StreamingSinc {
    resampler: rubato::SincFixedIn<f32>,
    buf: Vec<f32>,
    out: Vec<f32>,
}

impl StreamingSinc {
    /// `in_frames` is the frame count the caller already knows from the header;
    /// it sizes the output row in one allocation. A `Vec` left to grow by
    /// doubling holds 1.5× its finished size at the moment it grows, which for a
    /// row measured in hundreds of megabytes is an avoidable spike of the same
    /// order.
    pub fn new(from_sr: u32, to_sr: u32, chunk: usize, in_frames: usize) -> Result<Self> {
        if from_sr == 0 || to_sr == 0 || chunk == 0 {
            return Err(Error::Resample {
                detail: format!("invalid resample request {from_sr}->{to_sr} chunk {chunk}"),
            });
        }
        let ratio = to_sr as f64 / from_sr as f64;
        let resampler = rubato::SincFixedIn::<f32>::new(ratio, 2.0, sinc_params(), chunk, 1)
            .map_err(|e| Error::Resample {
                detail: format!("cannot build the sinc filter for {from_sr}->{to_sr}: {e}"),
            })?;
        // `output_frames_next()` is `chunk·ratio + 10` for this type; take the
        // larger of that and `chunk` so the scratch also fits an upsampling ratio.
        let cap = (chunk as f64 * ratio.max(1.0)) as usize + 64;
        Ok(Self {
            resampler,
            buf: vec![0.0f32; cap],
            out: Vec::with_capacity((in_frames as f64 * ratio) as usize + 64),
        })
    }

    /// One piece of the source row: `chunk` frames, with a short final piece.
    ///
    /// **`SincFixedIn` has no drain mode, and the short final piece *is* the
    /// flush.** `process_partial_into_buffer` zero-pads a short chunk internally
    /// and runs the real sinc tail through it, so every input frame reaches the
    /// output. The two ways to get this wrong were both shipped once:
    ///
    /// * calling `process_into_buffer` (the full-chunk form) for the last piece
    ///   and skipping it when the piece is short — that drops the tail of the
    ///   track, up to a whole block of audio;
    /// * looping `process_partial_into_buffer(None)` afterwards to "drain" the
    ///   delay line — which feeds the filter a full chunk of silence every time
    ///   and emits its response to that silence as if it were audio. There is
    ///   nothing to wait for: the call never returns `(0, 0)`, so a drain loop has
    ///   no terminating condition and every iteration adds about `chunk·ratio`
    ///   samples. Sixteen iterations — the number the first version of this code
    ///   ran before giving up and taking what it had — invent a few hundred
    ///   thousand samples on a three-minute input, and a probe that compares only
    ///   a common prefix will not see any of it. See
    ///   `draining_past_the_short_final_chunk_invents_samples`.
    ///
    /// So: push every piece, push no `None`. A caller that must land exactly on
    /// the one-shot length truncates the roughly `chunk·ratio` samples a full
    /// block produces past the true end; it must not invent more.
    pub fn push(&mut self, frames: &[f32]) -> Result<()> {
        let ins: [&[f32]; 1] = [frames];
        let filled = self
            .resampler
            .process_partial_into_buffer(Some(&ins), &mut [&mut self.buf], None)
            .map_err(|e| Error::Resample {
                detail: format!("sinc block failed: {e}"),
            })?;
        self.out.extend_from_slice(&self.buf[..filled.1]);
        Ok(())
    }

    /// Frames produced so far.
    pub fn output_frames(&self) -> usize {
        self.out.len()
    }

    pub fn into_output(self) -> Vec<f32> {
        self.out
    }
}

/// High-quality mono resample: windowed sinc, one piece the length of the input.
///
/// Kept as the oracle for [`resample_chunked`] rather than as the production
/// path: it allocates the filter's history buffer at `waveform.len()`, i.e. a
/// second copy of the row, which is precisely the term [`StreamingSinc`] removes.
/// A caller with a short buffer in hand (a window, a clip) is the right user of
/// this one.
pub fn resample_mono(waveform: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    if from_sr == to_sr {
        return Ok(waveform.to_vec());
    }
    if from_sr == 0 || to_sr == 0 {
        return Err(Error::Resample {
            detail: format!("invalid resample request {from_sr}->{to_sr}"),
        });
    }

    let mut resampler = rubato::SincFixedIn::<f32>::new(
        to_sr as f64 / from_sr as f64,
        2.0, // oversampling ratio (anti-aliasing headroom)
        sinc_params(),
        waveform.len().max(1),
        1,
    )
    .map_err(|e| Error::Resample {
        detail: format!("cannot build the sinc filter for {from_sr}->{to_sr}: {e}"),
    })?;

    let mut output = resampler
        .process(&[waveform], None)
        .map_err(|e| Error::Resample {
            detail: format!("resampling {from_sr}->{to_sr} failed: {e}"),
        })?;

    let row = output.pop().ok_or_else(|| Error::Resample {
        detail: "resampling produced no channel".to_string(),
    })?;
    log::info!(
        "[audio] resampled {} Hz / {} samples -> {} Hz / {} samples (mono, one-shot)",
        from_sr,
        waveform.len(),
        to_sr,
        row.len(),
    );
    Ok(row)
}

/// [`resample_mono`] driven in `chunk`-frame pieces instead of one piece the
/// length of the input.
///
/// **The last-bit story, stated rather than hidden.** For an exact
/// input/output ratio — 48 kHz → 16 kHz, where the filter's internal step is
/// `1/ratio` = 3.0 and every frame lands on an integer phase — chunking changes
/// nothing and the two rows agree bit for bit at any block size.
///
/// For a ratio `f64` cannot represent — 44.1 kHz → 16 kHz, whose step is
/// 2.75625 — they cannot agree, and this is a property of the filter rather than
/// of this function. rubato's `SincFixedIn` advances an output position by
/// accumulating that step (`idx += t_ratio`) and, at the end of every call,
/// re-anchors it by subtracting the block size (`last_index = idx -
/// chunk_size`). One-shot mode makes that subtraction once, at the end of the
/// track. Chunked mode does it every `chunk_size` frames, and each subtraction
/// rounds differently from the accumulation it undoes, so the phase the sinc is
/// sampled at drifts by ULPs. Most output samples end up differing by one or two
/// least-significant bits; the worst error stays at the 1e-6 scale against a full
/// 1.0 signal, i.e. better than 100 dB down, which is a long way below the 16-bit
/// quantisation of the file it is written to and below the noise floor of any
/// recording it came from.
///
/// The behaviour is asserted by `chunked_matches_one_shot_on_an_exact_ratio` and
/// `chunked_differs_in_lsbson_an_inexact_ratio`, so a change here cannot pass
/// unnoticed. A caller that needs the one-shot bytes needs the one-shot row:
/// hand it the whole buffer.
///
/// The other difference is length, and it favours the chunked row: the final
/// short piece is zero-padded to a full block, so the row runs up to
/// `chunk·ratio` samples past the one-shot's end. Those samples are the filter's
/// response to the audio cutting off — a couple of dozen samples of ring-out and
/// then exact zeros (see `streaming_resample_of_a_short_file_reports_its_own_tail`).
/// What the overshoot costs a consumer is length, not signal.
pub fn resample_chunked(
    waveform: &[f32],
    from_sr: u32,
    to_sr: u32,
    chunk: usize,
) -> Result<Vec<f32>> {
    // Same identity shortcut as [`resample_mono`]: an in-place request must not
    // be run through the filter, which would resample it by 1.0 and hand back a
    // row that differs in the last bits and runs longer than the input.
    if from_sr == to_sr {
        return Ok(waveform.to_vec());
    }
    let mut sinc = StreamingSinc::new(from_sr, to_sr, chunk, waveform.len())?;
    let mut pos = 0usize;
    while pos < waveform.len() {
        let take = chunk.min(waveform.len() - pos);
        sinc.push(&waveform[pos..pos + take])?;
        pos += take;
    }
    Ok(sinc.into_output())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Scratch fixtures go under the temp dir in a uniquely-named subdirectory,
    /// removed again at the end of each test.
    fn test_dir(name: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rust_roformer_audio_{name}_{}_{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    /// Non-periodic per-channel signals. Periodic test audio hides an off-by-one
    /// frame in the fold — the next frame can carry the same value — which is
    /// exactly the mistake this fold could make, so the generator deliberately
    /// produces a different value for every frame of every channel.
    fn sample_at(frame: usize, channel: usize) -> f32 {
        let f = frame as f32;
        let c = channel as f32;
        ((f * 0.37 + c * 1.13).sin() * 0.45 + (f * 0.0113 - c * 0.7).cos() * 0.45)
            .clamp(-0.95, 0.95)
    }

    /// Write `frames` frames of `channels` × `bits` audio, integer or float.
    fn write_wav(path: &Path, sr: u32, channels: u16, bits: u16, float: bool, frames: usize) {
        let spec = if float {
            hound::WavSpec {
                channels,
                sample_rate: sr,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            }
        } else {
            hound::WavSpec {
                channels,
                sample_rate: sr,
                bits_per_sample: bits,
                sample_format: SampleFormat::Int,
            }
        };
        let mut w = hound::WavWriter::create(path, spec).expect("create fixture wav");
        let full: f32 = match bits {
            16 => 32767.0,
            24 => 8_388_607.0,
            _ => 2_147_483_647.0,
        };
        for f in 0..frames {
            for ch in 0..channels as usize {
                let v = sample_at(f, ch);
                if float {
                    w.write_sample(v).expect("write float sample");
                } else {
                    w.write_sample((v * full) as i32).expect("write sample");
                }
            }
        }
        w.finalize().expect("finalize fixture");
    }

    /// The whole-buffer reader this module replaced: collect every sample, then
    /// fold, then resample in one call. Kept as the equivalence oracle, and only
    /// used on fixtures small enough for it to be a sane thing to do.
    fn read_mono_whole_buffer(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
        let reader = WavReader::open(path).map_err(|e| wav(path, format!("open: {e}")))?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let samples: Vec<f32> = match spec.bits_per_sample {
            16 => reader
                .into_samples::<i16>()
                .map(|r| r.map(|s| s as f32 / 32768.0))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| wav(path, format!("sample stream: {e}")))?,
            24 => reader
                .into_samples::<i32>()
                .map(|r| r.map(|s| s as f32 / 8_388_608.0))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| wav(path, format!("sample stream: {e}")))?,
            32 if spec.sample_format == SampleFormat::Float => reader
                .into_samples::<f32>()
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| wav(path, format!("sample stream: {e}")))?,
            32 => reader
                .into_samples::<i32>()
                .map(|r| r.map(|s| s as f32 / 2_147_483_648.0))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| wav(path, format!("sample stream: {e}")))?,
            other => return Err(wav(path, format!("unsupported bit depth {other}"))),
        };
        let mono: Vec<f32> = match channels {
            1 => samples,
            2 => samples
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| (c[0] + c[1]) * 0.5)
                .collect(),
            n => samples
                .chunks_exact(n)
                .map(|c| c.iter().sum::<f32>() / n as f32)
                .collect(),
        };
        if spec.sample_rate != target_sr {
            resample_mono(&mono, spec.sample_rate, target_sr)
        } else {
            Ok(mono)
        }
    }

    /// How far apart two equal-length rows really are, in terms a listener (or a
    /// downstream F0 / embedding / detection stage) sees rather than in raw bit
    /// patterns: a `to_bits()` distance on samples that both read as ~1e-30 says
    /// nothing about audio.
    fn diff_report(one_shot: &[f32], chunked: &[f32]) -> (usize, f64, f64, f64) {
        let common = one_shot.len().min(chunked.len());
        let differing = (0..common)
            .filter(|&i| one_shot[i].to_bits() != chunked[i].to_bits())
            .count();
        let (mut max_abs, mut max_rel, mut sq_sig, mut sq_err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..common {
            let a = one_shot[i] as f64;
            let e = (a - chunked[i] as f64).abs();
            max_abs = max_abs.max(e);
            sq_sig += a * a;
            sq_err += e * e;
            // Samples that are effectively zero carry no information in a
            // relative figure, so they are skipped rather than dominating it.
            if a.abs() > 1e-4 {
                max_rel = max_rel.max(e / a.abs());
            }
        }
        let snr = if sq_err > 0.0 {
            10.0 * (sq_sig / sq_err).log10()
        } else {
            f64::INFINITY
        };
        (differing, max_abs, max_rel, snr)
    }

    /// The fold replaced a whole-buffer decode, so it must hand the resampler and
    /// any downstream stage bit-identical samples — for every channel count and
    /// every supported depth, at the file's own rate and resampled.
    ///
    /// The fixtures here are all shorter than one block, which is the case the
    /// reader deliberately does *not* stream: it resamples them as a single piece,
    /// so agreement is exact and this test proves both the fold and the
    /// single-block rule at the same time.
    fn fold_matches_the_whole_buffer(channels: u16, bits: u16, float: bool, target_sr: u32) {
        let dir = test_dir("fold");
        let path = dir.join("track.wav");
        let sr = SAMPLE_RATE;
        write_wav(&path, sr, channels, bits, float, 9001);

        let got = read_mono(&path, target_sr).expect("streaming read");
        let want = read_mono_whole_buffer(&path, target_sr).expect("oracle read");
        assert_eq!(
            got.len(),
            want.len(),
            "ch={channels} bits={bits} float={float} {sr}->{target_sr} length"
        );
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g, w,
                "ch={channels} bits={bits} float={float} {sr}->{target_sr} sample {i}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_channel_count_and_depth_folds_to_the_same_mono() {
        // Same rate (no resample) and resampled, mono through quadraphonic, and
        // every depth the reader accepts.
        fold_matches_the_whole_buffer(1, 16, false, SAMPLE_RATE);
        fold_matches_the_whole_buffer(2, 16, false, SAMPLE_RATE);
        fold_matches_the_whole_buffer(1, 16, false, 16_000);
        fold_matches_the_whole_buffer(2, 16, false, 16_000);
        fold_matches_the_whole_buffer(2, 32, false, 16_000); // 32-bit integer
        fold_matches_the_whole_buffer(2, 32, true, 16_000); // 32-bit float
        fold_matches_the_whole_buffer(2, 24, false, 16_000); // 24-bit integer
        fold_matches_the_whole_buffer(3, 16, false, 16_000);
        fold_matches_the_whole_buffer(4, 32, false, 22_050);
    }

    /// A depth this reader does not implement is refused, not reinterpreted as
    /// the closest one that does. The file is built by patching the `bits_per_sample`
    /// field of a real 16-bit WAV rather than by writing an 8-bit one, because the
    /// point is what the *header* claims.
    #[test]
    fn depths_that_are_not_supported_are_rejected_rather_than_guessed() {
        let dir = test_dir("depth");
        let path = dir.join("track8.wav");
        write_wav(&path, SAMPLE_RATE, 2, 16, false, 500);
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            // `fmt ` chunk: bits/sample lives at byte 34 (RIFF+size+WAVE+`fmt `+
            // size+format+channels+rate+byteRate+blockAlign).
            f.seek(SeekFrom::Start(34)).unwrap();
            f.write_all(&8u16.to_le_bytes()).unwrap();
            f.flush().unwrap();
        }
        let err = match read_mono(&path, 16_000) {
            Ok(_) => panic!("an 8-bit track read as if it were supported"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("unsupported sample format"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A truncated file (header promises more than it holds) is a read error, not
    /// a short row: a silent end-of-data here would look like end-of-audio to a
    /// downstream detector.
    #[test]
    fn a_truncated_data_chunk_is_an_error() {
        let dir = test_dir("trunc");
        let path = dir.join("cut.wav");
        write_wav(&path, SAMPLE_RATE, 2, 16, false, 4000);
        let len = std::fs::metadata(&path).unwrap().len();
        // Drop the last frame's worth of payload but leave the header alone.
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 2)
            .unwrap();
        let err = match read_mono(&path, SAMPLE_RATE) {
            Ok(v) => panic!("a truncated track read cleanly as {} samples", v.len()),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("sample stream"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `stream_mono` must hand the sink blocks of exactly the requested size
    /// (plus one short final block), because that is the only thing separating
    /// this reader from the one that allocates a row the length of the track.
    #[test]
    fn blocks_arrive_at_the_requested_size() {
        let dir = test_dir("blocks");
        let path = dir.join("track.wav");
        write_wav(&path, SAMPLE_RATE, 2, 16, false, 4000);
        let mut sizes: Vec<usize> = Vec::new();
        let mut total = 0usize;
        stream_mono(&path, 1000, |row| {
            sizes.push(row.len());
            total += row.len();
            Ok(())
        })
        .expect("stream");
        assert_eq!(sizes, vec![1000, 1000, 1000, 1000]);
        assert_eq!(total, 4000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Several blocks, inexact ratio: the streamed read must stay within the
    /// documented last-bit envelope of the one-shot row and must not come back
    /// *shorter* than it — a short row would mean the final partial block had
    /// been dropped, which is the bug [`StreamingSinc::push`] exists to avoid.
    #[test]
    fn a_multi_block_read_differs_from_the_one_shot_only_in_lsb_s() {
        let dir = test_dir("multiblock");
        let path = dir.join("long.wav");
        let frames = 300_000usize; // ~6.8 s at 44.1 kHz, i.e. five blocks
        write_wav(&path, SAMPLE_RATE, 2, 16, false, frames);

        let streamed = read_mono_in_chunks(&path, 16_000, 65_536).expect("multi-block read");
        let one_shot = read_mono_whole_buffer(&path, 16_000).expect("oracle read");

        assert!(
            streamed.len() >= one_shot.len(),
            "the bounded read came back shorter than the one-shot it replaced \
             ({} < {}): real audio was dropped, not just padding",
            streamed.len(),
            one_shot.len()
        );
        // The overshoot is the zero-pad of the final block and nothing more.
        let overshoot = streamed.len() - one_shot.len();
        let bound = (65_536.0_f64 * (16_000.0 / SAMPLE_RATE as f64)) as usize + 64;
        assert!(
            overshoot <= bound,
            "the bounded read ran {overshoot} samples past the one-shot, more than one \
             output block ({bound}) — the tail rule has changed"
        );

        let (differing, max_abs, _max_rel, snr) = diff_report(&one_shot, &streamed);
        assert!(
            differing > 0,
            "44100->16000 over five blocks came out bit-identical to the one-shot. The ratio is \
             not representable in f64, so that would mean the filter stopped re-anchoring — \
             re-check the documented behaviour before believing it."
        );
        assert!(
            max_abs < 1e-4,
            "worst absolute difference {max_abs:.3e} is no longer a last-bit effect"
        );
        assert!(
            snr > 90.0,
            "SNR {snr:.1} dB is no longer far below the noise floor"
        );
        println!(
            "[multi-block] {frames} frames 44100->16000: one_shot={} streamed={} differing={differing} \
             max_abs={max_abs:.3e} snr={snr:.1} dB",
            one_shot.len(),
            streamed.len(),
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The harness has to be innocent before any of the above means anything: the
    /// chunked path driven with one chunk the length of the input is the same call
    /// the one-shot makes, so it must return the same bytes.
    #[test]
    fn a_single_chunk_resample_is_the_one_shot() {
        for from in [44_100u32, 48_000, 22_050, 16_000] {
            let wf: Vec<f32> = (0..(from as usize / 2))
                .map(|i| (i as f64 * 0.05).sin() as f32 * 0.5)
                .collect();
            for to in [16_000u32, 44_100] {
                let one_shot = resample_mono(&wf, from, to).expect("one-shot");
                let chunked = resample_chunked(&wf, from, to, wf.len()).expect("single chunk");
                let (differing, max_abs, _, _) = diff_report(&one_shot, &chunked);
                assert_eq!(
                    differing, 0,
                    "{from}->{to}: the same-shaped call through two entry points differs by \
                     up to {max_abs:.3e} — the comparison tests describe the harness, not rubato"
                );
            }
        }
    }

    /// 48 kHz → 16 kHz divides exactly, so the filter's phase step is exact and
    /// chunking cannot move a single bit. This is the half of the story that makes
    /// the other half a documented limit rather than a suspicion.
    #[test]
    fn chunked_matches_one_shot_on_an_exact_ratio() {
        let wf: Vec<f32> = (0..300_000)
            .map(|i| (i as f64 * 0.05).sin() as f32 * 0.5)
            .collect();
        let one_shot = resample_mono(&wf, 48_000, 16_000).expect("one-shot");
        for chunk in [4_096usize, 16_384, 65_536] {
            let chunked = resample_chunked(&wf, 48_000, 16_000, chunk).expect("chunked");
            assert!(
                chunked.len() >= one_shot.len(),
                "{chunk}: the chunked row is shorter than the one-shot"
            );
            let (differing, max_abs, _, _) = diff_report(&one_shot, &chunked);
            assert_eq!(
                differing, 0,
                "48000->16000 at chunk {chunk}: {differing} samples differ (worst {max_abs:.3e}). \
                 The ratio is exactly representable, so chunking must be bit-neutral here."
            );
        }
    }

    /// 44.1 kHz → 16 kHz is the inexact case, and it must differ in the last bits
    /// while staying a non-event in signal terms. Asserting both halves is the
    /// point: a future "optimisation" that silently changes the phase arithmetic
    /// will move these numbers, and so will a future rubato that anchors its
    /// read position differently.
    #[test]
    fn chunked_differs_in_lsb_s_on_an_inexact_ratio() {
        let wf: Vec<f32> = (0..300_000)
            .map(|i| (i as f64 * 0.05).sin() as f32 * 0.5)
            .collect();
        let one_shot = resample_mono(&wf, 44_100, 16_000).expect("one-shot");
        for chunk in [4_096usize, 16_384, 65_536] {
            let chunked = resample_chunked(&wf, 44_100, 16_000, chunk).expect("chunked");
            let common = one_shot.len().min(chunked.len());
            let first_diff = (0..common).find(|&i| one_shot[i].to_bits() != chunked[i].to_bits());
            let (differing, max_abs, max_rel, snr) = diff_report(&one_shot, &chunked);
            assert!(
                first_diff.is_some(),
                "44100->16000 at chunk {chunk}: no differing sample in the first {common}. \
                 Re-verify the documented inexactness claim before trusting its absence."
            );
            assert!(
                max_abs < 1e-4,
                "44100->16000 at chunk {chunk}: worst absolute error {max_abs:.3e} is no longer \
                 a last-bit difference (max relative {max_rel:.3e})"
            );
            assert!(
                snr > 90.0,
                "44100->16000 at chunk {chunk}: SNR {snr:.1} dB — the difference is audible scale, \
                 not LSB scale"
            );
            println!(
                "[inexact] 44100->16000 chunk {chunk}: one_shot={} chunked={} differing={differing}/{common} \
                 ({:.1}%) first_diff={:?} max_abs={max_abs:.3e} snr={snr:.1} dB",
                one_shot.len(),
                chunked.len(),
                100.0 * differing as f64 / common as f64,
                first_diff,
            );
        }
    }

    /// The negative result behind [`StreamingSinc::push`]'s tail rule, measured
    /// instead of asserted: draining a `SincFixedIn` that has already seen its
    /// short final chunk invents output the file never contained.
    #[test]
    fn draining_past_the_short_final_chunk_invents_samples() {
        let frames = 200_000usize; // 44.1 kHz, not a whole number of blocks
        let chunk = 32_768usize;
        let wf: Vec<f32> = (0..frames)
            .map(|i| (i as f64 * 0.05).sin() as f32 * 0.5)
            .collect();
        let ideal = (frames as f64 * 16_000.0 / SAMPLE_RATE as f64).ceil() as usize;

        let pushed = resample_chunked(&wf, SAMPLE_RATE, 16_000, chunk).expect("push-only");

        // The wrong shape: keep calling with `None` input, the way you would to
        // "flush" a streaming filter. Sixteen calls is what the first version of
        // this code did before it gave up and took what it had.
        let mut r = rubato::SincFixedIn::<f32>::new(
            16_000.0 / SAMPLE_RATE as f64,
            2.0,
            sinc_params(),
            chunk,
            1,
        )
        .expect("filter");
        let mut scratch = r.output_buffer_allocate(true);
        let mut drained: Vec<f32> = Vec::new();
        for part in wf.chunks(chunk) {
            let ins: [&[f32]; 1] = [part];
            let (_, n) = r
                .process_partial_into_buffer(Some(&ins), &mut [&mut scratch[0]], None)
                .expect("push");
            drained.extend_from_slice(&scratch[0][..n]);
        }
        let baseline = drained.len();
        let mut phantom_frames = 0usize;
        for call in 1..=16 {
            let (fi, fo) = r
                .process_partial_into_buffer(
                    Option::<&[&[f32]]>::None,
                    &mut [&mut scratch[0]],
                    None,
                )
                .expect("drain");
            drained.extend_from_slice(&scratch[0][..fo]);
            phantom_frames += fo;
            assert!(
                fi > 0 || fo > 0,
                "the drain terminated on call {call}, so `SincFixedIn` apparently grew a flush \
                 behaviour and the tail rule needs re-deriving"
            );
        }

        assert!(
            drained.len() > pushed.len(),
            "the drain loop added nothing ({} vs {}), so `SincFixedIn` apparently grew a flush \
             behaviour and the tail rule needs re-deriving",
            baseline,
            pushed.len()
        );
        println!(
            "[tail] {frames} frames at {chunk}: ideal={ideal} push-only={} drained={} \
             phantom_calls=16 invented={} ({phantom_frames} of them from the 16 phantom blocks)",
            pushed.len(),
            drained.len(),
            drained.len() - pushed.len(),
        );
        // Both rows overshoot `ideal` (the zero-pad of the final block), but only
        // the drained one overshoots by more than a block's worth of output: that
        // excess is silence, and a consumer that reads to the end of the row gets
        // it as audio.
        let one_block_out = (chunk as f64 * (16_000.0 / SAMPLE_RATE as f64)) as usize + 64;
        assert!(
            pushed.len() <= ideal + one_block_out,
            "push-only ran {} samples past the true end, more than one output block",
            pushed.len() - ideal
        );
        assert!(
            drained.len() > pushed.len() + one_block_out,
            "the drain invented {} samples, which is no longer obviously more than the padding \
             the honest shape already produces",
            drained.len() - pushed.len()
        );
    }

    /// What a bounded read returns for a file *shorter than one chunk*, which a
    /// long fixture cannot speak to: on a ten-minute track the overshoot is ≤
    /// `chunk·ratio` samples out of tens of millions, on a 0.1 s clip the same
    /// overshoot is most of the row. So measure it instead of assuming it scales.
    #[test]
    fn streaming_resample_of_a_short_file_reports_its_own_tail() {
        // Must stay in step with READ_CHUNK_FRAMES, the block the shipped reader
        // uses.
        const CHUNK: usize = READ_CHUNK_FRAMES;
        for (from, secs) in [
            (44_100u32, 0.1f64),
            (44_100, 1.0),
            (44_100, 1.4),
            (48_000, 3.0),
            (44_100, 30.0),
        ] {
            let n = (from as f64 * secs) as usize;
            let input: Vec<f32> = (0..n)
                .map(|i| (i as f64 * 0.05).sin() as f32 * 0.5)
                .collect();
            let one_shot = resample_mono(&input, from, 16_000).unwrap();
            let streamed = resample_chunked(&input, from, 16_000, CHUNK).unwrap();
            let ideal = (n as f64 * 16_000.0 / from as f64).ceil() as usize;
            let tail = streamed.get(ideal.min(streamed.len())..).unwrap_or(&[]);
            let peak = tail.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            let nonzero = tail.iter().filter(|s| **s != 0.0).count();
            // The sinc is FIR, so past the file's own duration the row carries a
            // couple of dozen samples of ring-out and then exact zeros. The
            // overshoot therefore costs length, not signal.
            assert!(
                nonzero <= 64,
                "{from}->{secs}s: {nonzero} samples past the file's own duration are non-zero, \
                 so `ideal + 64` is no longer where the row's signal ends"
            );
            assert!(
                tail.len() <= CHUNK + 64,
                "{from}->{secs}s: the bounded read ran {} samples past the true end, more than \
                 one chunk of output — the overshoot is supposed to be bounded by the zero-pad of \
                 the final chunk alone",
                tail.len()
            );
            println!(
                "[TAIL] {from}->16000 in={n} ({secs} s): one_shot={} streamed={} ideal={ideal} \
                 past_ideal={} peak={peak:.3e} nonzero_past_ideal={nonzero}",
                one_shot.len(),
                streamed.len(),
                streamed.len().saturating_sub(ideal),
            );
            assert!(
                streamed.len() >= one_shot.len(),
                "{from}->{secs}s: the bounded read came back shorter than the one-shot it \
                 replaced ({}) — real audio was dropped, not just padding",
                one_shot.len()
            );
        }
    }

    /// Files shorter than one block never see the tail: they are resampled as a
    /// single block, which is the one-shot's own shape.
    #[test]
    fn a_sub_block_file_is_resampled_as_one_block_and_matches_exactly() {
        let dir = test_dir("subblock");
        let path = dir.join("clip.wav");
        write_wav(&path, 48_000, 2, 16, false, 9_600); // 0.2 s, one block
        let streamed = read_mono(&path, 16_000).expect("read");
        let one_shot = resample_mono(
            &read_mono_whole_buffer(&path, 48_000).expect("oracle fold"),
            48_000,
            16_000,
        )
        .expect("oracle resample");
        assert_eq!(streamed.len(), one_shot.len(), "row length");
        for (i, (g, w)) in streamed.iter().zip(one_shot.iter()).enumerate() {
            assert_eq!(
                g, w,
                "sample {i}: a single-block read must equal the one-shot"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `stream_mono` propagates a sink error instead of decoding the rest of the
    /// track first — a caller that stopped at the first block must not pay for a
    /// four-hour file.
    #[test]
    fn a_sink_that_stops_ends_the_read() {
        let dir = test_dir("sink");
        let path = dir.join("long.wav");
        write_wav(&path, SAMPLE_RATE, 2, 16, false, 200_000);
        let mut seen = 0usize;
        let err = stream_mono(&path, 1000, |row| {
            seen += row.len();
            Err(Error::Cancelled)
        });
        assert!(matches!(err, Err(Error::Cancelled)), "{err:?}");
        assert_eq!(seen, 1000, "the read should stop at the block that failed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
