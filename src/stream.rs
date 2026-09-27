//! Reading the mix one window at a time, and writing the stems one checkpoint at
//! a time.
//!
//! This is the pair of halves an engine runs between. [`WavSource`] hands out one
//! inference window per call, so the cost of *input* is independent of track
//! length; [`StemWriter`] appends finalized frames to a `.part` pair and rewrites
//! each file's 44-byte WAV header at every window boundary, so the cost of
//! *output* is a buffer's worth of frames and the files on disk are playable at
//! every instant in between. [`chunk_starts`] is the grid they share: a resume
//! only means something if a restarted run's windows land where the dead run's
//! landed, and that is the whole content of [`resume_plan`].
//!
//! ## Why the staging file is a real WAV and not a headerless blob
//!
//! The obvious design for an intermediate file is raw PCM: no header to
//! maintain, and the frame count is just `len() - 0`. This module does the more
//! expensive thing on purpose. Because the header is rewritten in place at every
//! window boundary, at any instant on disk there is a complete, self-describing
//! pair of prefixes: you can open the `.part` files in any player, listen to how
//! far the run got, and feed them to `hound` on the next attempt without a
//! sidecar to tell you the format. A blob would be unreadable until the run
//! finished, which is exactly the moment a run that died at hour one does not
//! reach.
//!
//! The header is written *after* the payload it counts, so it can lag the file by
//! less than one window but can never outrun it. [`StemWriter::open_or_resume`]
//! still takes the smaller of the two figures: a process that died mid-write left
//! a torn frame, and losing the bytes nobody counted is not worth losing the
//! track.
//!
//! ## What has to match before a resume is allowed to continue
//!
//! A `.part` pair is the output of one specific run. Continuing it from a
//! different run splices two separations into one stem, which is worse than the
//! restart it costs, because it is silent. So [`StemJob`] records the run's
//! identity — input bytes and mtime, model bytes and mtime, the window geometry,
//! the chunk count, the track length — and [`StemWriter::open_or_resume`] checks
//! the fields one at a time. Every reason to start over (no sidecar, an
//! unparseable one, a mismatch, a pair whose headers disagree with its size) is
//! logged and turned into a fresh start; **none of them is an error**. A
//! checkpoint you cannot trust must never fail the step.
//!
//! Only *cancellation* discards staging ([`discard_staging`]), because only
//! cancellation means "nobody wants this result". Every other failure — on a
//! 16 GB laptop usually an allocation refusal — leaves the pair and its sidecar
//! where they are, which is the entire point of checkpointing: the retry picks up
//! at the last window that finished rather than at second zero.

use std::fs::File;
use std::io::{self, BufReader, Seek, Write};
use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader, WavSpec};
use ndarray::{Array2, ArrayViewMut2};
use serde::{Deserialize, Serialize};

use crate::audio::SAMPLE_RATE;
use crate::error::{Error, Result};

/// Source frames decoded per internal step. A requested range is walked in spans
/// of about this many frames so the scratch buffer between the decoder and the
/// interpolation stays bounded no matter how wide a window is asked for.
const SPAN_FRAMES: usize = 1 << 17;

/// What the samples in a track actually are, decided from the header once so the
/// decode loop does not branch per sample.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Depth {
    I16,
    I24,
    I32,
    F32,
}

impl Depth {
    fn of(spec: &WavSpec) -> Option<Depth> {
        match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Int, 16) => Some(Depth::I16),
            (SampleFormat::Int, 24) => Some(Depth::I24),
            (SampleFormat::Int, 32) => Some(Depth::I32),
            (SampleFormat::Float, 32) => Some(Depth::F32),
            _ => None,
        }
    }
}

fn wav_err(path: &Path, detail: impl Into<String>) -> Error {
    Error::Wav {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

/// A stereo, [`SAMPLE_RATE`]-normalised, f32 view of a WAV file on disk, decoded
/// one window at a time.
///
/// Opening reads the header only, so it costs a few kilobytes whatever the file's
/// length. [`WavSource::read`] then decodes exactly the frames a window needs,
/// which means the input side of a separation costs `window` frames and never the
/// track: an hour of 44.1 kHz stereo as `f32` is ~1.27 GB, and a separation loop
/// that held it would be paying that to touch each sample once.
///
/// The decode arithmetic is copied expression-for-expression from the
/// whole-buffer loader this replaced, which is kept in the tests as an oracle
/// (`chunk_reads_match_full_decode` asserts the two agree sample for sample).
/// Copying it verbatim matters: the linear resample index
/// `i * (orig - 1) / (target - 1)` is evaluated in that order, and on long tracks
/// the product leaves the range where `f64` is exact. "Improving" it here would
/// shift samples relative to anything ever compared against this loader.
///
/// The rate normalisation is linear interpolation, matching that loader. It is
/// deliberately *not* the windowed sinc of [`crate::audio::resample_mono`]: the
/// sinc is what a mono analysis read wants, and running it here would make a
/// decoded window disagree with the reference decode that every resume check and
/// every stem comparison in this module is stated against.
pub struct WavSource {
    path: PathBuf,
    reader: WavReader<BufReader<File>>,
    channels: usize,
    orig_frames: usize,
    target_frames: usize,
    depth: Depth,
    /// False exactly when the loader this mirrors skipped its resample step,
    /// which is also the case where samples are copied through unmodified.
    resample: bool,
}

impl WavSource {
    /// Read only the header. Nothing is decoded here, so opening costs a few KB
    /// regardless of the file's length.
    pub fn open(path: &Path) -> Result<WavSource> {
        let reader = WavReader::open(path).map_err(|e| wav_err(path, format!("open: {e}")))?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(wav_err(path, "zero channels"));
        }
        let depth = Depth::of(&spec).ok_or_else(|| {
            wav_err(
                path,
                format!(
                    "unsupported sample format: {} bit {:?}",
                    spec.bits_per_sample, spec.sample_format
                ),
            )
        })?;
        // `len()` counts values across all channels; dividing gives frames, the
        // same `samples.len() / channels` the full-buffer loader derived.
        let orig_frames = reader.len() as usize / spec.channels as usize;

        // Mirrors the loader: a 0-frame track is only rejected when a resample is
        // actually needed (a 0-length file at the target rate stays legal).
        let resample = spec.sample_rate != SAMPLE_RATE;
        let target_frames = if resample {
            if orig_frames == 0 {
                return Err(wav_err(path, "audio has 0 samples to resample"));
            }
            let t = (orig_frames as f64 * SAMPLE_RATE as f64 / spec.sample_rate as f64).round()
                as usize;
            if t == 0 {
                return Err(wav_err(
                    path,
                    format!(
                        "audio too short after resampling: {} frames at {} Hz rounds to 0 at {} Hz",
                        orig_frames, spec.sample_rate, SAMPLE_RATE
                    ),
                ));
            }
            log::info!(
                "[stream] resampling: {} Hz -> {} Hz ({} -> {} frames)",
                spec.sample_rate,
                SAMPLE_RATE,
                orig_frames,
                t
            );
            t
        } else {
            orig_frames
        };

        Ok(WavSource {
            path: path.to_path_buf(),
            reader,
            channels: spec.channels as usize,
            orig_frames,
            target_frames,
            depth,
            resample,
        })
    }

    /// Total frames at the normalised rate — the `total_len` a chunk schedule is
    /// built from.
    pub fn frames(&self) -> usize {
        self.target_frames
    }

    /// Where the input came from, for logging and for the caller's own guards.
    pub fn source_rate(&self) -> u32 {
        self.reader.spec().sample_rate
    }

    /// Decode output frames `[out_start, out_start + dst.len_of(Axis(1)))` into
    /// `dst` (shape `[2, len]`).
    ///
    /// Random access: this seeks, so callers may ask for overlapping or
    /// out-of-order windows — the crossfade grid asks for both. Asking past the
    /// end is an error rather than a clamp, because a schedule bug must not
    /// silently shorten a stem.
    pub fn read(&mut self, out_start: usize, dst: &mut ArrayViewMut2<'_, f32>) -> Result<()> {
        let len = dst.len_of(ndarray::Axis(1));
        if len == 0 {
            return Ok(());
        }
        if out_start + len > self.target_frames {
            return Err(wav_err(
                &self.path,
                format!(
                    "read out of range: {len} frames at {out_start}, track has {}",
                    self.target_frames
                ),
            ));
        }

        if !self.resample {
            // No resample: the loader copied decoded frames straight through, so
            // do the same rather than running the interpolation formula (whose
            // index product is not exact for large frame counts).
            return self.decode_into(out_start, len, dst);
        }

        let orig = self.orig_frames;
        let target = self.target_frames;
        let mut done = 0usize;
        while done < len {
            let sub = (len - done).min(SPAN_FRAMES);
            let o0 = out_start + done;
            let lo = src_index(o0, orig, target);
            let hi = src_index(o0 + sub - 1, orig, target)
                .saturating_add(1)
                .min(orig - 1);
            let span = hi - lo + 1;

            let mut buf = Array2::<f32>::zeros((2, span));
            {
                let mut bview = buf.view_mut();
                self.decode_into(lo, span, &mut bview)?;
            }

            for k in 0..sub {
                let i = o0 + k;
                let pos = i as f64 * (orig.saturating_sub(1)) as f64
                    / (target.saturating_sub(1)).max(1) as f64;
                let idx = pos.floor() as usize;
                let frac = (pos - idx as f64) as f32;
                let s0 = idx.min(orig.saturating_sub(1)) - lo;
                let s1 = (idx + 1).min(orig.saturating_sub(1)) - lo;
                for ch in 0..2 {
                    let a = buf[[ch, s0]];
                    let b = buf[[ch, s1]];
                    dst[[ch, done + k]] = a + frac * (b - a);
                }
            }
            done += sub;
        }
        Ok(())
    }

    /// Decode `count` source frames starting at `first` straight into `dst`,
    /// folding channels to stereo the way the loader does.
    fn decode_into(
        &mut self,
        first: usize,
        count: usize,
        dst: &mut ArrayViewMut2<'_, f32>,
    ) -> Result<()> {
        debug_assert!(
            count <= dst.len_of(ndarray::Axis(1)),
            "destination too small for {count} frames"
        );
        debug_assert!(
            first <= u32::MAX as usize,
            "frame index beyond what hound can seek"
        );
        // `WavReader::seek` takes a *frame* index and multiplies by the channel
        // count itself; passing a sample index here would skip twice as far on a
        // stereo file.
        self.reader
            .seek(first as u32)
            .map_err(|e| Error::io_at(self.path.as_path(), e))?;
        let channels = self.channels;
        let path = self.path.clone();
        // A free function rather than a method: `samples()` holds a mutable borrow
        // of `self.reader` for the whole loop, so nothing else about `self` can be
        // borrowed at the same time.
        match self.depth {
            Depth::I16 => {
                let it = self
                    .reader
                    .samples::<i16>()
                    .map(|r| r.map(|s| s as f32 / 32768.0));
                decode_frames(&path, first, it, channels, count, dst)
            }
            Depth::I24 => {
                let it = self
                    .reader
                    .samples::<i32>()
                    .map(|r| r.map(|s| s as f32 / 8_388_608.0));
                decode_frames(&path, first, it, channels, count, dst)
            }
            Depth::I32 => {
                let it = self
                    .reader
                    .samples::<i32>()
                    .map(|r| r.map(|s| s as f32 / 2_147_483_648.0));
                decode_frames(&path, first, it, channels, count, dst)
            }
            Depth::F32 => decode_frames(
                &path,
                first,
                self.reader.samples::<f32>(),
                channels,
                count,
                dst,
            ),
        }
    }
}

/// Pull `count` frames of `channels` interleaved, already-scaled `f32` off `it`
/// and fold each one into `dst`.
fn decode_frames<I>(
    path: &Path,
    first: usize,
    mut it: I,
    channels: usize,
    count: usize,
    dst: &mut ArrayViewMut2<'_, f32>,
) -> Result<()>
where
    I: Iterator<Item = std::result::Result<f32, hound::Error>>,
{
    let mut frame = vec![0.0f32; channels];
    for i in 0..count {
        for slot in frame.iter_mut() {
            match it.next() {
                Some(Ok(s)) => *slot = s,
                Some(Err(e)) => {
                    return Err(wav_err(
                        path,
                        format!("sample stream at frame {}: {e}", first + i),
                    ))
                }
                // The header promises `channels * duration` values; running out
                // early is a truncated file.
                None => {
                    return Err(wav_err(
                        path,
                        format!(
                        "truncated: fewer samples than the header declares, ran out at frame {}",
                        first + i
                    ),
                    ))
                }
            }
        }
        fold_channels(&frame, dst, i);
    }
    Ok(())
}

/// Source frame index that output frame `i` interpolates from — the loader's
/// expression, kept in the same evaluation order.
fn src_index(i: usize, orig: usize, target: usize) -> usize {
    let pos = i as f64 * (orig.saturating_sub(1)) as f64 / (target.saturating_sub(1)).max(1) as f64;
    (pos.floor() as usize).min(orig.saturating_sub(1))
}

/// One frame of scaled channel values → the two rows of `dst` at column `col`.
fn fold_channels(frame: &[f32], dst: &mut ArrayViewMut2<'_, f32>, col: usize) {
    match frame.len() {
        1 => {
            let s = frame[0];
            dst[[0, col]] = s;
            dst[[1, col]] = s;
        }
        2 => {
            dst[[0, col]] = frame[0];
            dst[[1, col]] = frame[1];
        }
        _ => {
            // More than two channels: average the even ones into the left channel
            // and the odd ones into the right, in channel order. That is the rule
            // the whole-buffer loader this mirrors applied, so a pair written
            // against it stays frame-for-frame comparable with one written here.
            let mut l_sum = 0.0f32;
            let mut r_sum = 0.0f32;
            let mut l_count = 0u32;
            let mut r_count = 0u32;
            for (ch, &v) in frame.iter().enumerate() {
                if ch % 2 == 0 {
                    l_sum += v;
                    l_count += 1;
                } else {
                    r_sum += v;
                    r_count += 1;
                }
            }
            dst[[0, col]] = l_sum / l_count as f32;
            dst[[1, col]] = r_sum / r_count as f32;
        }
    }
}

/// Temp path used while streaming a stem; renamed to the final path only once the
/// full track is written.
///
/// Keeping the final names absent until then is what lets an embedding host's
/// "already done, skip it" check stay safe: a half-written `vocals.wav` never
/// exists to be mistaken for a finished one.
pub fn part_path(final_path: &Path) -> PathBuf {
    let name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio.wav".to_string());
    final_path.with_file_name(format!("{name}.part"))
}

/// The sidecar that records which run a `.part` pair belongs to.
///
/// Kept *outside* the pair so the pair stays a plain WAV file that plays at every
/// moment of the run (see [`StemWriter::checkpoint`]).
pub fn job_path(vocals_output: &Path) -> PathBuf {
    let mut name = part_path(vocals_output).into_os_string();
    name.push(".job");
    PathBuf::from(name)
}

/// Remove a run's whole checkpoint: both `.part` files and the sidecar that binds
/// them.
///
/// Engines call this when a run is *cancelled*. A run that merely failed keeps
/// them, which is the distinction the whole module exists to make.
pub fn discard_staging(vocals_output: &Path, background_output: &Path) {
    for path in [
        part_path(vocals_output),
        part_path(background_output),
        job_path(vocals_output),
    ] {
        let _ = std::fs::remove_file(&path);
    }
}

/// Header size the writer reserves, and the payload it describes: stereo 16-bit
/// little-endian integer at [`SAMPLE_RATE`], 4 bytes per frame.
pub const STEM_HEADER_BYTES: u64 = 44;
const BYTES_PER_FRAME: u64 = 4;

// ─────────────────── window grid, and where a resume picks up ───────────────────
//
// Every separation engine infers the mix one fixed-length window at a time, and a
// checkpoint only means something if a resumed run's windows line up with the
// dead run's. So the geometry lives here, next to the writer it serves, written
// down once for every engine: [`chunk_starts`] says where the windows go,
// [`resume_plan`] says which of them a half-written track still needs.

/// Chunk start schedule for a track of `total_len` frames: hop = `win - overlap`,
/// plus a final chunk ending at the last sample when the regular grid doesn't
/// cover the tail. Pure function so the boundary behaviour is unit-testable.
///
/// The grid requires `hop >= overlap`, i.e. `overlap <= win / 2`. Two things break
/// without it: the not-yet-finalized region an engine keeps between windows would
/// grow past one window, and a sample could then be touched by three or more
/// chunks, so the fade weights of a resumed run would not reproduce the ones an
/// uninterrupted run applied — which is precisely the equality a resume is bought
/// for. It is asserted rather than returned as an error because it is a property
/// of the caller's constants, checked in debug builds and in tests.
pub fn chunk_starts(total_len: usize, win: usize, overlap: usize) -> Vec<usize> {
    let hop = win
        .checked_sub(overlap)
        .expect("chunk_starts: overlap must not exceed the window");
    debug_assert!(
        hop >= overlap,
        "grid {win}/{overlap}: hop({hop}) < overlap({overlap}) — the pending buffer \
         would exceed the window and samples can be touched by 3+ chunks"
    );
    let mut starts: Vec<usize> = Vec::new();
    let mut s = 0usize;
    if total_len <= win {
        starts.push(0);
    } else {
        while s + win <= total_len {
            starts.push(s);
            s += hop;
        }
        let last = total_len - win;
        if *starts.last().unwrap() < last {
            starts.push(last);
        }
    }
    starts
}

/// Whether chunk `m`'s window still covers frames at or beyond `keep`.
pub fn reaches_past(m: usize, starts: &[usize], total_len: usize, win: usize, keep: usize) -> bool {
    starts[m] + (total_len - starts[m]).min(win) > keep
}

/// Where a resumed run picks up.
///
/// `first` is the lowest chunk whose flush output reaches past what is already on
/// disk. The chunks in `prime_from..first` have to be inferred *again* — not
/// flushed, only merged — because the hop is shorter than the window, so the
/// samples after the seam are a crossfade and the earlier contributions live only
/// in those chunks' fade-outs.
///
/// Usually that is exactly one chunk, but a chunk may start less than a full
/// window behind its predecessor: [`chunk_starts`] appends a final chunk at
/// `total_len - win`, and when that lands inside an earlier chunk's span the seam
/// is a crossfade of *two* earlier chunks. `prime_from` is therefore the earliest
/// chunk whose window still reaches past the seam, not a fixed `first - 1`; one
/// chunk short of that produces a background stem that wobbles by 1 LSB on
/// tail-appended tracks, which is audible as nothing and wrong all the same.
///
/// The cost is at most two windows of recompute and the benefit is that the
/// resumed bytes are the same bytes an uninterrupted run would have written;
/// resuming at `first` with an empty pending buffer instead would quietly replace
/// a crossfade with one chunk's raw output.
#[derive(Debug, PartialEq, Eq)]
pub struct ResumePlan {
    /// The lowest chunk that still has to be flushed.
    pub first: usize,
    /// Chunks below this are skipped outright; `prime_from == first` means the
    /// run re-infers nothing (a fresh start, or a finished track).
    pub prime_from: usize,
    /// Frames of the pair to keep: `frames` snapped down to `starts[first - 1]`'s
    /// flush boundary, which is a chunk start.
    pub keep_frames: usize,
}

/// The schedule a resumed run needs, given how many frames the pair on disk
/// already holds.
pub fn resume_plan(starts: &[usize], total_len: usize, win: usize, frames: usize) -> ResumePlan {
    let n = starts.len();
    if frames >= total_len {
        // Everything is written; the run only has to publish.
        return ResumePlan {
            first: n,
            prime_from: n,
            keep_frames: total_len,
        };
    }
    // Chunk `i` finalizes `[starts[i], starts[i+1])`, so every flush boundary but
    // the last is also a chunk start. Snapping to one is what lets the pending
    // buffer be rebuilt exactly rather than approximately.
    let j = starts.partition_point(|&s| s <= frames).saturating_sub(1);
    let keep = starts[j];
    // Walk back over every chunk that still contributes to the pending region.
    // The starts are increasing, so once one falls short the rest do too.
    let mut prime_from = j;
    while prime_from > 0 && reaches_past(prime_from - 1, starts, total_len, win, keep) {
        prime_from -= 1;
    }
    ResumePlan {
        first: j,
        prime_from,
        keep_frames: keep,
    }
}

/// The spec every stem, and every `.part` file, is written against.
pub fn spec16() -> WavSpec {
    WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

/// Sample value → 16-bit PCM, with the loader's clamp and truncation.
fn to_i16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0) as i16
}

/// The canonical PCM header for [`spec16`] with `data_bytes` of payload.
///
/// Hand-rolled rather than borrowed from `hound` because the length has to be
/// rewritten *while* the file is still being appended to, and because a resume
/// needs the header to land at exactly the offset the appender expects.
/// `stem_header_is_hounds_header` pins it against `hound`'s own output, byte for
/// byte.
///
/// The caller caps `data_bytes` so `data_bytes + 36` still fits a `u32`; the RIFF
/// size field is the whole file minus its own 8 bytes.
fn wav_header(data_bytes: u32) -> [u8; STEM_HEADER_BYTES as usize] {
    let mut h = [0u8; STEM_HEADER_BYTES as usize];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(data_bytes + 36).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
    h[22..24].copy_from_slice(&2u16.to_le_bytes()); // channels
    h[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    h[28..32].copy_from_slice(&(SAMPLE_RATE * 4).to_le_bytes()); // bytes/sec
    h[32..34].copy_from_slice(&4u16.to_le_bytes()); // block align
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits/sample
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    h
}

/// Bump when the checkpoint on disk stops meaning what this build thinks it
/// means — a changed window grid, a different crossfade, a new frame layout.
///
/// The sidecar's [`StemJob::format`] field is compared first, so a stale pair from
/// an older build of *this crate* is recognised as one.
pub const STEM_JOB_FORMAT: u32 = 1;

/// What a resumable run is resumable *as*: the pair on disk may only be continued
/// by a run that would have produced the same bytes from the same input. Every
/// field is one way two runs differ in practice, and a silent splice of two
/// different separations into one stem is worse than the restart this record
/// forces.
///
/// The on-disk form is [`StemJob::encode`], a flat `key=value` file: a resume
/// record a user can paste into a bug report, and one whose failure names the
/// field that disagrees instead of naming a hash. The `serde` derives are for
/// callers that want the record inside a JSON report of their own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StemJob {
    pub format: u32,
    /// Frames of the normalised mix — the length the schedule was built from.
    pub total_frames: u64,
    pub n_chunks: u64,
    /// The engine's window geometry, so a changed grid invalidates the pair.
    pub window: u64,
    pub overlap: u64,
    pub input_bytes: u64,
    pub input_mtime_ns: u64,
    pub model_bytes: u64,
    pub model_mtime_ns: u64,
}

impl StemJob {
    /// Fingerprint a run from the things that define it. Both paths are the
    /// caller's: this crate does not know where a model lives, only that a
    /// different file there means a different separation.
    pub fn for_run(
        total_frames: usize,
        starts: &[usize],
        window: usize,
        overlap: usize,
        input: &Path,
        model: &Path,
    ) -> StemJob {
        let (input_bytes, input_mtime_ns) = file_stamp(input);
        let (model_bytes, model_mtime_ns) = file_stamp(model);
        StemJob {
            format: STEM_JOB_FORMAT,
            total_frames: total_frames as u64,
            n_chunks: starts.len() as u64,
            window: window as u64,
            overlap: overlap as u64,
            input_bytes,
            input_mtime_ns,
            model_bytes,
            model_mtime_ns,
        }
    }

    /// The sidecar text: one `key=value` per field, in declaration order.
    pub fn encode(&self) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("# rust-roformer stem checkpoint\n");
        s.push_str(&format!("format={}\n", self.format));
        s.push_str(&format!("total_frames={}\n", self.total_frames));
        s.push_str(&format!("n_chunks={}\n", self.n_chunks));
        s.push_str(&format!("window={}\n", self.window));
        s.push_str(&format!("overlap={}\n", self.overlap));
        s.push_str(&format!("input_bytes={}\n", self.input_bytes));
        s.push_str(&format!("input_mtime_ns={}\n", self.input_mtime_ns));
        s.push_str(&format!("model_bytes={}\n", self.model_bytes));
        s.push_str(&format!("model_mtime_ns={}\n", self.model_mtime_ns));
        s
    }

    /// Parse a sidecar. The error is a *reason to start over*, not a failure: a
    /// record this build cannot read is indistinguishable from no record at all,
    /// and the run must still be able to finish.
    pub fn decode(text: &str) -> std::result::Result<StemJob, String> {
        let mut seen: u32 = 0;
        let mut job = StemJob {
            format: 0,
            total_frames: 0,
            n_chunks: 0,
            window: 0,
            overlap: 0,
            input_bytes: 0,
            input_mtime_ns: 0,
            model_bytes: 0,
            model_mtime_ns: 0,
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("sidecar line is not `key=value`: {line}"))?;
            let slot = match key.trim() {
                "format" => {
                    job.format = num(key, value)?;
                    1 << 0
                }
                "total_frames" => {
                    job.total_frames = num(key, value)?;
                    1 << 1
                }
                "n_chunks" => {
                    job.n_chunks = num(key, value)?;
                    1 << 2
                }
                "window" => {
                    job.window = num(key, value)?;
                    1 << 3
                }
                "overlap" => {
                    job.overlap = num(key, value)?;
                    1 << 4
                }
                "input_bytes" => {
                    job.input_bytes = num(key, value)?;
                    1 << 5
                }
                "input_mtime_ns" => {
                    job.input_mtime_ns = num(key, value)?;
                    1 << 6
                }
                "model_bytes" => {
                    job.model_bytes = num(key, value)?;
                    1 << 7
                }
                "model_mtime_ns" => {
                    job.model_mtime_ns = num(key, value)?;
                    1 << 8
                }
                // Unknown keys are ignored: the `format` field is what says
                // whether the record means what this build thinks it means.
                other => {
                    log::debug!("[stream] ignoring unknown sidecar field `{other}`");
                    continue;
                }
            };
            if seen & slot != 0 {
                return Err(format!("sidecar repeats `{key}`"));
            }
            seen |= slot;
        }
        const ALL: u32 = (1 << 9) - 1;
        if seen != ALL {
            return Err(format!(
                "sidecar is missing {} field(s)",
                ALL.count_ones() - seen.count_ones()
            ));
        }
        Ok(job)
    }

    /// How this run differs from the one that wrote the sidecar, for the log line
    /// that explains a restart.
    fn diff(&self, other: &StemJob) -> String {
        let mut parts = Vec::new();
        if self.format != other.format {
            parts.push(format!("format {} vs {}", self.format, other.format));
        }
        if self.total_frames != other.total_frames {
            parts.push(format!(
                "frames {} vs {}",
                self.total_frames, other.total_frames
            ));
        }
        if self.n_chunks != other.n_chunks {
            parts.push(format!("chunks {} vs {}", self.n_chunks, other.n_chunks));
        }
        if (self.window, self.overlap) != (other.window, other.overlap) {
            parts.push(format!(
                "grid {}/{} vs {}/{}",
                self.window, self.overlap, other.window, other.overlap
            ));
        }
        if (self.input_bytes, self.input_mtime_ns) != (other.input_bytes, other.input_mtime_ns) {
            parts.push(format!(
                "input {}B@{}ns vs {}B@{}ns",
                self.input_bytes, self.input_mtime_ns, other.input_bytes, other.input_mtime_ns
            ));
        }
        if (self.model_bytes, self.model_mtime_ns) != (other.model_bytes, other.model_mtime_ns) {
            parts.push("model changed".to_string());
        }
        if parts.is_empty() {
            "identical".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Parse one sidecar number, naming the field it came from when it will not
/// parse. A restart that has to explain itself is the reason the record is a
/// flat text file rather than an opaque blob.
fn num<T>(key: &str, value: &str) -> std::result::Result<T, String>
where
    T: TryFrom<u128>,
{
    let raw: u128 = value.trim().parse().map_err(|e| format!("{key}: {e}"))?;
    T::try_from(raw).map_err(|_| format!("{key}: {raw} does not fit in this field"))
}

/// Size and modification time of a file, as the two numbers a run's identity is
/// built from.
///
/// `(0, 0)` for a file that is not there: the fingerprint fields treat a missing
/// file the same way they treat an empty one, so a model that was deleted between
/// runs produces a mismatch (start over) rather than a panic.
///
/// A filesystem whose modification time is coarser than nanoseconds reports what
/// it can; identity then rests on size, which is why both are recorded.
pub fn file_stamp(path: &Path) -> (u64, u64) {
    match std::fs::metadata(path) {
        Ok(md) => (
            md.len(),
            md.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        ),
        Err(_) => (0, 0),
    }
}

/// Writes both stems to `.part` files and publishes them only when the whole
/// track is on disk, so a cancelled or failed run can never leave a half-written
/// `vocals.wav` for a skip-if-exists check to find.
///
/// The pair is also a *checkpoint*: [`checkpoint`](Self::checkpoint) rewrites each
/// header in place at a chunk boundary, so at all times the `.part` files are
/// complete, playable WAVs containing exactly the samples the engine has
/// finalized — which is what lets [`open_or_resume`](Self::open_or_resume) append
/// to them instead of throwing an hour of inference away.
pub struct StemWriter {
    tmp_v: PathBuf,
    tmp_b: PathBuf,
    job: PathBuf,
    vocals: PathBuf,
    background: PathBuf,
    // `Option` only so `finish` can hand the file handles back before renaming,
    // which Windows requires; every other path treats them as infallible.
    vw: Option<io::BufWriter<File>>,
    bw: Option<io::BufWriter<File>>,
    frames: usize,
    published: bool,
}

impl StemWriter {
    /// Start from nothing. A stale `.part` pair is truncated here and its sidecar
    /// removed, so an engine that does not support resuming behaves exactly as it
    /// did before checkpointing existed: a retry redoes the whole step.
    pub fn create(vocals_output: &Path, background_output: &Path) -> Result<StemWriter> {
        Self::start(vocals_output, background_output, None, 0)
    }

    /// Continue the run `job` describes, or start over if the pair on disk is not
    /// that run's. [`StemWriter::frames`] then reports how much was kept, which is
    /// 0 exactly when nothing was resumed.
    ///
    /// Every reason to start over goes to the log rather than to the caller: a
    /// checkpoint we cannot trust must never fail the step, it only costs the work
    /// already done.
    pub fn open_or_resume(
        vocals_output: &Path,
        background_output: &Path,
        job: &StemJob,
    ) -> Result<StemWriter> {
        match Self::resume_frames(vocals_output, background_output, job) {
            Ok(frames) => {
                if frames > 0 {
                    log::info!(
                        "[stream] resuming from the checkpoint: {} frames already finalized",
                        frames
                    );
                }
                Self::start(vocals_output, background_output, Some(job), frames)
            }
            Err(why) => {
                log::info!("[stream] starting over, not resuming: {why}");
                Self::start(vocals_output, background_output, Some(job), 0)
            }
        }
    }

    /// Frames both `.part` files agree they hold, or why they cannot be used.
    fn resume_frames(
        vocals_output: &Path,
        background_output: &Path,
        job: &StemJob,
    ) -> std::result::Result<usize, String> {
        let (tmp_v, tmp_b) = (part_path(vocals_output), part_path(background_output));
        let sidecar = job_path(vocals_output);
        let text = std::fs::read_to_string(&sidecar)
            .map_err(|e| format!("no checkpoint sidecar ({}): {e}", sidecar.display()))?;
        let saved =
            StemJob::decode(&text).map_err(|e| format!("unreadable checkpoint sidecar: {e}"))?;
        if &saved != job {
            return Err(format!(
                "checkpoint is for another run: {}",
                saved.diff(job)
            ));
        }
        let mut frames = usize::MAX;
        for path in [&tmp_v, &tmp_b] {
            let on_disk = std::fs::metadata(path)
                .map_err(|e| format!("{} missing: {e}", path.display()))?
                .len();
            if on_disk < STEM_HEADER_BYTES {
                return Err(format!("{} is shorter than its header", path.display()));
            }
            let r = WavReader::open(path)
                .map_err(|e| format!("{} has no readable header: {e}", path.display()))?;
            if r.spec() != spec16() {
                return Err(format!(
                    "{} is not the stem format: {:?}",
                    path.display(),
                    r.spec()
                ));
            }
            // The header is written after the data it counts (see
            // `checkpoint`), so it can lag the file but never outrun it. Take the
            // smaller of the two anyway: a torn write at the tail is not worth
            // losing the whole track over.
            let declared = r.len() as u64 / spec16().channels as u64;
            let actual = (on_disk - STEM_HEADER_BYTES) / BYTES_PER_FRAME;
            frames = frames.min(declared.min(actual) as usize);
        }
        // Whatever the pair claims, the track is the length the schedule was built
        // for.
        let frames = frames.min(job.total_frames as usize);
        if frames == 0 {
            return Err("checkpoint holds no finalized frames".to_string());
        }
        Ok(frames)
    }

    fn start(
        vocals_output: &Path,
        background_output: &Path,
        job: Option<&StemJob>,
        frames: usize,
    ) -> Result<StemWriter> {
        if let Some(parent) = vocals_output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io_at(parent, e))?;
            }
        }
        let tmp_v = part_path(vocals_output);
        let tmp_b = part_path(background_output);
        let data_bytes = frames as u64 * BYTES_PER_FRAME;
        let open = |path: &Path| -> Result<io::BufWriter<File>> {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                // Not truncate: an existing checkpoint keeps its samples and the
                // set_len below snaps the file to the length it declares, which
                // also drops anything a crashed run wrote past the count.
                .truncate(false)
                .open(path)
                .map_err(|e| Error::io_at(path, e))?;
            // Appending resumes at the declared length; anything the previous run
            // wrote past it (an interrupted chunk, a stale longer track) is dropped
            // rather than kept as a tail nobody counted.
            file.set_len(STEM_HEADER_BYTES + data_bytes)
                .map_err(|e| Error::io_at(path, e))?;
            let mut w = io::BufWriter::new(file);
            if frames == 0 {
                w.write_all(&wav_header(0))
                    .map_err(|e| Error::io_at(path, e))?;
            }
            w.flush().map_err(|e| Error::io_at(path, e))?;
            // A file opens at offset 0 and `write_frame` appends blindly, so the
            // append point has to be placed by hand — otherwise a resumed pair gets
            // its own header overwritten from byte 0.
            w.seek(io::SeekFrom::Start(STEM_HEADER_BYTES + data_bytes))
                .map_err(|e| Error::io_at(path, e))?;
            Ok(w)
        };
        let writer = StemWriter {
            job: job_path(vocals_output),
            tmp_v,
            tmp_b,
            vocals: vocals_output.to_path_buf(),
            background: background_output.to_path_buf(),
            vw: Some(open(&part_path(vocals_output))?),
            bw: Some(open(&part_path(background_output))?),
            frames,
            published: false,
        };
        match job {
            Some(job) => std::fs::write(&writer.job, job.encode())
                .map_err(|e| Error::io_at(writer.job.as_path(), e))?,
            None => {
                // This engine does not resume: `create` truncated the pair, so a
                // sidecar left by an older resumable run must not survive to vouch
                // for bytes it has nothing to do with. That is the one way two
                // different separations could be spliced into one stem.
                let _ = std::fs::remove_file(&writer.job);
            }
        }
        Ok(writer)
    }

    /// Append one finalized frame: the vocals pair, then the background pair.
    pub fn write_frame(&mut self, vocals: [f32; 2], background: [f32; 2]) -> Result<()> {
        append_frame(&mut self.vw, &self.tmp_v, vocals)?;
        append_frame(&mut self.bw, &self.tmp_b, background)?;
        self.frames += 1;
        Ok(())
    }

    /// Frames written so far; engines assert this equals the track length before
    /// publishing.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Roll the pair back to an earlier checkpoint. Only used to snap a resumed
    /// frame count down to a chunk boundary, so the tail an interrupted run had
    /// appended past its last checkpoint is dropped rather than spliced onto a
    /// different crossfade.
    pub fn rewind_to(&mut self, frames: usize) -> Result<()> {
        if frames > self.frames {
            return Err(Error::Wav {
                path: self.tmp_v.clone(),
                detail: format!("cannot rewind forward: {frames} > {}", self.frames),
            });
        }
        self.frames = frames;
        self.checkpoint()
    }

    /// Make the `.part` pair self-describing: flush the payload, then rewrite each
    /// header with the frame count reached so far, and return to the end of the
    /// file to keep appending.
    ///
    /// Call this only at a point where every sample up to `frames` is final. For
    /// both engines that is a chunk boundary, which is what makes a resume land on
    /// a boundary too.
    pub fn checkpoint(&mut self) -> Result<()> {
        let payload = self.frames as u64 * BYTES_PER_FRAME;
        // Both size fields have to survive: `data` holds the payload, RIFF holds
        // the payload plus the 36 bytes that precede it.
        if payload > u64::from(u32::MAX) - 36 {
            return Err(Error::Wav {
                path: self.tmp_v.clone(),
                detail: format!("stem of {payload} bytes exceeds the WAV size fields"),
            });
        }
        let bytes = payload as u32;
        let header = wav_header(bytes);
        for (slot, path) in [
            (&mut self.vw, self.tmp_v.as_path()),
            (&mut self.bw, self.tmp_b.as_path()),
        ] {
            let w = writer_of(slot, path)?;
            w.flush().map_err(|e| Error::io_at(path, e))?;
            w.seek(io::SeekFrom::Start(0))
                .map_err(|e| Error::io_at(path, e))?;
            w.write_all(&header).map_err(|e| Error::io_at(path, e))?;
            w.seek(io::SeekFrom::Start(STEM_HEADER_BYTES + payload))
                .map_err(|e| Error::io_at(path, e))?;
        }
        Ok(())
    }

    /// Flush, then rename into place. Background goes first so `vocals.wav` — the
    /// file a caller gates on — only appears once both stems are complete. If the
    /// second rename fails, the first is undone: a track with one stem published
    /// and one not is worse than a track with none.
    pub fn finish(mut self) -> Result<PathBuf> {
        self.checkpoint()?;
        self.published = true;
        // Close both handles before renaming: a rename over an open handle fails
        // on Windows, and the buffer has already been flushed by `checkpoint`.
        let _ = self.vw.take();
        let _ = self.bw.take();
        std::fs::rename(&self.tmp_b, &self.background)
            .map_err(|e| Error::io_at(self.tmp_b.as_path(), e))?;
        if let Err(e) = std::fs::rename(&self.tmp_v, &self.vocals) {
            let _ = std::fs::remove_file(&self.background);
            let _ = std::fs::remove_file(&self.tmp_v);
            let _ = std::fs::remove_file(&self.job);
            return Err(Error::io_at(self.tmp_v.as_path(), e));
        }
        let _ = std::fs::remove_file(&self.job);
        Ok(self.vocals.clone())
    }

    /// Drop a run's checkpoint: the `.part` pair and its sidecar. This is the
    /// cancellation path; a run that failed keeps them.
    pub fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.tmp_v);
        let _ = std::fs::remove_file(&self.tmp_b);
        let _ = std::fs::remove_file(&self.job);
    }
}

/// Make the pair self-describing on the way out.
///
/// [`checkpoint`](StemWriter::checkpoint) is a method an engine calls at a chunk
/// boundary; this is the same guarantee for the paths that do not reach one — an
/// early `?`, a panic in the loop, a `cleanup`-less error return. Without it, the
/// pair still *resumes* correctly (the header lags the payload and the resume takes
/// the smaller figure), but the files are not playable up to where the work
/// actually stands, which is the other half of why staging is a real WAV.
///
/// Errors are dropped: this is a best-effort tidy-up inside a path that is already
/// failing, and there is no caller left to report to. A pair that could not be
/// repaired is still caught by the resume checks.
impl Drop for StemWriter {
    fn drop(&mut self) {
        if !self.published {
            let _ = self.checkpoint();
        }
    }
}

/// Unwrap an optional writer handle with the path in the error.
fn writer_of<'a>(
    slot: &'a mut Option<io::BufWriter<File>>,
    path: &Path,
) -> Result<&'a mut io::BufWriter<File>> {
    slot.as_mut()
        .ok_or_else(|| Error::io_at(path, io::Error::other("stem writer is already closed")))
}

/// Interleave one frame into one stem.
fn append_frame(slot: &mut Option<io::BufWriter<File>>, path: &Path, pair: [f32; 2]) -> Result<()> {
    let w = writer_of(slot, path)?;
    let mut buf = [0u8; 4];
    for (k, &v) in pair.iter().enumerate() {
        buf[k * 2..k * 2 + 2].copy_from_slice(&to_i16(v).to_le_bytes());
    }
    w.write_all(&buf).map_err(|e| Error::io_at(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Scratch fixtures go under the temp dir in a uniquely-named subdirectory per
    /// test, removed again at the end of it.
    fn test_dir(name: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rust_roformer_stream_{name}_{}_{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    fn write_fixture(path: &Path, channels: u16, rate: u32, frames: usize) {
        let spec = WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).expect("create fixture");
        for i in 0..frames {
            for ch in 0..channels {
                // Several shapes per frame so interpolation is exercised at every
                // phase, including the extremes of the range.
                let t = i as f32 / (frames.max(1) as f32);
                let v = ((t * 40.0 + ch as f32) * std::f32::consts::PI).sin() * 30000.0;
                w.write_sample(v as i16).expect("write fixture sample");
            }
        }
        w.finalize().expect("finalize fixture");
    }

    /// (path, case index) covering both resample directions, the no-resample case,
    /// mono duplication, >2-channel folding, and tracks long enough that a single
    /// window crosses `SPAN_FRAMES` (which a 4 s window at 44.1 kHz always does).
    ///
    /// Written and removed inside `chunk_reads_match_full_decode` — one owner, so
    /// no test can read a file another is still writing. That hazard is real and
    /// pinned separately by `open_before_finalize_reads_zero_samples`: `hound` only
    /// patches the RIFF data length in `finalize()`, so a WAV opened mid-write
    /// reads back as a 0-frame track.
    const CASES: [(u16, u32, usize); 8] = [
        (2, 44100, 5000),    // already at the target rate
        (2, 48000, 5000),    // the common case: video audio at 48 kHz
        (1, 48000, 3000),    // mono, duplicated and resampled
        (6, 44100, 3000),    // channel folding without resampling
        (2, 16000, 2000),    // upsampling
        (2, 48000, 9),       // shorter than one interpolation step
        (2, 48000, 400_000), // resampling across several decode spans
        (2, 44100, 400_000), // the same, without resampling
    ];

    fn write_corpus(dir: &Path) -> Vec<(PathBuf, usize)> {
        CASES
            .iter()
            .enumerate()
            .map(|(i, &(c, r, n))| {
                let p = dir.join(format!("case{i}.wav"));
                write_fixture(&p, c, r, n);
                (p, i)
            })
            .collect()
    }

    /// A WAV whose header still carries the pre-`finalize()` data length opens as
    /// an empty track. Worth pinning because it is the one way a fixture can lie:
    /// a reader that reports 0 frames looks like a silent file, not like a bug.
    #[test]
    fn open_before_finalize_reads_zero_samples() {
        let dir = test_dir("unfinalized");
        let path = dir.join("unfinalized.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        {
            let mut w = hound::WavWriter::create(&path, spec).expect("create");
            // Enough samples to push hound's own buffer out, so the header the
            // reader below sees really is the on-disk one.
            for _ in 0..100_000 {
                w.write_sample(1000i16).expect("write");
            }
            let err = match WavSource::open(&path) {
                Ok(_) => panic!("an unfinalized WAV opened as a usable track"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("0 samples"),
                "expected the empty-track error, got: {err}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The full-buffer loader this replaces, kept as the oracle: read every sample,
    /// fold to stereo, then linear-resample the whole thing. 16-bit integer input
    /// only, which is what the loader itself accepted.
    fn decode_whole_file(path: &Path) -> Result<Array2<f32>> {
        let reader = WavReader::open(path).map_err(|e| wav_err(path, format!("open: {e}")))?;
        let spec = reader.spec();
        let n_channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;
        if n_channels == 0 {
            return Err(wav_err(path, "zero channels"));
        }
        let samples: Vec<f32> = reader
            .into_samples::<i16>()
            .map(|r| r.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| wav_err(path, format!("sample stream: {e}")))?;

        let mut audio = match n_channels {
            1 => {
                let n = samples.len();
                let mut arr = Array2::<f32>::zeros((2, n));
                for (i, &s) in samples.iter().enumerate() {
                    arr[[0, i]] = s;
                    arr[[1, i]] = s;
                }
                arr
            }
            2 => {
                let n = samples.len() / 2;
                let mut arr = Array2::<f32>::zeros((2, n));
                for i in 0..n {
                    arr[[0, i]] = samples[i * 2];
                    arr[[1, i]] = samples[i * 2 + 1];
                }
                arr
            }
            c => {
                let n = samples.len() / c;
                let mut arr = Array2::<f32>::zeros((2, n));
                for i in 0..n {
                    let mut l_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    let mut l_count = 0u32;
                    let mut r_count = 0u32;
                    for ch in 0..c {
                        let s = samples[i * c + ch];
                        if ch % 2 == 0 {
                            l_sum += s;
                            l_count += 1;
                        } else {
                            r_sum += s;
                            r_count += 1;
                        }
                    }
                    arr[[0, i]] = l_sum / l_count as f32;
                    arr[[1, i]] = r_sum / r_count as f32;
                }
                arr
            }
        };

        if sample_rate != SAMPLE_RATE {
            let original_len = audio.ncols();
            if original_len == 0 {
                return Err(wav_err(path, "audio has 0 samples"));
            }
            let target_len =
                (original_len as f64 * SAMPLE_RATE as f64 / sample_rate as f64).round() as usize;
            if target_len == 0 {
                return Err(wav_err(path, "audio too short after resampling"));
            }
            let mut resampled = Array2::<f32>::zeros((2, target_len));
            for ch in 0..2 {
                for i in 0..target_len {
                    let src_pos = i as f64 * (original_len.saturating_sub(1)) as f64
                        / (target_len.saturating_sub(1)).max(1) as f64;
                    let src_idx = src_pos.floor() as usize;
                    let frac = (src_pos - src_idx as f64) as f32;
                    let s0 = audio[[ch, src_idx.min(original_len.saturating_sub(1))]];
                    let s1 = audio[[ch, (src_idx + 1).min(original_len.saturating_sub(1))]];
                    resampled[[ch, i]] = s0 + frac * (s1 - s0);
                }
            }
            audio = resampled;
        }
        Ok(audio)
    }

    /// The point of the module: windowed reads must return exactly what the
    /// full-buffer loader returned, sample for sample.
    #[test]
    fn chunk_reads_match_full_decode() {
        let dir = test_dir("corpus");
        for (path, i) in write_corpus(&dir) {
            let (ch, rate, n) = CASES[i];
            let tag = format!("{ch}ch {rate}Hz {n}f");
            let full = decode_whole_file(&path).unwrap_or_else(|e| panic!("{tag}: oracle: {e}"));
            let mut src = WavSource::open(&path).unwrap_or_else(|e| panic!("{tag}: open: {e}"));
            assert_eq!(src.frames(), full.ncols(), "{tag}: frame count disagrees");
            let total = src.frames();

            // Probes: a normal engine window at many offsets — overlapping and
            // forward-jumping, the shape a crossfade grid asks for — plus windows
            // wider than `SPAN_FRAMES`, anchored at every point where the span loop
            // changes over, because that is where a chunked decode can go wrong.
            let short = 1024usize;
            let mut probes: Vec<(usize, usize)> = Vec::new();
            for s in (0..total).step_by((short * 7 / 5).max(1)).take(40) {
                probes.push((s, short.min(total - s)));
            }
            probes.push((total.saturating_sub(short), short.min(total)));
            for len in [SPAN_FRAMES + 4099, 2 * SPAN_FRAMES + 1] {
                if len > total {
                    continue;
                }
                for s in [
                    0usize,
                    1,
                    SPAN_FRAMES - 1,
                    SPAN_FRAMES,
                    SPAN_FRAMES + 1,
                    2 * SPAN_FRAMES,
                    total - len,
                ] {
                    if s + len <= total {
                        probes.push((s, len));
                    }
                }
            }

            for (start, len) in probes {
                let mut buf = Array2::<f32>::zeros((2, len));
                {
                    let mut view = buf.view_mut();
                    src.read(start, &mut view)
                        .unwrap_or_else(|e| panic!("{tag}: read at {start} len {len}: {e}"));
                }
                let expect = full.slice(ndarray::s![.., start..start + len]);
                assert_eq!(
                    buf,
                    expect.to_owned(),
                    "{tag}: window at {start} len {len} differs"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Out-of-range requests are an error, not a silent clamp: a schedule bug must
    /// not be able to shorten a stem by asking past the end and getting zeros.
    #[test]
    fn read_past_end_is_rejected() {
        let dir = test_dir("range");
        let path = dir.join("track.wav");
        write_fixture(&path, 2, 44_100, 5_000);
        let mut src = WavSource::open(&path).unwrap();
        let mut buf = Array2::<f32>::zeros((2, 10));
        let total = src.frames();
        assert_eq!(total, 5_000);
        {
            let mut view = buf.view_mut();
            assert!(src.read(total, &mut view).is_err());
            assert!(src.read(total - 5, &mut view).is_err());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Widening over the loader this mirrors, which accepted 16-bit only: a DAW or
    /// a video extractor hands over 24-bit or 32-bit input as often as not, and the
    /// fold rule is the same one the 16-bit path uses. The expectation here is
    /// computed from what was written — the quantised integer the file stores,
    /// divided by that depth's own full scale in the same order the reader does it —
    /// so the comparison is exact rather than to a tolerance.
    #[test]
    fn other_sample_widths_fold_the_same_way() {
        let dir = test_dir("widths");
        // (bits, float, stored-value divisor)
        for (bits, float, divisor) in [
            (24u16, false, 8_388_608.0f32),
            (32, false, 2_147_483_648.0),
            (32, true, 1.0),
        ] {
            let name = if float { "f32" } else { "i" };
            let path = dir.join(format!("case{bits}{name}.wav"));
            let spec = WavSpec {
                channels: 2,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: bits,
                sample_format: if float {
                    SampleFormat::Float
                } else {
                    SampleFormat::Int
                },
            };
            let frames = 2_000usize;
            // The largest magnitude a sample of this depth can carry, i.e. what the
            // fixture scales its signal by on the way in.
            let full_scale: f32 = match bits {
                16 => 32767.0,
                24 => 8_388_607.0,
                _ => 2_147_483_647.0,
            };
            let mut want: Vec<[f32; 2]> = Vec::with_capacity(frames);
            {
                let mut w = hound::WavWriter::create(&path, spec).expect("create");
                for i in 0..frames {
                    let l = (i as f32 * 0.11).sin() * 0.6;
                    let r = (i as f32 * 0.29).cos() * 0.4;
                    if float {
                        w.write_sample(l).unwrap();
                        w.write_sample(r).unwrap();
                        want.push([l, r]);
                    } else {
                        let (sl, sr) = ((l * full_scale) as i32, (r * full_scale) as i32);
                        w.write_sample(sl).unwrap();
                        w.write_sample(sr).unwrap();
                        want.push([sl as f32 / divisor, sr as f32 / divisor]);
                    }
                }
                w.finalize().unwrap();
            }
            let mut src = WavSource::open(&path)
                .unwrap_or_else(|e| panic!("{bits} bit float={float}: open: {e}"));
            assert_eq!(src.frames(), frames);
            assert_eq!(src.source_rate(), SAMPLE_RATE);
            let mut buf = Array2::<f32>::zeros((2, frames));
            {
                let mut view = buf.view_mut();
                src.read(0, &mut view).expect("read");
            }
            for i in 0..frames {
                for ch in 0..2 {
                    assert_eq!(
                        buf[[ch, i]].to_bits(),
                        want[i][ch].to_bits(),
                        "{bits} bit float={float} frame {i} channel {ch}"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A job record for the checkpoint tests. The geometry fields are arbitrary on
    /// purpose — nothing here runs inference; only their *equality* matters.
    fn job_for(total_frames: u64) -> StemJob {
        StemJob {
            format: STEM_JOB_FORMAT,
            total_frames,
            n_chunks: total_frames / 4 + 1,
            window: 4,
            overlap: 1,
            input_bytes: 7,
            input_mtime_ns: 11,
            model_bytes: 13,
            model_mtime_ns: 17,
        }
    }

    /// Append `frames` ramp frames and leave the pair checkpointed, the way an
    /// interrupted run leaves it. Any earlier checkpoint at these paths is dropped
    /// first, so "40 frames" always means 40 and not "40 more".
    fn write_checkpointed(v: &Path, b: &Path, job: &StemJob, frames: usize) {
        discard_staging(v, b);
        let mut w = StemWriter::open_or_resume(v, b, job).unwrap();
        for i in 0..frames {
            w.write_frame([i as f32 / 1000.0, -(i as f32) / 1000.0], [0.5, -0.25])
                .unwrap();
        }
        w.checkpoint().unwrap();
        assert_eq!(w.frames(), frames);
        drop(w);
    }

    /// The hand-rolled header must be *hound's* header byte for byte: the pair is
    /// written by this module and read back by `hound` on every resume, so any
    /// disagreement is a checkpoint nobody can trust.
    #[test]
    fn stem_header_is_hounds_header() {
        let dir = test_dir("header");
        for payload in [0u32, 4, 4096, 44_100 * 4] {
            let path = dir.join(format!("hdr_{payload}.wav"));
            {
                let mut w = hound::WavWriter::create(&path, spec16()).unwrap();
                // hound counts samples, not frames: two per frame here.
                for i in 0..payload / 2 {
                    w.write_sample(i as i16).unwrap();
                }
                w.finalize().unwrap();
            }
            let on_disk = std::fs::read(&path).unwrap();
            assert_eq!(
                &on_disk[..STEM_HEADER_BYTES as usize],
                &wav_header(payload)[..],
                "payload of {payload} bytes: header differs from hound's"
            );
            let _ = std::fs::remove_file(&path);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stem_writer_publishes_both_files_and_cleans_up() {
        let dir = test_dir("publish");
        let v = dir.join("vocals.wav");
        let b = dir.join("background.wav");
        {
            let mut w = StemWriter::create(&v, &b).unwrap();
            for i in 0..50 {
                let x = (i as f32) / 100.0;
                w.write_frame([x, -x], [0.5, -0.5]).unwrap();
            }
            assert_eq!(w.frames(), 50);
            assert!(!v.exists(), "vocals must stay absent until finish()");
            assert!(part_path(&v).exists());
            w.finish().unwrap();
        }
        assert!(v.exists() && b.exists());
        assert!(!part_path(&v).exists() && !part_path(&b).exists());

        // A run that is abandoned removes its staging.
        let v2 = dir.join("v2.wav");
        let b2 = dir.join("b2.wav");
        let w = StemWriter::create(&v2, &b2).unwrap();
        assert!(part_path(&v2).exists());
        w.cleanup();
        assert!(!part_path(&v2).exists() && !part_path(&b2).exists());

        // Round-trip through the windowed reader: the frames landed in the
        // interleaved stereo layout (a planar write would read back as a two-speed
        // mess), and the two stems are distinct files.
        let mut w = StemWriter::create(&v, &b).unwrap();
        for i in 0..64 {
            let s = (i as f32) / 64.0;
            w.write_frame([s, -s], [0.25, -0.25]).unwrap();
        }
        w.finish().unwrap();
        let back = WavSource::open(&v).unwrap();
        assert_eq!(back.frames(), 64);
        let back = decode_whole_file(&v).unwrap();
        assert!((back[[0, 32]] - 0.5).abs() < 1e-3, "L: {}", back[[0, 32]]);
        assert!((back[[1, 32]] + 0.5).abs() < 1e-3, "R: {}", back[[1, 32]]);
        let bg = decode_whole_file(&b).unwrap();
        assert!((bg[[0, 32]] - 0.25).abs() < 1e-3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `.part` pair is a playable WAV in the middle of a run, not only after
    /// `finish` — that is what lets a killed run keep its work, and it is why the
    /// staging file is not a headerless blob.
    #[test]
    fn checkpoint_leaves_a_readable_pair() {
        let dir = test_dir("readable");
        let (v, b) = (dir.join("cp_v.wav"), dir.join("cp_b.wav"));
        let job = job_for(100);
        write_checkpointed(&v, &b, &job, 40);

        for (path, ch) in [(part_path(&v), 0usize), (part_path(&b), 1)] {
            let mut r = hound::WavReader::open(&path).unwrap();
            assert_eq!(r.spec(), spec16(), "{}: spec", path.display());
            assert_eq!(r.len(), 80, "{}: declared frames", path.display());
            let samples: Vec<i16> = r.samples::<i16>().map(|s| s.unwrap()).collect();
            assert_eq!(samples.len(), 80);
            if ch == 0 {
                // Same rounding the writer applies, computed independently.
                for i in 0..40 {
                    let want = to_i16(i as f32 / 1000.0);
                    assert_eq!(samples[i * 2], want, "vocals L at {i}");
                    assert_eq!(samples[i * 2 + 1], -want, "vocals R at {i}");
                }
            } else {
                let (l, r) = (to_i16(0.5), to_i16(-0.25));
                for i in 0..40 {
                    assert_eq!(samples[i * 2], l, "background L at {i}");
                    assert_eq!(samples[i * 2 + 1], r, "background R at {i}");
                }
            }
        }
        StemWriter::open_or_resume(&v, &b, &job).unwrap().cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A writer that is dropped without an explicit checkpoint — an engine that
    /// bails out between windows — still leaves the pair self-describing.
    #[test]
    fn dropping_the_writer_leaves_the_pair_readable() {
        let dir = test_dir("drop");
        let (v, b) = (dir.join("drop_v.wav"), dir.join("drop_b.wav"));
        let job = job_for(100);
        {
            let mut w = StemWriter::open_or_resume(&v, &b, &job).unwrap();
            for i in 0..30 {
                w.write_frame([i as f32 / 1000.0, 0.0], [0.0, 0.0]).unwrap();
            }
            // No checkpoint() call, and no `cleanup`: this is the early return.
        }
        for path in [part_path(&v), part_path(&b)] {
            let r = hound::WavReader::open(&path).unwrap();
            assert_eq!(r.len(), 60, "{}: header after a bare drop", path.display());
        }
        // The resume path agrees with the header.
        assert_eq!(StemWriter::resume_frames(&v, &b, &job).unwrap(), 30);
        StemWriter::open_or_resume(&v, &b, &job).unwrap().cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every field of the job record is one way two runs differ, and a splice of two
    /// separations into one stem is exactly what must never ship.
    #[test]
    fn checkpoint_resumes_only_the_same_run() {
        let dir = test_dir("identity");
        let (v, b) = (dir.join("id_v.wav"), dir.join("id_b.wav"));
        let job = job_for(100);
        write_checkpointed(&v, &b, &job, 40);
        assert_eq!(StemWriter::resume_frames(&v, &b, &job).unwrap(), 40);

        // Each case names one identity field and perturbs it; a mismatch must
        // mean "start over", never "continue from a stale prefix".
        type JobMutation = (&'static str, fn(&mut StemJob));
        let mutations: Vec<JobMutation> = vec![
            ("format", |j| j.format += 1),
            ("total_frames", |j| j.total_frames += 1),
            ("n_chunks", |j| j.n_chunks += 1),
            ("window", |j| j.window += 1),
            ("overlap", |j| j.overlap += 1),
            ("input_bytes", |j| j.input_bytes += 1),
            ("input_mtime_ns", |j| j.input_mtime_ns += 1),
            ("model_bytes", |j| j.model_bytes += 1),
            ("model_mtime_ns", |j| j.model_mtime_ns += 1),
        ];
        for (field, f) in mutations {
            let mut other = job.clone();
            f(&mut other);
            let err = StemWriter::resume_frames(&v, &b, &other)
                .expect_err("{field}: a different run must not resume");
            assert!(err.contains("another run"), "{field}: {err}");
            // The reason names the field that disagreed, so a restart can be
            // explained rather than guessed at.
            assert!(!err.contains("identical"), "{field}: {err}");
        }

        // A sidecar that cannot be read is a reason to start over, never an error
        // that fails the step.
        std::fs::remove_file(job_path(&v)).unwrap();
        let err = StemWriter::resume_frames(&v, &b, &job).unwrap_err();
        assert!(err.contains("sidecar"), "{err}");
        std::fs::write(job_path(&v), b"not=a=sidecar\nand=no=fields").unwrap();
        let err = StemWriter::resume_frames(&v, &b, &job).unwrap_err();
        assert!(err.contains("unreadable"), "{err}");
        StemWriter::open_or_resume(&v, &b, &job).unwrap().cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A checkpoint we cannot trust must never fail the step: `open_or_resume` with
    /// a different run's record on disk starts over, silently and successfully, and
    /// the fresh pair holds none of the stale prefix.
    #[test]
    fn a_mismatched_checkpoint_starts_over_instead_of_failing() {
        let dir = test_dir("mismatch");
        let (v, b) = (dir.join("mm_v.wav"), dir.join("mm_b.wav"));
        let job = job_for(100);
        write_checkpointed(&v, &b, &job, 40);

        let mut other = job.clone();
        other.input_bytes += 1; // the input was re-exported since the dead run
        let mut w = StemWriter::open_or_resume(&v, &b, &other).expect("must not fail the step");
        assert_eq!(w.frames(), 0, "a stale prefix must not be continued");
        // The pair is truncated to nothing but a header, and the sidecar now
        // describes *this* run.
        assert_eq!(
            std::fs::metadata(part_path(&v)).unwrap().len(),
            STEM_HEADER_BYTES
        );
        assert_eq!(
            StemJob::decode(&std::fs::read_to_string(job_path(&v)).unwrap()).unwrap(),
            other
        );
        w.checkpoint().unwrap();
        drop(w);
        assert!(StemWriter::resume_frames(&v, &b, &job).is_err());
        StemWriter::open_or_resume(&v, &b, &other)
            .unwrap()
            .cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The sidecar is flat text on purpose: it must survive a round trip and name
    /// its own fields when it does not.
    #[test]
    fn sidecar_round_trips_and_is_readable_text() {
        let job = job_for(123_456);
        let text = job.encode();
        assert!(text.contains("total_frames=123456"), "{text}");
        assert!(text.contains("overlap="), "{text}");
        assert_eq!(StemJob::decode(&text).unwrap(), job);

        // A missing field and a non-numeric field are both reasons to start over,
        // described as such.
        let truncated: String = text
            .lines()
            .filter(|l| !l.starts_with("model_bytes"))
            .collect::<Vec<_>>()
            .join("\n");
        let err = StemJob::decode(&truncated).unwrap_err();
        assert!(err.contains("missing"), "{err}");
        let err = StemJob::decode(&text.replace("window=4", "window=four")).unwrap_err();
        assert!(err.contains("window"), "{err}");
        // Repeating a field is not "the last one wins".
        let err = StemJob::decode(&format!("{text}window=9\n")).unwrap_err();
        assert!(err.contains("repeats"), "{err}");
        // An unknown field is tolerated; `format` is what guards meaning.
        assert_eq!(
            StemJob::decode(&format!("{}whatever=1\n", text)).unwrap(),
            job
        );
    }

    /// A process dies at an arbitrary byte, not at a frame: the kept prefix is the
    /// smaller of what the headers claim and what is actually on disk, for whichever
    /// of the two stems is further behind.
    #[test]
    fn checkpoint_accounts_for_a_torn_or_ragged_tail() {
        let dir = test_dir("torn");
        let (v, b) = (dir.join("torn_v.wav"), dir.join("torn_b.wav"));
        let job = job_for(100);
        let (pv, pb) = (part_path(&v), part_path(&b));

        write_checkpointed(&v, &b, &job, 40);
        // Bytes written past the last header patch: the header is behind, so it
        // wins and the uncounted tail is dropped.
        let mut extra = std::fs::OpenOptions::new().append(true).open(&pv).unwrap();
        extra.write_all(&[1u8; 8]).unwrap();
        drop(extra);
        assert_eq!(StemWriter::resume_frames(&v, &b, &job).unwrap(), 40);

        // A torn frame: the byte count floors to the last whole frame.
        write_checkpointed(&v, &b, &job, 40);
        std::fs::File::options()
            .write(true)
            .open(&pv)
            .unwrap()
            .set_len(STEM_HEADER_BYTES + 37 * 4 + 2)
            .unwrap();
        assert_eq!(StemWriter::resume_frames(&v, &b, &job).unwrap(), 37);

        // Two files, one track: the lagging stem sets the boundary.
        write_checkpointed(&v, &b, &job, 40);
        std::fs::File::options()
            .write(true)
            .open(&pb)
            .unwrap()
            .set_len(STEM_HEADER_BYTES + 20 * 4)
            .unwrap();
        assert_eq!(StemWriter::resume_frames(&v, &b, &job).unwrap(), 20);

        // Never more than the track has, whatever the pair claims.
        let small = job_for(50);
        write_checkpointed(&v, &b, &small, 60);
        assert_eq!(StemWriter::resume_frames(&v, &b, &small).unwrap(), 50);

        // Nothing usable is not an error: it is a fresh start, reported.
        let (v2, b2) = (dir.join("empty_v.wav"), dir.join("empty_b.wav"));
        let mut w = StemWriter::open_or_resume(&v2, &b2, &job).unwrap();
        w.checkpoint().unwrap();
        drop(w);
        assert!(StemWriter::resume_frames(&v2, &b2, &job)
            .unwrap_err()
            .contains("no finalized frames"));
        StemWriter::open_or_resume(&v2, &b2, &job)
            .unwrap()
            .cleanup();
        StemWriter::open_or_resume(&v, &b, &job).unwrap().cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The resume path's promise: appending to a checkpointed pair yields the bytes
    /// a single uninterrupted write would have produced, and `rewind_to` (which is
    /// how the engine snaps the pair down to a window boundary) keeps that true.
    #[test]
    fn resumed_append_equals_a_single_run() {
        let dir = test_dir("resumed");
        let job = job_for(60);

        let one_shot = |v: PathBuf, b: PathBuf| {
            let mut w = StemWriter::create(&v, &b).unwrap();
            for i in 0..60 {
                w.write_frame([i as f32 / 1000.0, -(i as f32) / 1000.0], [0.5, -0.25])
                    .unwrap();
            }
            w.finish().unwrap();
        };
        let (v1, b1) = (dir.join("once_v.wav"), dir.join("once_b.wav"));
        one_shot(v1.clone(), b1.clone());

        let (v2, b2) = (dir.join("resumed_v.wav"), dir.join("resumed_b.wav"));
        write_checkpointed(&v2, &b2, &job, 45);
        let mut w = StemWriter::open_or_resume(&v2, &b2, &job).unwrap();
        assert_eq!(w.frames(), 45);
        // Snap down to a "window boundary" (40 here), dropping the 5 frames the
        // interrupted run had written past its last checkpoint.
        w.rewind_to(40).unwrap();
        assert_eq!(w.frames(), 40);
        assert!(w.rewind_to(41).is_err(), "rewinding forward would splice");
        for i in 40..60 {
            w.write_frame([i as f32 / 1000.0, -(i as f32) / 1000.0], [0.5, -0.25])
                .unwrap();
        }
        w.finish().unwrap();

        for (a, b) in [(&v1, &v2), (&b1, &b2)] {
            assert_eq!(
                std::fs::read(a).unwrap(),
                std::fs::read(b).unwrap(),
                "{} != resumed {}",
                a.display(),
                b.display()
            );
        }
        // Publishing removes the checkpoint, including its sidecar.
        assert!(!job_path(&v2).exists() && !part_path(&v2).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A non-resumable engine (`create`) must not leave an old sidecar standing over
    /// the pair it just truncated.
    #[test]
    fn create_drops_a_stale_sidecar() {
        let dir = test_dir("stale");
        let (v, b) = (dir.join("stale_v.wav"), dir.join("stale_b.wav"));
        let job = job_for(100);
        write_checkpointed(&v, &b, &job, 40);
        assert!(job_path(&v).exists());

        let mut w = StemWriter::create(&v, &b).unwrap();
        assert!(
            !job_path(&v).exists(),
            "sidecar outlived the pair it described"
        );
        assert_eq!(
            std::fs::metadata(part_path(&v)).unwrap().len(),
            STEM_HEADER_BYTES
        );
        assert_eq!(w.frames(), 0);
        w.write_frame([0.1, 0.2], [0.3, 0.4]).unwrap();
        w.cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Both header size fields are 32-bit, so a payload past 4 GiB has to be an
    /// error rather than a header that wraps and describes a shorter track.
    #[test]
    fn checkpoint_refuses_what_the_headers_cannot_hold() {
        let dir = test_dir("huge");
        let (v, b) = (dir.join("huge_v.wav"), dir.join("huge_b.wav"));
        let mut w = StemWriter::create(&v, &b).unwrap();
        // No data is written; only the counter is pushed past what fits.
        w.frames = (u32::MAX as usize / 4) + 1;
        let err = w
            .checkpoint()
            .expect_err("a >4 GiB stem must not silently wrap");
        assert!(
            err.to_string().contains("exceeds the WAV size fields"),
            "{err}"
        );
        assert_eq!(
            std::fs::metadata(part_path(&v)).unwrap().len(),
            STEM_HEADER_BYTES,
            "the refused checkpoint still changed the file"
        );
        // `published` is still false, so the Drop path tries the same refused
        // checkpoint and must not corrupt the header it left alone.
        drop(w);
        assert_eq!(
            std::fs::metadata(part_path(&v)).unwrap().len(),
            STEM_HEADER_BYTES
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Only cancellation discards staging. A failure that is not cancellation leaves
    /// the pair, the sidecar and their bytes exactly where they were.
    #[test]
    fn only_cancellation_throws_the_checkpoint_away() {
        let dir = test_dir("cancel");
        let (v, b) = (dir.join("cancel_v.wav"), dir.join("cancel_b.wav"));
        let job = job_for(100);
        write_checkpointed(&v, &b, &job, 40);
        let before = std::fs::metadata(part_path(&v)).unwrap().len();

        discard_staging(&v, &b);
        assert_eq!(
            (
                part_path(&v).exists(),
                part_path(&b).exists(),
                job_path(&v).exists()
            ),
            (false, false, false),
            "cancel must leave nothing for the next run to trip over"
        );

        // The same pair written again, now abandoned by a failure: nothing removes
        // it, so the numbers a later resume sees are the numbers that were written.
        write_checkpointed(&v, &b, &job, 40);
        assert_eq!(std::fs::metadata(part_path(&v)).unwrap().len(), before);
        assert!(job_path(&v).exists());
        assert_eq!(StemWriter::resume_frames(&v, &b, &job).unwrap(), 40);
        StemWriter::open_or_resume(&v, &b, &job).unwrap().cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_stamp_treats_a_missing_file_as_zeroes() {
        let dir = test_dir("stamp");
        assert_eq!(file_stamp(&dir.join("never_written.wav")), (0, 0));
        let path = dir.join("here.wav");
        std::fs::write(&path, b"0123456789").unwrap();
        let (bytes, mtime) = file_stamp(&path);
        assert_eq!(bytes, 10);
        assert!(mtime > 0, "a freshly written file should report a mtime");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The grid's own contract: hop covers the overlap, the schedule starts at 0,
    /// has no holes, and ends flush with the track.
    #[test]
    fn grid_covers_the_track_without_holes() {
        // The two bindings this crate's engines use: a 4 s window with 1.5 s of
        // overlap, and an 8 s window with 2.5 s.
        for &(win, overlap) in &[(176_400usize, 66_150usize), (352_800, 110_250)] {
            assert!(
                win - overlap >= overlap,
                "win={win}: hop must cover the overlap"
            );
            for total in [
                0usize,
                1,
                win / 2,
                win,
                win + 1,
                win * 2,
                win * 2 + win / 3,
                44_100 * 3600,
            ] {
                let starts = chunk_starts(total, win, overlap);
                assert!(
                    !starts.is_empty(),
                    "win={win} total={total}: empty schedule"
                );
                assert_eq!(starts[0], 0);
                for w in starts.windows(2) {
                    assert!(w[1] > w[0], "win={win} total={total}: {starts:?}");
                    assert!(
                        w[1] - w[0] <= win,
                        "win={win} total={total}: hole between {} and {}",
                        w[0],
                        w[1]
                    );
                }
                // Every frame is covered by some window, and the last window ends
                // flush with the track.
                let last = *starts.last().unwrap();
                assert!(
                    last + (total - last).min(win) >= total,
                    "win={win} total={total}: tail not covered"
                );
                // For a track small enough to check frame by frame, do exactly
                // that: coverage is the one property the whole grid exists for.
                if total <= 3 * win && total <= 600_000 {
                    for f in 0..total {
                        assert!(
                            starts
                                .iter()
                                .any(|&s| s <= f && f < s + (total - s).min(win)),
                            "win={win} total={total}: frame {f} is in no window"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tail_chunk_lands_exactly_at_total_minus_window() {
        for &(win, overlap) in &[(176_400usize, 66_150usize), (352_800, 110_250)] {
            let total = 44_100 * 3600; // an hour
            let starts = chunk_starts(total, win, overlap);
            assert_eq!(*starts.last().unwrap(), total - win, "win={win}");
        }
    }

    /// The grid invariant is a `debug_assert`, and an engine that asked for a window
    /// of 4 s with 3 s of overlap would corrupt a resumed stem rather than fail. In
    /// release builds the assert is gone, so the requirement is stated here as a
    /// property of the schedule the engines may use.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "hop(")]
    fn a_grid_whose_hop_cannot_cover_the_overlap_is_rejected() {
        chunk_starts(10 * 176_400, 176_400, 100_000);
    }

    /// The plan's proofs, parameterised over both window bindings rather than read
    /// from constants: a binding that drifts would silently change what a resumed run
    /// replays.
    #[test]
    fn resume_plan_holds_for_both_grids() {
        for &(win, overlap) in &[(176_400usize, 66_150usize), (352_800, 110_250)] {
            let hop = win - overlap;
            assert!(hop >= overlap, "win={win}: hop must cover the overlap");

            // Property scan: every sample receives at least one contribution, and a
            // checkpoint is always snapped back to a window boundary.
            let totals: Vec<usize> = (0..24)
                .flat_map(|k| {
                    let base = k * (hop / 9);
                    [
                        win / 2,
                        win,
                        win + 1,
                        win + base,
                        2 * win + base,
                        3 * win + 2 * hop + base,
                    ]
                })
                .collect();
            for total in totals {
                let starts = chunk_starts(total, win, overlap);
                assert!(
                    !starts.is_empty(),
                    "win={win} total={total}: empty schedule"
                );
                assert_eq!(starts[0], 0);
                for w in starts.windows(2) {
                    assert!(w[1] > w[0], "win={win} total={total}: {starts:?}");
                    assert!(
                        w[1] - w[0] <= win,
                        "win={win} total={total}: hole between {} and {}",
                        w[0],
                        w[1]
                    );
                }
                let n = starts.len();

                let mut probes: Vec<usize> = vec![0, 1, total.saturating_sub(1), total];
                for &s in &starts {
                    probes.extend([s.saturating_sub(1), s, s + 1]);
                }
                for frames in probes.into_iter().filter(|&f| f <= total) {
                    let p = resume_plan(&starts, total, win, frames);
                    assert!(
                        p.keep_frames <= frames,
                        "win={win} total={total} frames={frames}: kept {p:?}"
                    );
                    if frames >= total {
                        assert_eq!(
                            (p.first, p.keep_frames, p.prime_from),
                            (n, total, n),
                            "win={win} total={total}: a complete track must publish without inferring"
                        );
                        continue;
                    }
                    assert!(p.first < n, "win={win} total={total} frames={frames}");
                    assert_eq!(
                        p.keep_frames, starts[p.first],
                        "win={win} total={total} frames={frames}"
                    );
                    assert!(p.prime_from <= p.first);
                    assert_eq!(
                        p.prime_from < p.first,
                        p.first > 0,
                        "win={win} total={total} frames={frames}: {p:?}"
                    );
                    assert!(
                        p.first + 1 >= n || starts[p.first + 1] > frames,
                        "win={win} total={total} frames={frames}: kept too little, {p:?}"
                    );
                    for q in p.prime_from..p.first {
                        assert!(
                            reaches_past(q, &starts, total, win, p.keep_frames),
                            "win={win} total={total} frames={frames}: window {q} replayed for nothing"
                        );
                    }
                    if p.prime_from > 0 {
                        assert!(
                            !reaches_past(p.prime_from - 1, &starts, total, win, p.keep_frames),
                            "win={win} total={total} frames={frames}: window {} replayed for nothing",
                            p.prime_from - 1
                        );
                    }
                }
            }

            // The two-window seam: the appended chunk lands inside an earlier window,
            // so the pending region has two contributors and the retry must replay
            // both. One concrete total per grid.
            let tail_total = if win == 176_400 { 3 * win } else { 850_000 };
            let starts = chunk_starts(tail_total, win, overlap);
            let n = starts.len();
            assert_eq!(starts[n - 1], tail_total - win);
            assert!(
                starts[n - 1] - starts[n - 2] < overlap,
                "win={win}: expected an appended tail chunk, {starts:?}"
            );
            let p = resume_plan(&starts, tail_total, win, starts[n - 1]);
            assert_eq!((p.first, p.prime_from), (n - 1, n - 3), "win={win}");

            // Every regular seam replays exactly the windows that reach it.
            let total = 3 * win;
            let starts = chunk_starts(total, win, overlap);
            let n = starts.len();
            for i in 0..n {
                let flush_end = if i < n - 1 { starts[i + 1] } else { total };
                let p = resume_plan(&starts, total, win, flush_end);
                assert_eq!(p.keep_frames, flush_end.min(total), "win={win} window {i}");
                if flush_end == total {
                    continue;
                }
                assert_eq!(
                    p.first,
                    i + 1,
                    "win={win}: window {i} finalized to {flush_end}"
                );
                assert_eq!(
                    p.first - p.prime_from,
                    (0..p.first)
                        .filter(|&m| reaches_past(m, &starts, total, win, p.keep_frames))
                        .count(),
                    "win={win}: window {i} replay set is not exactly the windows reaching {flush_end}"
                );
            }
        }
    }

    /// A resume's *cost* has a bound, and the bound is what makes the design worth
    /// paying for: at most two windows of recompute, whatever the track length.
    #[test]
    fn a_resume_replays_at_most_two_windows() {
        for &(win, overlap) in &[(176_400usize, 66_150usize), (352_800, 110_250)] {
            for total in [win + 1, 3 * win, 7 * win, 44_100 * 600] {
                let starts = chunk_starts(total, win, overlap);
                for &s in &starts {
                    let p = resume_plan(&starts, total, win, s);
                    assert!(
                        p.first - p.prime_from <= 2,
                        "win={win} total={total} at {s}: replaying {} windows",
                        p.first - p.prime_from
                    );
                }
            }
        }
    }

    /// Streaming the input means each window decodes `win` frames while the grid only
    /// advances `hop`, so a track is read `win/hop` times over. That re-read is the
    /// price of not holding the mix in RAM, and it is a property of the grid rather
    /// than of the decoder: with a 4 s window and a 1.5 s overlap the factor is
    /// 176_400/110_250 = 1.6, and with an 8 s window and 2.5 s it is 352_800/242_550
    /// = 1.45. Recorded here so the number is attached to the geometry that produces
    /// it.
    #[test]
    fn re_read_factor_follows_from_the_grid() {
        for &(win, overlap) in &[(176_400usize, 66_150usize), (352_800, 110_250)] {
            let total = 44_100 * 60;
            let starts = chunk_starts(total, win, overlap);
            // Same arithmetic as the engine loop: each window decodes
            // `min(total - start, win)` frames.
            let decoded: usize = starts.iter().map(|&s| (total - s).min(win)).sum();
            assert!(decoded >= total, "the grid must cover every frame");
            let ratio = decoded as f64 / total as f64;
            let expected = win as f64 / (win - overlap) as f64;
            assert!(
                (ratio - expected).abs() < 0.1,
                "win={win}: decoded {ratio:.2}x the track, expected {expected:.2}x from the grid"
            );
        }
    }
}
