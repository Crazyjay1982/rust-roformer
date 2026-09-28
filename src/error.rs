//! Error type for the whole crate.
//!
//! Hand-rolled rather than macro-generated: the variants below are the ones the
//! engines can actually reach, and each one carries the numbers a caller needs
//! to decide what to do (a `Memory` refusal, for example, reports both what the
//! window needs and what the OS says is available).

use std::fmt;
use std::io;
use std::path::PathBuf;

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Filesystem or stream failure, with the path we were working on.
    Io {
        path: Option<PathBuf>,
        source: io::Error,
    },
    /// A WAV file that is not what we expected: unreadable header, unsupported
    /// sample width, truncated data chunk.
    Wav { path: PathBuf, detail: String },
    /// Resampling refused or failed (chunked resampler, see `audio`).
    Resample { detail: String },
    /// The model file is missing, unreadable, or not the graph we expect.
    Model { detail: String },
    /// The inference session could not be created or run.
    Session { detail: String },
    /// Inference produced something unusable: missing output, wrong shape,
    /// non-finite samples.
    Output { detail: String },
    /// The window this model asks for does not fit the memory budget.
    /// `need_mb` is a measured/estimated per-forward figure, not a guess about
    /// the whole track; see `mem` for how it is derived.
    Memory { need_mb: u64, avail_mb: Option<u64> },
    /// The caller's [`crate::config::CancelFlag`] was set.
    Cancelled,
    /// Feature not compiled in, or not available on this platform.
    Unsupported { what: &'static str },
}

impl Error {
    /// Convenience for the common `io::Error` + known path case.
    pub fn io_at(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Error::Io {
            path: Some(path.into()),
            source,
        }
    }

    /// True when the failure means "this machine is too small for this window",
    /// as opposed to "the model file is broken, so re-download it".
    ///
    /// The distinction exists because the wrong mapping is expensive: an
    /// under-sized laptop asked for an 8-second window will fail identically
    /// after any number of re-downloads. Our own [`Error::Memory`] answers by
    /// variant (a `Display` string match would rot the moment the wording
    /// changes), while text from a runtime we do not control (ONNX Runtime, MLX)
    /// is classified by [`crate::mem::is_allocation_failure`].
    pub fn looks_like_allocation_failure(&self) -> bool {
        match self {
            Error::Memory { .. } => true,
            Error::Session { detail } | Error::Output { detail } | Error::Model { detail } => {
                crate::mem::is_allocation_failure(detail)
            }
            other => crate::mem::is_allocation_failure(&other.to_string()),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io {
                path: Some(p),
                source,
            } => {
                write!(f, "I/O error at {}: {}", p.display(), source)
            }
            Error::Io { path: None, source } => write!(f, "I/O error: {}", source),
            Error::Wav { path, detail } => {
                write!(f, "WAV error at {}: {}", path.display(), detail)
            }
            Error::Resample { detail } => write!(f, "resample error: {}", detail),
            Error::Model { detail } => write!(f, "model error: {}", detail),
            Error::Session { detail } => write!(f, "inference session error: {}", detail),
            Error::Output { detail } => write!(f, "model output error: {}", detail),
            Error::Memory { need_mb, avail_mb } => match avail_mb {
                Some(avail) => write!(
                    f,
                    "memory gate: this window needs ~{} MB per forward, {} MB available",
                    need_mb, avail
                ),
                None => write!(
                    f,
                    "memory gate: this window needs ~{} MB per forward; \
                     the platform exposes no usable figure, pass \
                     SeparationOptions::memory_budget_mb to decide explicitly",
                    need_mb
                ),
            },
            Error::Cancelled => write!(f, "cancelled by caller"),
            Error::Unsupported { what } => write!(f, "not supported: {}", what),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(source: io::Error) -> Self {
        Error::Io { path: None, source }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> PathBuf {
        PathBuf::from("m.onnx")
    }

    #[test]
    fn our_own_refusal_classifies_by_variant_not_wording() {
        // The expensive mistake this guards: treating "too small" as "file
        // broken" sends an embedding host into delete → re-download → rerun on a
        // machine that can never succeed. Rewording Display must not break it.
        let e = Error::Memory {
            need_mb: 19_400,
            avail_mb: Some(15_000),
        };
        assert!(e.looks_like_allocation_failure());
        assert!(e.to_string().contains("19400"), "{e}");
    }

    #[test]
    fn runtime_text_is_still_sniffed() {
        // Verbatim shape of the failure the arena produces on an under-sized box.
        let e = Error::Session {
            detail: "Got: non-OK CUDA status / BFCArena::AllocateMem chunk allocation failed \
                     while trying to allocate 1073741824 bytes"
                .into(),
        };
        assert!(e.looks_like_allocation_failure());
    }

    #[test]
    fn a_missing_or_broken_model_is_not_an_allocation_failure() {
        assert!(!Error::Model {
            detail: format!("model file not found: {}", p().display())
        }
        .looks_like_allocation_failure());
        assert!(!Error::io_at(p(), io::Error::other("permission denied"))
            .looks_like_allocation_failure());
        assert!(!Error::Cancelled.looks_like_allocation_failure());
    }
}
