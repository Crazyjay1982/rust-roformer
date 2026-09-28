//! Cheap memory accounting, used to decide *before* a step starts whether the
//! step can finish, and to log where the peak actually happens during a run.
//!
//! ## Why "commit" and not "free RAM"
//!
//! ONNX Runtime's CPU failures on Windows are commit-charge failures
//! (`BFCArena ... Failed to allocate memory`), i.e. RAM + pagefile together,
//! while the working set is still small. A machine with 6 GB of free RAM can
//! still fail an 8 GB allocation if the pagefile is capped. On Windows we
//! therefore gate on `ullAvailPageFile` (available commit) and never on
//! physical free alone. macOS has no commit ceiling to query — memory is
//! overcommitted and the kernel kills the process rather than refusing an
//! allocation — so there [`MemSnapshot::commit_available`] is `None`, callers
//! must not downgrade an engine on a `None`, and what this module reports for
//! that platform instead is the quantity the killer uses: this process's
//! `phys_footprint`.
//!
//! ## "Peak" is not one quantity across platforms
//!
//! [`MemSnapshot::proc_peak_rss_mb`] is the field a caller reaches for when it
//! wants "how big did this get", and the three platforms answer three different
//! questions:
//!
//! | OS | current | peak | peak can come down? |
//! |---|---|---|---|
//! | Windows | `WorkingSetSize` (pages actually in RAM) | `PeakWorkingSetSize` | yes — trimming lowers it |
//! | Linux | `VmRSS` | `VmHWM` | no, and it counts shared pages in full |
//! | macOS | `ri_phys_footprint` | `ri_lifetime_max_phys_footprint` | no — lifetime high-water since process start |
//!
//! Two consequences worth stating because they bite quietly. On Windows the
//! working-set pair is the *wrong* half of the story for feasibility: it moves
//! with trimming and paging, so a run can look flat there while its commit
//! charge climbs to the ceiling; [`MemSnapshot::proc_peak_commit_mb`]
//! (`PeakPagefileUsage`) is the one that pairs with `ullAvailPageFile`. And on
//! macOS a lifetime peak includes everything the process has ever held, so it
//! describes the process, not the call you just made — which is why
//! [`peak_mb_while`] exists to bracket a single call, and why
//! `SeparationReport::peak_mb` should be filled from it rather than from the
//! OS peak.

use serde::Serialize;

use crate::error::{Error, Result};

/// How often [`peak_mb_while`] samples, in milliseconds.
const SAMPLE_INTERVAL_MS: u64 = 20;

/// One point-in-time reading. All fields are MiB — this crate says "MB" and
/// means 1024*1024 bytes throughout — and `None` = "this platform does not
/// report it", which is deliberately distinct from 0.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MemSnapshot {
    /// Installed physical memory.
    pub total_phys_mb: Option<u64>,
    /// Physical memory immediately available to a new allocation.
    pub avail_phys_mb: Option<u64>,
    /// RAM + pagefile, i.e. the ceiling on committed memory.
    pub commit_limit_mb: Option<u64>,
    /// Headroom left below `commit_limit_mb`. This is the number that predicts
    /// an ORT allocation failure, and on Windows it is the *only* one that
    /// does: `ullAvailPageFile`, i.e. how much more this machine can commit.
    /// Physical free is not it, and the two differ by gigabytes on a machine
    /// with a capped pagefile.
    pub commit_avail_mb: Option<u64>,
    /// This process's current commit charge (`PagefileUsage`; `VmSize` on
    /// Linux, where that is address space rather than charge — the closest
    /// per-process analogue the platform exports). `None` on macOS, which does
    /// not keep this accounting.
    pub proc_commit_mb: Option<u64>,
    /// This process's peak commit charge since it started
    /// (`PeakPagefileUsage` / `VmPeak`). The pair that belongs with
    /// [`MemSnapshot::commit_avail_mb`], and the one to read if you want to
    /// know how close a finished run came to a refusal.
    pub proc_peak_commit_mb: Option<u64>,
    /// What this process is holding: the working set on Windows, `VmRSS` on
    /// Linux, and on macOS the `phys_footprint` — the number the jetsam killer
    /// compares against the per-process limit, which is far above what `ps`
    /// calls resident for an accelerator-backed workload (measured on an Apple
    /// Silicon separation of a long track: 7110 MB footprint against a 910 MB
    /// resident peak, 7.8x).
    pub proc_rss_mb: Option<u64>,
    /// Peak of [`MemSnapshot::proc_rss_mb`] since the process started
    /// (`PeakWorkingSetSize` / `VmHWM` / `ri_lifetime_max_phys_footprint`, the
    /// last of those clamped up to the current value — see
    /// `macos::footprint_mb` for why).
    ///
    /// All three are maxima, but they max over different quantities, and none
    /// of them is "the largest this call got": on Windows the working set is
    /// trimmable, so its peak records what stayed in RAM, not what was asked
    /// for; on Linux `VmHWM` charges shared pages to every process sharing
    /// them; on macOS the lifetime maximum of `phys_footprint` never comes
    /// down for the life of the process, so after one large run every later
    /// snapshot carries that earlier peak. For a per-call figure use
    /// [`peak_mb_while`]; for a feasibility question on Windows use
    /// [`MemSnapshot::proc_peak_commit_mb`] alongside
    /// [`MemSnapshot::commit_avail_mb`].
    pub proc_peak_rss_mb: Option<u64>,
}

impl MemSnapshot {
    /// Commit headroom, if this platform reports it. `None` on macOS.
    pub fn commit_available(&self) -> Option<u64> {
        self.commit_avail_mb
    }

    /// The single scalar worth tracking on this platform: the commit charge
    /// where the OS keeps that accounting (Windows, Linux), and otherwise what
    /// the process is holding. This is what makes the sampling in
    /// [`peak_mb_while`] report a curve on macOS instead of the `None` that
    /// [`MemSnapshot::proc_commit_mb`] is there by design.
    ///
    /// Not a substitute for [`Self::fits`]: only commit headroom can authorise
    /// an engine downgrade, and `held_mb` says nothing about the ceiling.
    pub fn held_mb(&self) -> Option<u64> {
        self.proc_commit_mb.or(self.proc_rss_mb)
    }

    /// Whether `need_mb` of *additional* commit fits in what is left.
    ///
    /// Returns `None` when the platform does not report commit, so callers
    /// cannot silently treat "unknown" as "fine" or "not fine" — they must
    /// decide which of the two they mean. [`window_fits`] is that decision
    /// already made, for the common case.
    pub fn fits(&self, need_mb: u64) -> Option<bool> {
        self.commit_avail_mb.map(|avail| avail >= need_mb)
    }

    /// The most authoritative figure this platform offers for "could this
    /// process be handed `need_mb` more?", in priority order:
    ///
    /// 1. commit headroom, where the OS enforces a commit ceiling (always
    ///    Windows; Linux under strict overcommit accounting);
    /// 2. physically available RAM, where the OS reports it but commit is not a
    ///    ceiling worth quoting — the usual Linux configuration, where the
    ///    heuristic lets `Committed_AS` pass `CommitLimit` (see
    ///    `linux::snapshot` for both cases);
    /// 3. installed RAM minus what this process already holds — on macOS there
    ///    is no commit ceiling to consult, and this is the only OS-side number
    ///    that says anything about capacity. It is deliberately conservative
    ///    (see [`window_fits`]);
    /// 4. `None`, meaning "no figure", which a caller must treat as "the check
    ///    was not made", never as a verdict.
    ///
    /// Note that 3 subtracts *this process's* footprint, and the footprint
    /// includes pages the allocator has already decided to reuse. A window
    /// that is refused by rule 3 with a large existing footprint is a real
    /// judgement call, not a bug: `Some(budget)` on [`window_fits`] is how a
    /// host that knows better overrides it.
    pub fn preflight_ceiling_mb(&self) -> Option<u64> {
        if self.commit_avail_mb.is_some() {
            return self.commit_avail_mb;
        }
        if self.avail_phys_mb.is_some() {
            return self.avail_phys_mb;
        }
        match (self.total_phys_mb, self.held_mb()) {
            (Some(total), Some(held)) => Some(total.saturating_sub(held)),
            (Some(total), None) => Some(total),
            _ => None,
        }
    }

    /// One-line form for logs and for the error detail we surface, e.g.
    /// `phys 8192/16384MB commit 6200/24500MB proc 5100(peak 8800)/12000(peak 15000)MB`.
    /// A platform that has no commit accounting prints `?` there and still
    /// carries the process half — on macOS that half is the footprint.
    pub fn summary(&self) -> String {
        let opt = |v: Option<u64>| v.map(|x| x.to_string()).unwrap_or_else(|| "?".to_string());
        format!(
            "phys {}/{}MB commit {}/{}MB proc {}(peak {})/{}(peak {})MB",
            opt(self.avail_phys_mb),
            opt(self.total_phys_mb),
            opt(self.commit_avail_mb),
            opt(self.commit_limit_mb),
            opt(self.proc_commit_mb),
            opt(self.proc_peak_commit_mb),
            opt(self.proc_rss_mb),
            opt(self.proc_peak_rss_mb),
        )
    }
}

/// Read the counters. Never fails: `None` means the platform has no cheap API
/// here, and callers must proceed as if the check had not been made.
pub fn snapshot() -> Option<MemSnapshot> {
    #[cfg(target_os = "windows")]
    {
        windows::snapshot()
    }
    #[cfg(target_os = "linux")]
    {
        linux::snapshot()
    }
    #[cfg(target_os = "macos")]
    {
        macos::snapshot()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Log a snapshot when the step-to-step growth is the thing we care about:
/// `label` is a short site name, `seq` is a monotonically increasing index so
/// `grep` over a user's log shows the curve.
///
/// Engines call this between windows. It costs one counter read — a
/// `proc_pid_rusage` plus a `sysctl` on macOS, two small `/proc` reads on
/// Linux — which is why it can sit inside a window loop at a cadence of some
/// number of windows rather than every sample. The cadence is the value: a
/// curve sampled every few dozen windows is what lets a field report be read
/// as "commit is flat" instead of "commit climbs until the machine runs out".
/// Does nothing when [`snapshot`] has nothing to report.
pub fn log_point(seq: usize, label: &str) {
    if let Some(s) = snapshot() {
        log::info!("[mem] #{seq} {label}: {}", s.summary());
    }
}

/// Peak of [`MemSnapshot::held_mb`] (MB) observed while `f` runs, sampled every
/// 20 ms, alongside `f`'s return value.
///
/// This is the honest way to fill `SeparationReport::peak_mb`: the OS-side
/// peaks are lifetime-since-process-start (and, on Windows, an evictable
/// working set), whereas this is the bracket around the work you asked about.
///
/// The sampling is the point: a reading taken after the work returns reports
/// the resident plateau and misses the transient entirely, which is what made
/// an earlier round of this investigation understate the real peak by around
/// ten times. 20 ms is a compromise, not a bound — a transient shorter than
/// the interval can be missed, so treat the result as "at least this", and
/// raise the cadence locally if you need to resolve a narrow spike.
///
/// Baselines subtracted from this must come from [`MemSnapshot::held_mb`] too —
/// reading `proc_commit_mb` for the base would mix a `None` (macOS) with a
/// footprint peak.
///
/// ```
/// use rust_roformer::mem;
/// // Holds 64 MiB and touches it, so the sampler has something to see.
/// let (n, peak) = mem::peak_mb_while(|| {
///     let mut buf = vec![0u8; 64 * 1024 * 1024];
///     for slot in buf.chunks_mut(4096) {
///         slot[0] = 1;
///     }
///     std::thread::sleep(std::time::Duration::from_millis(120));
///     buf.len()
/// });
/// assert_eq!(n / (1024 * 1024), 64);
/// // A platform with no counters reports 0 rather than inventing a peak.
/// assert!(peak == 0 || peak >= 64, "peak {peak}");
/// ```
pub fn peak_mb_while<T>(f: impl FnOnce() -> T) -> (T, u64) {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let (s, p) = (Arc::clone(&stop), Arc::clone(&peak));
    let sampler = std::thread::spawn(move || {
        while !s.load(Ordering::Relaxed) {
            if let Some(x) = snapshot() {
                p.fetch_max(x.held_mb().unwrap_or(0), Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(SAMPLE_INTERVAL_MS));
        }
    });

    // Stops and joins the sampler even if `f` unwinds, so a panicking `f`
    // cannot leave a thread reading this process behind it.
    struct SamplerGuard(Option<std::thread::JoinHandle<()>>, Arc<AtomicBool>);
    impl Drop for SamplerGuard {
        fn drop(&mut self) {
            self.1.store(true, Ordering::Relaxed);
            if let Some(handle) = self.0.take() {
                let _ = handle.join();
            }
        }
    }
    let _guard = SamplerGuard(Some(sampler), Arc::clone(&stop));

    let out = f();
    // One read on the way out, taken by this thread: the sampler sleeps up to
    // `SAMPLE_INTERVAL_MS`, so without it the sample nearest the end of `f` is
    // up to one interval stale.
    if let Some(x) = snapshot() {
        peak.fetch_max(x.held_mb().unwrap_or(0), Ordering::Relaxed);
    }
    (out, peak.load(Ordering::Relaxed))
}

/// Does this engine error read like "the allocation was refused"?
///
/// Used to decide whether a failed step is a memory answer — re-run it on a
/// smaller window or a lighter engine — or everything else, which re-running
/// cannot fix. Matching is on the message text because ONNX Runtime surfaces
/// its allocator failures as plain strings (`ort` forwards the C API status),
/// e.g. `bfsa1::Allocator<...>::allocate(...) failed to allocate memory` and
/// `Got a bad allocation... BFCArena`.
///
/// The asymmetry is the whole reason this is a curated list rather than a
/// `contains("memory")` test. A machine that is too small for a window will
/// say so forever; a corrupt model file says so once and is fixed by a
/// re-download. Mapping the first onto the second buys a
/// delete-and-re-download-and-rerun loop on a host that can never succeed:
/// minutes of bandwidth and CPU to reproduce the identical failure, having
/// deleted the model file that was fine. Mapping the second onto the first
/// loses the only recovery that works. So the list is deliberately narrow, and
/// the exclusions are as considered as the inclusions:
///
/// * a cancellation must not be re-run on another engine — that would turn a
///   user-visible "you cancelled" into a several-minute silent re-run;
/// * a missing or unreadable model must not be read as pressure;
/// * a decode, I/O or shape error must not either.
///
/// `Error::Memory` is a gate refusal this crate produced itself
/// ([`window_fits`]), and it is the strongest memory signal there is, so its
/// own message prefix is in the set: a caller that routes an error message
/// through here must not send a refusal down the re-download branch.
///
/// ```
/// use rust_roformer::mem;
/// assert!(mem::is_allocation_failure(
///     "BFCArena malloc failed to allocate memory of size 4194304 bytes"));
/// assert!(!mem::is_allocation_failure("model error: no such file melband.onnx"));
/// ```
pub fn is_allocation_failure(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    if t.contains("cancel") {
        return false;
    }
    const MARKERS: [&str; 10] = [
        // ONNX Runtime's arena allocator names itself `BFCArena`; lowercased that
        // is "bfcarena". This marker spent its whole life in the app mis-typed as
        // "bfcaarena", where it could never fire — and stayed invisible because
        // the messages quoted next to it also contain "failed to allocate".
        "bfcarena",
        "failed to allocate",
        "memory allocation of",
        "bad allocation",
        "bad_alloc",
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "cannot allocate",
        // This crate's own pre-flight refusal, from `Error::Memory`'s Display.
        "memory gate:",
    ];
    MARKERS.iter().any(|m| t.contains(m))
}

/// The pre-flight gate: refuse to start a window that cannot fit.
///
/// `need_mb` is the *additional* memory one forward pass costs — a measured
/// per-forward figure from the engine (weights resident, arena growth observed
/// across a few windows), not an estimate about the whole track, because the
/// whole track never coexists: input is read a window at a time and both stems
/// are streamed out.
///
/// `budget_override` is `SeparationOptions::memory_budget_mb`: `Some(mb)` says
/// "the host knows its own ceiling, use `mb` and do not ask the OS"; `None`
/// says "ask the OS", which resolves through
/// [`MemSnapshot::preflight_ceiling_mb`] — commit headroom where the OS keeps
/// that accounting, installed-minus-resident on macOS.
///
/// Two decisions worth having written down:
///
/// * **No figure is not a refusal.** On a platform this module cannot read,
///   the gate answers "unknown" and the run proceeds, with a `warn!` saying so.
///   The alternative — fail closed — would make this crate unusable on any host
///   whose counters we have not mapped, and "unusable" is not the same message
///   as "too small".
/// * **A macOS refusal is conservative.** The ceiling there subtracts this
///   process's current footprint, so a process that already holds several
///   gigabytes can be refused a window that the engine's arena reuse would
///   have absorbed. Pass a budget in that case. This asymmetry is intentional:
///   a false refusal costs one retried call, a false acceptance costs an hour
///   of partial work and, on macOS, a jetsam kill that leaves no error from us
///   at all.
///
/// Call it before the first forward, not immediately after releasing memory:
/// the kernel debits `phys_footprint` asynchronously, so a footprint read right
/// after a cache-clearing call is a mid-flight number (see `macos::footprint_mb`).
///
/// ```
/// use rust_roformer::mem;
/// // An explicit budget decides without consulting the OS.
/// assert!(mem::window_fits(1000, Some(2000)).is_ok());
/// assert!(mem::window_fits(3000, Some(2000)).is_err());
/// ```
pub fn window_fits(need_mb: u64, budget_override: Option<u64>) -> Result<()> {
    let snap = snapshot();
    match gate(need_mb, budget_override, snap.as_ref()) {
        Some(ceiling) => {
            let err = Error::Memory {
                need_mb,
                avail_mb: Some(ceiling),
            };
            log::warn!(
                "{err}; counters: {}",
                snap.map(|s| s.summary())
                    .unwrap_or_else(|| "unavailable".to_string())
            );
            Err(err)
        }
        None => {
            if snap.and_then(|s| s.preflight_ceiling_mb()).is_none() && budget_override.is_none() {
                log::warn!(
                    "[mem] memory gate could not decide a ~{need_mb} MB window: no usable \
                     figure from this platform; set memory_budget_mb to gate explicitly"
                );
            }
            Ok(())
        }
    }
}

/// [`window_fits`]'s decision as a pure function, so the whole truth table is
/// testable without an OS to read. `Some(ceiling)` means "refuse, and this was
/// the figure it refused against".
fn gate(need_mb: u64, budget_override: Option<u64>, snap: Option<&MemSnapshot>) -> Option<u64> {
    let ceiling = match budget_override {
        Some(mb) => Some(mb),
        None => snap.and_then(|s| s.preflight_ceiling_mb()),
    };
    ceiling.filter(|&ceiling| need_mb > ceiling)
}

#[cfg(target_os = "windows")]
mod windows {
    use super::MemSnapshot;

    use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    const MB: u64 = 1024 * 1024;

    /// `GetCurrentProcess()` returns the pseudo-handle documented as
    /// `(HANDLE)-1`, which is what `INVALID_HANDLE_VALUE` is. Spelling the
    /// constant rather than calling the API keeps this module inside the
    /// `windows-sys` features this crate declares — the function lives behind
    /// `Win32_System_Threading`, and pulling a feature for one constant the
    /// platform documents as fixed is not worth the dependency surface.
    const THIS_PROCESS: HANDLE = INVALID_HANDLE_VALUE;

    pub fn snapshot() -> Option<MemSnapshot> {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        // The API refuses the call unless it recognises this size.
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok == 0 {
            return None;
        }

        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        let got = unsafe {
            GetProcessMemoryInfo(
                THIS_PROCESS,
                &mut counters as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS,
                cb,
            )
        };
        // The pseudo-handle always works, but if the call ever fails we still
        // return the system-wide half.
        let (proc_commit, proc_peak_commit, proc_rss, proc_peak_rss) = if got != 0 {
            (
                // `PagefileUsage` is this process's commit charge (the `_EX`
                // struct's `PrivateUsage` is documented as the same quantity),
                // and `PeakPagefileUsage` its high-water. That pair is what
                // belongs with `ullAvailPageFile`: it is the accounting the
                // allocation is charged against.
                Some(counters.PagefileUsage as u64 / MB),
                Some(counters.PeakPagefileUsage as u64 / MB),
                Some(counters.WorkingSetSize as u64 / MB),
                // Read this one for what it is: the working set is pages in
                // physical RAM, which the system trims when it wants the memory
                // back. It is the least trustworthy number in this struct, and
                // never the one to gate on.
                Some(counters.PeakWorkingSetSize as u64 / MB),
            )
        } else {
            (None, None, None, None)
        };

        Some(MemSnapshot {
            total_phys_mb: Some(status.ullTotalPhys / MB),
            avail_phys_mb: Some(status.ullAvailPhys / MB),
            commit_limit_mb: Some(status.ullTotalPageFile / MB),
            commit_avail_mb: Some(status.ullAvailPageFile / MB),
            proc_commit_mb: proc_commit,
            proc_peak_commit_mb: proc_peak_commit,
            proc_rss_mb: proc_rss,
            proc_peak_rss_mb: proc_peak_rss,
        })
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::MemSnapshot;

    const KB_TO_MB: u64 = 1024;

    fn field(text: &str, key: &str) -> Option<u64> {
        text.lines().find_map(|line| {
            let rest = line.strip_prefix(key)?;
            // No `trim()` here: `split_whitespace` already skips leading
            // whitespace, and `field` is the whole parser for `/proc/meminfo`.
            rest.split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|kb| kb / KB_TO_MB)
        })
    }

    pub fn snapshot() -> Option<MemSnapshot> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let selfstat = std::fs::read_to_string("/proc/self/status")
            .ok()
            .unwrap_or_default();
        // Committed_AS/CommitLimit are the same quantity Windows calls commit
        // charge/ceiling, and the subtraction is the same figure: how much more
        // this kernel will accept before an allocation is refused.
        //
        // Two configurations make that pair say something else, and both are
        // handled by reporting no commit figure rather than a wrong one. The
        // first is `overcommit_memory=0`, the default: the heuristic is allowed
        // to approve allocations past the limit, so `Committed_AS` can sit
        // above `CommitLimit` — the `limit >= used` test below exists because
        // subtracting anyway would produce an unsigned wrap-around that reads
        // as a monstrous headroom, and as a *refusal* if it ever wrapped the
        // other way. The second is `overcommit_memory=1`, where the kernel
        // prints an unsigned-max sentinel for `CommitLimit`: the resulting
        // headroom is real arithmetic over a figure that means "no ceiling", so
        // the gate will essentially never refuse. That is the configuration
        // saying what it means — allocate past RAM, reap with the OOM killer —
        // and a host that has done it deliberately and still wants a gate
        // should pass `memory_budget_mb`.
        //
        // One more limitation, this one about *which* machine is being read:
        // `/proc/meminfo` reports the host, so inside a cgroup-limited
        // container (the usual shape for a CI runner or a serving pod)
        // `MemTotal` and `MemAvailable` can both be several times the quota
        // this process may actually use, and the kernel's answer to exceeding
        // that quota is an OOM kill rather than a failed allocation.
        Some(MemSnapshot {
            total_phys_mb: field(&meminfo, "MemTotal:"),
            avail_phys_mb: field(&meminfo, "MemAvailable:"),
            commit_limit_mb: field(&meminfo, "CommitLimit:"),
            commit_avail_mb: match (
                field(&meminfo, "CommitLimit:"),
                field(&meminfo, "Committed_AS:"),
            ) {
                (Some(limit), Some(used)) if limit >= used => Some(limit - used),
                _ => None,
            },
            // `VmSize` is address space, not touched pages: it is the closest
            // per-process analogue of a commit charge that Linux exposes here,
            // and it is what a cgroup-less host will actually be limited by.
            proc_commit_mb: field(&selfstat, "VmSize:"),
            proc_peak_commit_mb: field(&selfstat, "VmPeak:"),
            proc_rss_mb: field(&selfstat, "VmRSS:"),
            proc_peak_rss_mb: field(&selfstat, "VmHWM:"),
        })
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::MemSnapshot;

    const MB: u64 = 1024 * 1024;

    /// Read an integer sysctl in-process. Shelling out to `sysctl(8)` is not an
    /// option for a library: a sandboxed host may forbid exec entirely, and the
    /// cost is a process spawn instead of one syscall.
    ///
    /// `name` is the NUL-terminated sysctl name; passing bytes rather than a
    /// `CStr` keeps this off the C-string-literal feature, which is newer than
    /// the Rust version this crate declares.
    fn sysctl_u64(name: &[u8]) -> Option<u64> {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr() as *const libc::c_char,
                &mut value as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && len == std::mem::size_of::<u64>() {
            Some(value)
        } else {
            None
        }
    }

    /// This process's `phys_footprint` and its lifetime high-water mark.
    ///
    /// `task_for_pid` would need an entitlement, but `proc_pid_rusage` does not
    /// — it is the same source `footprint(1)` reads. The binding is declared
    /// against an opaque `*mut c_void` buffer, so the size the kernel copies is
    /// agreed by the flavor argument plus the caller's struct layout; using
    /// libc's own `rusage_info_v4` keeps that layout off us.
    ///
    /// The lifetime maximum is clamped up to the current value rather than
    /// trusted, because the kernel refreshes it a step behind the ledger it
    /// reads the current value from: measured over 835 455 back-to-back
    /// `proc_pid_rusage` calls taken while the footprint climbed, 30 389 of
    /// them (3.6%) reported a maximum *below* the current value, worst gap
    /// 147 456 bytes — enough for the MiB-truncated pair below to read
    /// `peak = current - 1`. The same 200 000 calls taken while nothing was
    /// allocated reported no inversion, which is what pins it to growth rather
    /// than to the read itself. Clamping is exact at any lag size, so no
    /// tolerance is needed, and nothing that consumes these two can observe the
    /// difference except the invariant they are documented to satisfy — which
    /// is the pair `peak >= current` that [`super::MemSnapshot`] promises and
    /// `peak_never_trails_current_while_memory_is_climbing` re-checks under a
    /// climbing ledger. That ledger is a strict test precisely because
    /// truncation hides most of the skew: the raw-byte inversion rate above is
    /// 3.6%, while the MiB-truncated pair inverts on only tens of reads in
    /// 200 000 (76 observed here with the clamp removed, 0 with it), so the
    /// paired reads have to be many to catch a regression.
    ///
    /// The debit side is asynchronous too, which is a sampling hazard rather
    /// than an invariant one: a cache-clearing release returns the pages to the
    /// allocator synchronously but the kernel lowers `phys_footprint` over the
    /// following fraction of a second. Reading immediately therefore samples an
    /// in-progress reclaim — one measured cache-clearing site reported 25 / 26
    /// / 233 MB "returned" on successive runs of the *same binary*, while at
    /// +0.7 s the same site read 271 / 271 / 272 / 272 / 272 MB (283 MB at
    /// another size), i.e. a repeatable number. It still drifts after that
    /// (243 MB at 2 s), so +0.7 s is a stated sampling instant, not an
    /// asymptote. Nothing here may therefore treat a footprint read taken just
    /// after a release as a result: gate before the first forward, or wait out
    /// the interval.
    fn footprint_mb() -> (Option<u64>, Option<u64>) {
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                &mut info as *mut libc::rusage_info_v4 as *mut libc::rusage_info_t,
            )
        };
        if rc != 0 {
            return (None, None);
        }
        let current = info.ri_phys_footprint;
        (
            Some(current / MB),
            Some(info.ri_lifetime_max_phys_footprint.max(current) / MB),
        )
    }

    pub fn snapshot() -> Option<MemSnapshot> {
        let total = sysctl_u64(b"hw.memsize\0").map(|b| b / MB);
        // macOS has no commit ceiling to report (memory is overcommitted and
        // the kernel kills rather than refusing an allocation), so `fits()`
        // stays None and the separation step is never downgraded here on a
        // commit reading it does not have. The same reason `avail_phys_mb` is
        // left unknown: free+inactive would not be the quantity a feasibility
        // gate needs, and inventing one from it would be a guess with a number
        // attached.
        //
        // What the kernel *does* gate on is the footprint pair below, and that
        // is what `preflight_ceiling_mb` falls back to here.
        let (footprint, peak_footprint) = footprint_mb();
        Some(MemSnapshot {
            total_phys_mb: total,
            avail_phys_mb: None,
            commit_limit_mb: None,
            commit_avail_mb: None,
            proc_commit_mb: None,
            proc_peak_commit_mb: None,
            proc_rss_mb: footprint,
            proc_peak_rss_mb: peak_footprint,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 512 MiB of touched heap: big enough to move every counter this module
    /// reads by a countable amount, small enough to stay a unit test.
    const PROBE_MB: u64 = 512;

    fn touched_mb(mb: u64) -> Vec<u8> {
        let mut buf: Vec<u8> = vec![0u8; (mb as usize) * 1024 * 1024];
        // Touch every page: an untouched reservation is not a commitment, and
        // on macOS it is not footprint either.
        for slot in buf.chunks_mut(4096) {
            slot[0] = 1;
        }
        buf
    }

    #[test]
    fn snapshot_reports_something_and_summarizes() {
        let s = snapshot().expect("memory query must work on a dev machine");
        // Physical memory is known on every platform we implement; a zero here
        // would mean the struct fields are misaligned with the API's layout.
        assert!(s.total_phys_mb.unwrap_or(0) > 0, "total_phys: {s:?}");
        assert!(s.proc_rss_mb.is_some(), "proc_rss: {s:?}");
        assert!(
            s.proc_peak_rss_mb >= s.proc_rss_mb,
            "peak below current: {s:?}"
        );
        let text = s.summary();
        // Records the platform's actual reading in the test log, which is how a
        // silently-unknown field gets noticed.
        eprintln!("[mem] {text}");
        assert!(text.contains("commit"), "{text}");
        // Unknown fields must render as '?', never as a plausible 0.
        if s.commit_avail_mb.is_none() {
            assert!(text.contains("commit ?/?MB"), "{text}");
        }
        // The gate must have something to say on every platform we implement,
        // or it has to be honest that it does not.
        assert!(
            s.preflight_ceiling_mb().is_some(),
            "no usable figure: {s:?}"
        );
    }

    #[test]
    fn fits_is_conservative_and_none_on_unknown_commit() {
        let unknown = MemSnapshot::default();
        assert_eq!(unknown.fits(1), None, "unknown commit must not answer");
        assert_eq!(
            unknown.preflight_ceiling_mb(),
            None,
            "an all-unknown snapshot must not invent a ceiling"
        );

        let s = MemSnapshot {
            commit_avail_mb: Some(1000),
            ..Default::default()
        };
        assert_eq!(s.fits(999), Some(true));
        assert_eq!(s.fits(1001), Some(false), "must gate on commit headroom");
    }

    /// `summary` is what a user pastes into a bug report, so its shape is part
    /// of the API: field order and the `?` for unknown are both pinned here
    /// rather than left to whatever the format string drifts into.
    #[test]
    fn summary_field_order_and_unknown_rendering() {
        let full = MemSnapshot {
            total_phys_mb: Some(16384),
            avail_phys_mb: Some(8192),
            commit_limit_mb: Some(24500),
            commit_avail_mb: Some(6200),
            proc_commit_mb: Some(5100),
            proc_peak_commit_mb: Some(8800),
            proc_rss_mb: Some(12000),
            proc_peak_rss_mb: Some(15000),
        };
        assert_eq!(
            full.summary(),
            "phys 8192/16384MB commit 6200/24500MB proc 5100(peak 8800)/12000(peak 15000)MB"
        );
        assert_eq!(
            MemSnapshot::default().summary(),
            "phys ?/?MB commit ?/?MB proc ?(peak ?)/?(peak ?)MB"
        );
        // The macOS shape: no commit half at all, process half still present.
        let mac = MemSnapshot {
            total_phys_mb: Some(16384),
            proc_rss_mb: Some(7110),
            proc_peak_rss_mb: Some(7110),
            ..Default::default()
        };
        assert_eq!(
            mac.summary(),
            "phys ?/16384MB commit ?/?MB proc ?(peak ?)/7110(peak 7110)MB"
        );
    }

    /// Which figure the gate compares a window against, and in what order.
    /// Pure, so all four arms are exercised on whatever machine runs them.
    #[test]
    fn preflight_ceiling_ranks_the_platform_figures_in_order() {
        // Commit headroom wins even though it is the smallest number: it is the
        // one that predicts an allocator refusal, and a gate that quietly
        // preferred a bigger figure would stop refusing.
        let win = MemSnapshot {
            total_phys_mb: Some(16384),
            avail_phys_mb: Some(8192),
            commit_avail_mb: Some(600),
            proc_rss_mb: Some(500),
            ..Default::default()
        };
        assert_eq!(win.preflight_ceiling_mb(), Some(600));

        // Linux with commit accounting off: fall to MemAvailable.
        let linux = MemSnapshot {
            total_phys_mb: Some(16384),
            avail_phys_mb: Some(8000),
            proc_rss_mb: Some(500),
            ..Default::default()
        };
        assert_eq!(linux.preflight_ceiling_mb(), Some(8000));

        // macOS: no commit, no free figure, so installed minus what this
        // process already holds.
        let mac = MemSnapshot {
            total_phys_mb: Some(16384),
            proc_rss_mb: Some(3000),
            proc_peak_rss_mb: Some(3348),
            ..Default::default()
        };
        assert_eq!(mac.preflight_ceiling_mb(), Some(13384));

        // Nothing readable at all must be `None`, i.e. "no verdict", on the way
        // to the `window_fits` arm that warns instead of refusing.
        assert_eq!(MemSnapshot::default().preflight_ceiling_mb(), None);
        assert_eq!(
            MemSnapshot {
                total_phys_mb: Some(16384),
                ..Default::default()
            }
            .preflight_ceiling_mb(),
            Some(16384),
            "installed RAM alone still bounds a window"
        );
    }

    /// The gate's truth table: need vs. ceiling, override vs. OS, and the
    /// equality edge (a window exactly the size of the headroom fits, because
    /// `fits` means `>=`).
    #[test]
    fn gate_truth_table() {
        let snap = MemSnapshot {
            total_phys_mb: Some(16384),
            commit_avail_mb: Some(4000),
            ..Default::default()
        };
        assert_eq!(gate(3999, None, Some(&snap)), None);
        assert_eq!(gate(4000, None, Some(&snap)), None, "exactly enough fits");
        assert_eq!(gate(4001, None, Some(&snap)), Some(4000));

        // An explicit budget replaces the OS figure in both directions.
        assert_eq!(gate(9000, Some(10000), Some(&snap)), None);
        assert_eq!(gate(100, Some(50), Some(&snap)), Some(50));
        assert_eq!(gate(50, Some(50), Some(&snap)), None);

        // No snapshot, no figure: never a refusal.
        assert_eq!(gate(1_000_000, None, None), None);
        assert_eq!(gate(1_000_000, None, Some(&MemSnapshot::default())), None);
        // ... but an explicit budget decides even with nothing to read.
        assert_eq!(gate(1000, Some(10), None), Some(10));
        // Zero-cost window can never be refused by a ceiling.
        assert_eq!(gate(0, Some(0), None), None);
    }

    #[test]
    fn window_fits_honours_the_explicit_budget_without_the_os() {
        assert!(window_fits(1000, Some(2000)).is_ok());
        let err = window_fits(3000, Some(2000)).expect_err("3000 > 2000 must refuse");
        match &err {
            Error::Memory { need_mb, avail_mb } => {
                assert_eq!(*need_mb, 3000);
                assert_eq!(*avail_mb, Some(2000));
            }
            other => panic!("wrong variant: {other:?}"),
        }
        // The refusal is the memory answer, so a caller routing the message
        // text through the classifier must not send it to re-download.
        assert!(err.looks_like_allocation_failure(), "{err}");
        assert!(err.to_string().contains("3000"), "{err}");
    }

    /// A window no machine can hold has to be refused wherever the platform
    /// can answer at all, and has to be *allowed* where it cannot — the two
    /// arms of "unknown is not evidence of incapacity".
    #[test]
    fn window_fits_refuses_the_absurd_and_abstains_on_no_figure() {
        const ABSURD_MB: u64 = u64::MAX / 2;
        let can_answer = snapshot().and_then(|s| s.preflight_ceiling_mb()).is_some();
        let res = window_fits(ABSURD_MB, None);
        if can_answer {
            match res {
                // Deliberately not compared against a ceiling read here: the
                // process's own footprint moves between the two reads, and on
                // the macOS arm that figure is `installed - resident`.
                Err(Error::Memory { need_mb, avail_mb }) => {
                    assert_eq!(need_mb, ABSURD_MB);
                    assert!(
                        avail_mb.is_some_and(|c| c < ABSURD_MB),
                        "must report the figure it refused against: {avail_mb:?}"
                    );
                }
                other => panic!("a ~8 exabyte window must be refused, got {other:?}"),
            }
        } else {
            assert!(res.is_ok(), "no figure must not refuse: {res:?}");
        }
    }

    /// The feasibility gate is only as good as the counters' units, so verify
    /// them against a known allocation: committing 512 MiB has to move both
    /// this process's charge and the system's remaining headroom by ~512 MiB.
    /// A swapped or mis-scaled field fails here instead of in production.
    #[test]
    fn counters_track_a_known_allocation() {
        let Some(before) = snapshot() else {
            return; // platform without counters: nothing to check
        };
        let (Some(avail0), Some(commit0)) = (before.commit_avail_mb, before.proc_commit_mb) else {
            // Absent by design on macOS, which keeps no commit accounting at
            // all, and legitimately absent on a Linux host whose overcommit
            // heuristic has let `Committed_AS` pass `CommitLimit` (see
            // `linux::snapshot`). Windows always reports both, so a `None`
            // there means the fields are wired to the wrong API.
            let commit_is_optional_here = cfg!(any(target_os = "macos", target_os = "linux"));
            assert!(
                commit_is_optional_here,
                "commit accounting missing where it cannot be: {before:?}"
            );
            return;
        };

        let buf = touched_mb(PROBE_MB);
        let during = snapshot().unwrap();
        std::hint::black_box(&buf);

        let grew = during.proc_commit_mb.unwrap().saturating_sub(commit0);
        let drained = avail0.saturating_sub(during.commit_avail_mb.unwrap());
        drop(buf);
        assert!(
            (PROBE_MB - 32..=PROBE_MB + 88).contains(&grew),
            "proc_commit grew {grew} MB for a {PROBE_MB} MB allocation: {before:?} -> {during:?}"
        );
        // The system-wide half can only be bounded loosely: `ullAvailPageFile`
        // is shared with every other process, so a browser releasing a tab
        // during the measurement window offsets our own charge (seen here:
        // 259 MB drained for a 512 MB allocation). What it does have to prove is
        // that the field is the commit counter at all and is in MB: a zero, a
        // sign flip or a bytes/KB mis-scaling (524288) fails here. A swap with
        // `ullAvailPhys` is *not* catchable by bounds (the two move by similar
        // amounts for one allocation: 698 vs 623 in the failing run), so that
        // pairing rests on `snapshot` naming each `MEMORYSTATUSEX` field, not on
        // this number.
        //
        // The upper bound has to carry the *suite*, not just the machine: tests
        // run one per logical CPU, and any sibling allocating inside this window
        // adds its own charge to the same system-wide counter (observed 698 MB
        // for a 512 MB allocation, which a 620 ceiling failed spuriously). The
        // strict half is `grew` above, which reads this process only.
        assert!(
            (150..=1400).contains(&drained),
            "commit headroom fell {drained} MB for a {PROBE_MB} MB allocation: {before:?} -> {during:?}"
        );
    }

    /// macOS has no commit accounting, so the test above can only skip it — and
    /// a skipped arm is exactly how a wrong footprint stays invisible: read
    /// through `proc_commit_mb` the separation runs report "0 MB", which looks
    /// like "no pressure" rather than "this platform does not say". Pin the
    /// macOS scalar against the same known allocation: 512 MiB of touched heap
    /// has to show up as ~512 MiB more footprint.
    #[cfg(target_os = "macos")]
    #[test]
    fn footprint_tracks_a_known_allocation() {
        let before = snapshot().expect("snapshot");
        let (Some(fp0), Some(peak0)) = (before.proc_rss_mb, before.proc_peak_rss_mb) else {
            panic!("macOS must report its footprint pair: {before:?}");
        };
        // The gate must stay shut here: a footprint is not headroom, and an
        // engine downgraded on that reading would be downgraded on a guess.
        assert_eq!(before.commit_available(), None, "{before:?}");
        assert_eq!(before.held_mb(), Some(fp0), "held_mb must be the footprint");

        let buf = touched_mb(PROBE_MB);
        let during = snapshot().expect("snapshot during");
        std::hint::black_box(&buf);

        let grew = during.proc_rss_mb.unwrap().saturating_sub(fp0);
        drop(buf);
        // The upper bound carries the *suite*, not just the machine, exactly as
        // the Windows arm above: this test runs beside every other test in the
        // binary, and a sibling that commits memory inside the window is charged
        // to the same per-process footprint (measured: 1587 MB of reported
        // growth for this same 512 MB allocation in a full-suite run, against
        // 518 MB every time when the identical binary runs alone). Repeating the
        // window cannot isolate a sibling either — libmalloc keeps a freed
        // 512 MB region charged to the task (footprint still 517 MB after
        // `drop`), so a later window reuses it and reports its own growth
        // against the first baseline anyway. What this arm has to prove is that
        // the scalar moves with a real allocation and is in MB; the exact
        // pairing lives in the standalone run and in the peak invariants below.
        //
        // Which is why the ceiling is a unit sanity check and not a tight one.
        // 4096 still fails a probe that reports bytes, or one that has stopped
        // tracking this task at all (the floor catches both of those too).
        assert!(
            (440..=4096).contains(&grew),
            "footprint grew {grew} MB for a {PROBE_MB} MB allocation: {before:?} -> {during:?}"
        );
        // The two peak invariants that hold from any starting point. `peak` may
        // already sit above `fp0` — in the full-suite order an earlier test
        // leaves a high-water mark behind, so this allocation only climbs
        // partway back to it — and the lifetime maximum never comes down
        // (measured: current 2836 -> 3348 MB against a peak already at 3348).
        let (current_during, peak_during) = (
            during.proc_rss_mb.unwrap(),
            during.proc_peak_rss_mb.unwrap(),
        );
        assert!(
            peak_during >= peak0,
            "lifetime peak came down: {peak0} -> {peak_during}"
        );
        assert!(
            peak_during >= current_during,
            "lifetime peak below current: {during:?}"
        );
        // Only when this test starts at the high-water mark can it demand that
        // the allocation move the peak too — which is the case a standalone run
        // is in, and the one that catches a peak counter read from the wrong
        // field.
        if peak0 == fp0 {
            assert!(
                peak_during >= peak0 + grew,
                "at the high-water mark the peak must absorb the allocation: {before:?} -> {during:?}"
            );
        }
        println!("[mem] footprint {fp0} MB -> {current_during} MB (peak {peak0} -> {peak_during})");
    }

    /// Read the footprint pair while nothing moves and it is always consistent;
    /// the inversion lives entirely in the window where the ledger is climbing,
    /// which is the window the separation step actually reads. So this arm
    /// samples as fast as possible beside a feeder thread and requires the
    /// documented invariant every time — it is the test that fails if the clamp
    /// in `macos::footprint_mb` is removed. Measured on this machine with the
    /// clamp deleted: 36, 70, 76 and 93 inversions in four separate runs, each
    /// out of 200 000 paired reads (and 0 in every run with the clamp). The
    /// raw-byte inversion rate is far higher — 3.6% of 835 455 calls — because
    /// truncating both numbers to MiB hides any skew smaller than a megabyte;
    /// that gap is why this test needs a six-figure read count rather than a
    /// thousand.
    #[cfg(target_os = "macos")]
    #[test]
    fn peak_never_trails_current_while_memory_is_climbing() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let done = Arc::new(AtomicBool::new(false));
        let d = Arc::clone(&done);
        let feeder = std::thread::spawn(move || {
            // Each buffer is touched (an untouched reservation is not
            // footprint) and then kept, so the ledger only rises for the length
            // of the window; the thread dropping them releases it again.
            let mut held: Vec<Vec<u8>> = Vec::new();
            for _ in 0..6 {
                held.push(touched_mb(32));
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            drop(held);
            d.store(true, Ordering::Relaxed);
        });

        let (mut reads, mut inversions) = (0u32, 0u32);
        while !done.load(Ordering::Relaxed) && reads < 200_000 {
            let Some(s) = snapshot() else {
                break;
            };
            let (Some(current), Some(peak)) = (s.proc_rss_mb, s.proc_peak_rss_mb) else {
                panic!("macOS must report both halves of the footprint pair: {s:?}");
            };
            reads += 1;
            if peak < current {
                inversions += 1;
            }
        }
        feeder.join().expect("feeder thread");
        // Without enough samples "zero inversions" is just a quiet window.
        assert!(reads >= 2000, "only {reads} paired reads during the climb");
        assert_eq!(
            inversions, 0,
            "{inversions} of {reads} reads saw peak below current"
        );
        println!("[mem] {reads} reads during a 192 MB climb, {inversions} inversions");
    }

    // The sampler-vs-climb arm lives in `tests/peak_sampler.rs` now, not here: it
    // reads process-wide counters, and cargo runs these arms as threads in one
    // process, so a sibling arm moving memory changes what the comparisons mean.
    // Windows said so first.

    /// `log_point` sits inside window loops, so its cost has to be one
    /// formatting call and its failure mode has to be nothing at all.
    #[test]
    fn log_point_never_fails_a_run() {
        log_point(0, "test: site with no logger installed");
        log_point(usize::MAX - 1, "test: absurd sequence number");
    }

    /// The mid-run fallback trigger is a string match, so the strings it must
    /// fire on are pinned here — including the shapes it must *not* fire on,
    /// because re-running a cancelled or missing-model step on another engine
    /// is worse than failing it.
    #[test]
    fn allocation_failure_classification() {
        let hits = [
            "inference session error: [ONNXRuntimeError] : 3 : FAIL : \
                 Unexpected: BFCArena malloc failed to allocate memory of size 4194304 bytes",
            "inference session error: [ONNXRuntimeError] : 10 : EP_FAIL : bad allocation",
            // The real field report, in the shape an engine now wraps it: the
            // prefix and the appended memory sample must not bury the marker
            // the classifier works off.
            "inference session error at window 412/687: [ONNXRuntimeError : 10 : EP_FAIL : \
             bad allocation] [at failure: phys 2956/16183MB commit 0/50353MB \
             proc 3400(peak 3450)/1500(peak 1600)MB]",
            "memory allocation of 4423680 bytes failed",
            "model error: cannot allocate memory block",
            "std::bad_alloc thrown",
        ];
        for t in hits {
            assert!(is_allocation_failure(t), "should classify as OOM: {t}");
        }

        let misses = [
            "cancelled by caller",
            "model error: Mel-Band RoFormer model not found: /models/melband.onnx",
            "WAV error at original.wav: expected data block",
            "resample error: unsupported input channel count 7",
            "model output error: output tensor 'audio' has shape [1, 0, 2]",
            "connection timeout after 30s",
        ];
        for t in misses {
            assert!(!is_allocation_failure(t), "should NOT classify as OOM: {t}");
        }

        // Both signals in one message: the cancellation wins, because that is
        // the reading under which doing nothing again is the safe action.
        assert!(
            !is_allocation_failure("cancelled during BFCArena failed to allocate memory"),
            "a cancel must never be re-run"
        );
    }

    /// `Error::looks_like_allocation_failure` in `error.rs` forwards here, so
    /// the pair is one contract and is tested as one: the caller's fork is
    /// "smaller window / other engine" versus "the file on disk is broken", and
    /// getting either one wrong is worse than not having the helper.
    #[test]
    fn error_contract_separates_a_small_machine_from_a_broken_file() {
        let oom = Error::Session {
            detail: "BFCArena malloc failed to allocate memory of size 8589934592 bytes".into(),
        };
        assert!(oom.looks_like_allocation_failure(), "{oom}");

        // The two shapes of "the weights are wrong": neither is a memory
        // answer, and re-downloading each of them is the correct repair.
        for detail in [
            "invalid protobuf wire format in melband_roformer_vocals.onnx",
            "expected 214 output tensors, found 12",
        ] {
            let corrupt = Error::Model {
                detail: detail.into(),
            };
            assert!(!corrupt.looks_like_allocation_failure(), "{corrupt}");
        }

        // The gate's own refusal, on a platform with a figure and without one.
        assert!(Error::Memory {
            need_mb: 19000,
            avail_mb: Some(15000)
        }
        .looks_like_allocation_failure());
        let no_figure = Error::Memory {
            need_mb: 19000,
            avail_mb: None,
        };
        assert!(no_figure.looks_like_allocation_failure(), "{no_figure}");
        assert!(
            no_figure.to_string().contains("memory_budget_mb"),
            "{no_figure}"
        );

        assert!(!Error::Cancelled.looks_like_allocation_failure());
    }
}

#[cfg(test)]
mod marker_tests {
    use super::is_allocation_failure;

    /// The failure mode this pins is subtle: a marker that can never match looks
    /// exactly like a marker that works, because the real messages usually trip a
    /// second marker too. So this text deliberately contains `BFCArena` and
    /// nothing else from the list.
    #[test]
    fn bfcarena_alone_is_enough() {
        assert!(is_allocation_failure(
            "Ort::Run: BFCArena requested 4194304000 bytes and the system refused"
        ));
        assert!(!is_allocation_failure(
            "Ort::Run: unexpected output name 'source'"
        ));
    }
}
