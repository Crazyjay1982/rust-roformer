//! The memory sampler, measured in a process that is doing nothing else.
//!
//! This arm used to live in `mem`'s own module tests, and it failed there on a
//! Windows runner and nowhere else. Not because Windows is stranger: the
//! assertions compare *instants* — the sampled peak against a reading taken at
//! one moment — and `peak_mb_while` samples a **current** quantity (`PagefileUsage`
//! on Windows, `VmSize` on Linux, `phys_footprint` on macOS). `cargo test` runs
//! the library's arms as threads in one process, so a sibling arm allocating or
//! releasing tens of megabytes between two of this test's samples changes the
//! number being compared. macOS hid that because its ledger keeps freed regions
//! charged; the Windows runner did not.
//!
//! An integration test is the fix, because it is its own process: the only thing
//! that moves this process's memory during the bracket is the allocation below.
//! The alternative — sampling the OS's own lifetime counter instead — would make
//! the assertions monotone and the test trivial, and would also throw away the
//! reason the sampler exists: `SeparationReport::peak_mb` is meant to describe
//! *this bracket*, and a process-lifetime high-water describes every arm that ran
//! before it. See the module docs of `rust_roformer::mem`.
//!
//! What this arm can and cannot catch, measured rather than hoped: gutting the
//! sampler's loop entirely leaves it **green on macOS**, because
//! `phys_footprint` never falls, so the single read taken on the way out of the
//! bracket still reports the high-water. The arm therefore guards Windows and
//! Linux, the two platforms whose counters do fall — which is also the only place
//! it has ever failed. Nothing here pretends to be a portable check.

use rust_roformer::mem;

const MB: usize = 1024 * 1024;
/// The allocation under test, in MiB: enough to be unmistakable against a
/// process this small, small enough to be honest about what it measures.
const HOLD: usize = 96;
/// 20 ms is `mem`'s private sampling interval; twelve intervals is the hold.
const INTERVAL_MS: u64 = 20;

/// Touch every page, so the allocation is charged rather than merely reserved.
fn touched(mib: usize) -> Vec<u8> {
    let mut buf = vec![0u8; mib * MB];
    for slot in buf.chunks_mut(4096) {
        slot[0] = 1;
    }
    buf
}

fn held() -> Option<u64> {
    mem::snapshot().and_then(|s| s.held_mb())
}

/// A 96 MiB allocation held across a dozen sample intervals must appear in the
/// bracket's own peak, and the peak must never exceed the OS's lifetime
/// high-water — the two directions that are meaningful when nothing else in the
/// process is moving.
#[test]
fn the_sampler_sees_what_the_bracket_holds() {
    let Some(base) = held() else {
        eprintln!("[peak_sampler] [SKIP] this platform reports no process counter");
        return;
    };
    let ((len, during), peak) = mem::peak_mb_while(|| {
        let buf = touched(HOLD);
        std::thread::sleep(std::time::Duration::from_millis(
            (HOLD / 8) as u64 * INTERVAL_MS,
        ));
        let during = held().unwrap_or(0);
        std::hint::black_box(&buf);
        (buf.len(), during)
    });
    assert_eq!(len, HOLD * MB);

    // Nothing else touched this process, so a sample within one interval of the
    // inside reading is the same quantity; MiB truncation is the only slack.
    assert!(
        peak + 1 >= during,
        "sampled peak {peak} MB trailed a {HOLD} MB allocation held for \
         {during} MB of sampling intervals (base {base} MB)"
    );
    // The sampler cannot invent a peak above what the OS itself recorded, and
    // this is the comparison that stays true whatever else a platform's counters
    // decide to do.
    let os_peak = mem::snapshot().and_then(|s| s.proc_peak_commit_mb.or(s.proc_peak_rss_mb));
    assert!(
        os_peak.is_none_or(|p| peak <= p),
        "sampled peak {peak} MB exceeds the OS-reported lifetime peak {os_peak:?} MB"
    );
    // A quiet process may still be served from pages it already held, so the
    // ledger not moving is a fact about the allocator, not a bug here. It is
    // worth printing: it says how much this arm can actually see.
    if during + 8 < base {
        eprintln!(
            "[peak_sampler] note: {HOLD} MB read {during} MB against a {base} MB base \
             (the allocator answered from pages already held)"
        );
    }
    println!("[peak_sampler] base {base} MB, inside {during} MB, peak {peak} MB, OS high-water {os_peak:?}");
}
