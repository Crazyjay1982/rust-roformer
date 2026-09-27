# Paired-summary estimator for bench_pair.sh.
#
# Input: the TSV that `bench run` prints, wrapped in two "## order …" sections.
# Design: each configuration is timed once in the first position and once in the
# second. A position penalty p that adds to whatever runs second cancels out of
#
#   dd = ((B_when_second - A_when_first) + (B_when_first - A_when_second)) / 2
#
# which is what this prints, together with the position term itself and the
# largest within-configuration spread. It deliberately refuses to name a winner
# when the two orders disagree in sign or the difference is inside the scatter.
BEGIN { FS = "\t" }
/^## order A,B/ { ord = "AB"; next }
/^## order B,A/ { ord = "BA"; next }
/^#/            { next }
/^label/        { next }
NF >= 4 && $4 ~ /^[0-9.]+$/ {
    key = $1
    w[key SUBSEP ord] = $4
    seen[key] = 1
    secs[key] = $2
    wins[key] = $3
    peaks[key] = $6
}
END {
    printf "%-40s %8s %8s %16s %10s\n", "config", "secs", "windows", "wall AB / BA", "peak_MB"
    for (k in seen) {
        ab = ((k SUBSEP "AB") in w) ? w[k SUBSEP "AB"] : "MISSING"
        ba = ((k SUBSEP "BA") in w) ? w[k SUBSEP "BA"] : "MISSING"
        printf "%-40s %8s %8s %8s / %-6s %10s\n", k, secs[k], wins[k], ab, ba, peaks[k]
    }

    n = 0
    for (k in seen) lbl[++n] = k
    if (n != 2) { printf "\ncould not pair: expected 2 configs, got %d\n", n; exit 1 }
    a = lbl[1]; b = lbl[2]
    if (substr(a, 1, 1) != "A") { t = a; a = b; b = t }

    A_ab = w[a SUBSEP "AB"]; A_ba = w[a SUBSEP "BA"]
    B_ab = w[b SUBSEP "AB"]; B_ba = w[b SUBSEP "BA"]
    if (A_ab == "" || A_ba == "" || B_ab == "" || B_ba == "") {
        print "\na run is missing; refusing to estimate"; exit 1
    }

    d_ab = B_ab - A_ab
    d_ba = B_ba - A_ba
    dd = (d_ab + d_ba) / 2.0
    pos = ((A_ba - A_ab) + (B_ab - B_ba)) / 2.0
    scatter = A_ba - A_ab; if (scatter < 0) scatter = -scatter
    s2 = B_ba - B_ab;      if (s2 < 0) s2 = -s2
    if (s2 > scatter) scatter = s2

    printf "\npaired estimate (position cancelled): %s is %+.2f s vs %s (%+.1f%%)\n",
           b, dd, a, 100.0 * dd / A_ab
    printf "position effect: %+.2f s   within-config scatter: %.2f s\n", pos, scatter
    printf "raw orders: A-first %+.2f s   B-first %+.2f s\n", d_ab, d_ba

    agree = !((dd > 0 && d_ab < 0) || (dd < 0 && d_ab > 0))
    mag = dd; if (mag < 0) mag = -mag
    if (mag <= (pos < 0 ? -pos : pos))
        print "VERDICT: smaller than the position term it cancels — inconclusive, do not quote it"
    else if (!agree)
        print "VERDICT: the two orders disagree in sign — inconclusive, do not write this down"
    else if (mag < scatter)
        print "VERDICT: |difference| is within the scatter of a single config — inconclusive"
    else
        print "VERDICT: consistent across the swap — reportable, quoting both orders"
}
