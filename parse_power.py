"""parse_power.py - compress a powermetrics text log into one CSV row per sample.

powermetrics prints, per sample:
  *** Sampled system activity (Sat Nov 21 00:27:33 2026 +0530) (254.41ms elapsed) ***
  CPU <i> frequency: <MHz> MHz
  CPU <i> active residency:  <pct>% (...)
  CPU Power: <mW> mW
The header clock has only 1-second resolution, so sample end times are rebuilt by
adding up the 'elapsed' durations from the first header, and the drift of that sum
against the header clock is printed as a sanity check.
Output columns: t_end (epoch s), dt_s, cpu_mw, then for each CPU i: a<i> (active %)
and f<i> (MHz).
Usage: python3 parse_power.py results/power_lpmoff.txt results/power_lpmoff.csv
"""
import csv, re, sys
from datetime import datetime

HDR = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
FREQ = re.compile(r"^CPU (\d+) frequency: (\d+) MHz")
ACT = re.compile(r"^CPU (\d+) active residency:\s+([\d.]+)%")
PWR = re.compile(r"^CPU Power: (\d+) mW")


def main(src, dst):
    rows, cur, clock, t0, maxdrift = [], None, 0.0, None, 0.0
    ncpu = 0

    def flush():
        if cur and "cpu_mw" in cur:
            rows.append(cur)

    for line in open(src, errors="replace"):
        m = HDR.search(line)
        if m:
            flush()
            hdr = datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Y %z").timestamp()
            dt = float(m.group(2)) / 1000.0
            if t0 is None:
                t0 = hdr                      # header = time the sample was printed
                clock = hdr
            else:
                clock += dt
            maxdrift = max(maxdrift, abs(clock - hdr))
            cur = {"t_end": clock, "dt_s": dt}
            continue
        if cur is None:
            continue
        for rx, key in ((FREQ, "f"), (ACT, "a")):
            m = rx.match(line)
            if m:
                i = int(m.group(1))
                ncpu = max(ncpu, i + 1)
                cur[key + str(i)] = float(m.group(2))
                break
        else:
            m = PWR.match(line)
            if m:
                cur["cpu_mw"] = int(m.group(1))
    flush()
    cols = ["t_end", "dt_s", "cpu_mw"] + [f"{k}{i}" for i in range(ncpu) for k in "af"]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"samples={len(rows)} cpus={ncpu} span_h={(rows[-1]['t_end'] - rows[0]['t_end']) / 3600:.2f} "
          f"max_clock_drift_s={maxdrift:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
