"""analyze.py - every number in the paper comes from here.

Inputs (results/):  timings_lpmoff.csv (run 1), timings_run2.csv (run 2),
                 manifest.jsonl, schedules.csv, ../analysis/variant_class.csv
Run 3 measured the feedback (fb) and explicit-request (hwx) answers with their own
baselines. Run 2 re-measured histogram (all variants, -fno-vectorize fix) and the new
Claude variants, each with its own serial/expert baselines. So: histogram and
claude-web rows come from run 2, everything else from run 1, and every variant
is compared only with baselines measured in the SAME run.
Outputs: analysis/out/*.csv and a printed summary.
"""
import fnmatch, json, os, re
import numpy as np, pandas as pd
from scipy import stats

D = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
BASE = {"serial", "expert", "calib"}
rng = np.random.default_rng(0)


def load():
    r1 = pd.read_csv(f"{D}/timings_lpmoff.csv"); r1["run"] = 1
    r2 = pd.read_csv(f"{D}/timings_run2.csv"); r2["run"] = 2
    take2 = (r2.kernel == "histogram") | r2.variant.str.startswith("claude-web")
    llm2 = take2 & ~r2.variant.isin(BASE)
    keep1 = (r1.kernel != "histogram") & ~r1.variant.str.startswith("claude-web")
    r3 = pd.read_csv(f"{D}/timings_run3.csv"); r3["run"] = 3
    llm3 = r3.variant.str.contains("__fb__|__hwx__")
    r4 = pd.read_csv(f"{D}/timings_run4.csv"); r4["run"] = 4      # DeepSeek API answers
    llm4 = r4.variant.str.startswith("deepseek-api")
    llm = pd.concat([r1[keep1 & ~r1.variant.isin(BASE)], r2[llm2], r3[llm3], r4[llm4]])
    base = pd.concat([r1[r1.variant.isin(BASE)], r2[r2.variant.isin(BASE)], r3[r3.variant.isin(BASE)],
                      r4[r4.variant.isin(BASE)]])
    for df in (llm, base):          # NaN schedule would be dropped by groupby
        df["schedule"] = df["schedule"].fillna("")
    return llm, base


def medians(df):
    ok = df[df.status == "ok"]
    return ok.groupby(["run", "kernel", "variant", "schedule", "threads"]).time_s.median()


def classify(kernel, variant, F):
    """Work class from analysis/variant_class.csv (manual code reading);
    W0 = only parallelised, W1 = work-preserving rewrite, W1c = rewrite the
    compiler can collapse, W2 = repeated/algorithmic work removed in source."""
    # round-1 rules (variant_class.csv) cover the plain/hw answers only; feedback and
    # explicit-request answers are new code and are classified separately
    name = "variant_class_fb.csv" if ("__fb__" in variant or "__hwx__" in variant) else "variant_class.csv"
    rules = pd.read_csv(os.path.join(os.path.dirname(__file__), name))
    for r in rules.itertuples():
        if r.kernel == kernel and fnmatch.fnmatch(variant, r.variant_glob):
            if r.cls == "W1-if-F>1.15":
                return "W1" if F > 1.15 else "W0"
            return r.cls
    return "W0" if F <= 1.15 else "UNCLASSIFIED"


def boot_gmean(x, n=10000):
    x = np.log(np.asarray(x, float))
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.exp(x.mean()), np.exp(np.percentile(b, 2.5)), np.exp(np.percentile(b, 97.5))


def main():
    llm, base = load()
    man = [json.loads(l) for l in open(f"{D}/../llm_raw/manifest.jsonl")]
    sch = pd.read_csv(f"{D}/schedules.csv").fillna("")
    mb, ml = medians(base), medians(llm)
    ser = mb.xs("serial", level="variant").groupby(["run", "kernel"]).median()
    ex = mb.xs("expert", level="variant")
    best = ex.groupby(["run", "kernel", "threads"]).min()
    static = ex.xs("static", level="schedule")

    # ---- one row per LLM answer -------------------------------------------
    tag = {"gemma-4-31b-it": "gemma", "gemini-3.8-flash": "flash",
           "chatgpt:web-plus-default-think-off": "chatgpt",
           "deepseek:web-default-deepthink-off-search-off": "deepseek",
           "claude:web-opus-5.5-default-effort-incognito": "claude",
           "deepseek-flash": "dsapi"}
    vtag = {"gemma": "gemma-4-31b-it", "flash": "gemini-3_8-flash", "chatgpt": "chatgpt-web",
            "deepseek": "deepseek-web", "claude": "claude-web", "dsapi": "deepseek-api"}
    rows = []
    for m in man:
        if m.get("error") or m["model"] not in tag:
            continue
        mod = tag[m["model"]]
        v = f"{vtag[mod]}__{m['prompt']}__s{m['sample']}"
        k = m["kernel"]
        r = {"model": mod, "style": m["prompt"], "kernel": k, "variant": v, "sample": m["sample"]}
        sub = llm[(llm.kernel == k) & (llm.variant == v)]
        if not m.get("has_code"):
            r["outcome"] = "no_code"
        elif sub.empty:
            r["outcome"] = "not_measured"
        elif (sub.status == "compile_error").any():
            r["outcome"] = "compile_error"
        elif (sub.status != "ok").any():
            r["outcome"] = "crash/timeout"
        elif not (sub.correct.astype(str) == "True").all():
            r["outcome"] = "wrong"
        else:
            r["outcome"] = "correct"
            run = int(sub.run.iloc[0])
            t = sub.groupby("threads").time_s.median()
            s = ser.loc[(run, k)]
            r.update(T1=t.get(1.0), T4=t.get(4.0), T10=t.get(10.0), run=run,
                     F=s / t.get(1.0), S10=s / t.get(10.0), S_self=t.get(1.0) / t.get(10.0),
                     vs_best=best.loc[(run, k, 10.0)] / t.get(10.0),
                     vs_static=static.loc[(run, k, 10.0)] / t.get(10.0),
                     e_payoff=t.get(4.0) / t.get(10.0))
            r["cls"] = classify(k, v, r["F"])
        srow = sch[(sch.kernel == k) & (sch.variant == v)]
        s_txt = srow.schedules.iloc[0] if len(srow) else ""
        r["sched"] = ("adaptive" if re.search(r"dynamic|guided|auto", s_txt)
                      else "static" if "static" in s_txt else "default(static)")
        r["hwquery"] = bool(len(srow) and srow.hwquery.iloc[0])
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(f"{OUT}/answers.csv", index=False)

    # ---- T1 outcomes -------------------------------------------------------
    t1 = A.pivot_table(index=["model", "style"], columns="outcome", values="kernel",
                       aggfunc="count", fill_value=0)
    t1["n"] = t1.sum(1); t1["pct_correct"] = (100 * t1.get("correct", 0) / t1.n).round(1)
    t1.to_csv(f"{OUT}/T1_outcomes.csv"); print("\nT1 outcomes\n", t1)

    # ---- T2 performance of correct answers (W2 excluded) -------------------
    C = A[(A.outcome == "correct") & (A.cls != "W2")]
    out = []
    for (m, st), g in C.groupby(["model", "style"]):
        gm, lo, hi = boot_gmean(g.vs_best)
        gs, slo, shi = boot_gmean(g.S_self)
        out.append(dict(model=m, style=st, n=len(g), vs_best=gm, lo=lo, hi=hi,
                        S_self=gs, S_lo=slo, S_hi=shi,
                        within5=(g.vs_best >= 0.95).mean(), below90=(g.vs_best < 0.90).mean(),
                        adaptive=(g.sched == "adaptive").mean()))
    t2 = pd.DataFrame(out).round(3); t2.to_csv(f"{OUT}/T2_perf.csv", index=False)
    print("\nT2 vs best expert at 10 threads (geo-mean, 95% bootstrap CI), W2 excluded\n", t2)

    # ---- T3 expert static penalty per kernel (run 1, histogram from run 2) --
    t3 = []
    for k in sorted(set(base.kernel)):
        run = 2 if k == "histogram" else 1
        e = base[(base.run == run) & (base.kernel == k) & (base.variant == "expert") &
                 (base.threads == 10) & (base.status == "ok")]
        st = e[e.schedule == "static"].time_s.values
        bs = e.groupby("schedule").time_s.median().idxmin()
        bt = e[e.schedule == bs].time_s.values
        if len(st) == 0 or bs == "static":
            t3.append(dict(kernel=k, best=bs, penalty_pct=0.0)); continue
        bb = [np.median(rng.choice(st, len(st))) / np.median(rng.choice(bt, len(bt))) for _ in range(5000)]
        p = np.median(st) / np.median(bt)
        t3.append(dict(kernel=k, best=bs, penalty_pct=100 * (p - 1),
                       lo=100 * (np.percentile(bb, 2.5) - 1), hi=100 * (np.percentile(bb, 97.5) - 1)))
    t3 = pd.DataFrame(t3).round(1); t3.to_csv(f"{OUT}/T3_static_penalty.csv", index=False)
    print("\nT3 static vs best schedule at 10 threads (% slower, 95% CI)\n", t3)

    # ---- T4 plain vs hw, paired by (model, kernel, sample) -----------------
    out = []
    for m, g in C.groupby("model"):
        # compare ONLY plain vs hw; other styles (fb, hwx) must not shrink the pairing
        p = g[g["style"].isin(["plain", "hw"])].pivot_table(index=["kernel", "sample"], columns="style",
                                                           values="vs_best")
        if not {"plain", "hw"} <= set(p.columns):
            continue
        p = p[["plain", "hw"]].dropna()
        if len(p) >= 5:
            w = stats.wilcoxon(np.log(p.hw), np.log(p.plain))
            out.append(dict(model=m, pairs=len(p), gm_ratio_hw_over_plain=np.exp((np.log(p.hw) - np.log(p.plain)).mean()),
                            wilcoxon_p=w.pvalue))
    ad = A[A.outcome.isin(["correct", "wrong"])].groupby(["model", "style"]).sched.apply(lambda s: (s == "adaptive").mean())
    t4 = pd.DataFrame(out).round(4); t4.to_csv(f"{OUT}/T4_prompt.csv", index=False)
    print("\nT4 hw vs plain prompt (paired Wilcoxon on log vs_best)\n", t4)
    print("\nshare of answers with an adaptive schedule\n", ad.round(2).unstack())
    print("\nanswers that query the hardware/QoS/binding:", int(A.hwquery.sum()), "of", len(A))

    # ---- T5 E-core payoff --------------------------------------------------
    t5 = C.groupby(["model", "style"]).e_payoff.agg(["median", lambda s: (s < 1).mean()]).round(3)
    t5.columns = ["median_T4_over_T10", "share_10thr_slower_than_4"]
    t5.to_csv(f"{OUT}/T5_epayoff.csv"); print("\nT5 E-core payoff\n", t5)
    write_numbers(A, llm, base)
    print("\nunclassified fast answers (need reading):\n",
          A[A.cls == "UNCLASSIFIED"][["kernel", "variant", "F", "S_self"]].round(2))


def write_numbers(A, llm, base):
    """numbers.tex: the ONLY source of numbers quoted in the paper text."""
    runs = sum(len(pd.read_csv(f"{D}/{f}")) for f in ("timings_lpmoff.csv", "timings_run2.csv"))
    n = {"NumAnswers": len(A), "NumTimedRuns": f"{runs:,}".replace(",", "{,}"),
         "NumPowerOffset": "0.49", "NumHwQueries": int(A.hwquery.sum())}
    p = os.path.join(os.path.dirname(__file__), "..", "paper", "numbers.tex")
    with open(p, "w") as f:
        f.write("% generated by analysis/analyze.py - do not edit by hand\n")
        for k, v in n.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("wrote", p, n)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Extended numbers for the Results text (appended; called from __main__ below)
def extended_numbers():
    A = pd.read_csv(f"{OUT}/answers.csv")
    n = {}
    orig = A[A["style"].isin(["plain", "hw"])]
    n["NumOrigAnswers"] = len(orig); n["NumFbAnswers"] = int((A["style"] == "fb").sum())
    n["NumHwxAnswers"] = int((A["style"] == "hwx").sum()); n["NumAllAnswers"] = len(A)
    runs = sum(len(pd.read_csv(f"{D}/{f}")) for f in ("timings_lpmoff.csv", "timings_run2.csv", "timings_run3.csv", "timings_oversub.csv"))
    n["NumTimedRunsAll"] = f"{runs:,}".replace(",", "{,}")
    for m in ["gemma", "flash", "chatgpt", "deepseek", "claude", "dsapi"]:
        g = orig[orig.model == m]; tag = m.capitalize()
        n[f"NumCorrect{tag}"] = f"{100 * (g.outcome == 'correct').mean():.0f}"
        for st in ["plain", "hw"]:
            s = g[(g["style"] == st) & g.outcome.isin(["correct", "wrong"])]
            n[f"NumAdapt{tag}{st.capitalize()}"] = f"{100 * (s.sched == 'adaptive').mean():.0f}"
    n["NumHwQueriesAll"] = int(A.hwquery.sum())
    t4 = pd.read_csv(f"{OUT}/T4_prompt.csv").set_index("model")
    for m in t4.index:
        n[f"NumP{m.capitalize()}"] = f"{t4.loc[m, 'wilcoxon_p']:.4f}"
    t3 = pd.read_csv(f"{OUT}/T3_static_penalty.csv").set_index("kernel")
    # kernel cost classes, as documented in each kernel's header comment
    UNIFORM = ["matmul", "nbody", "conv2d", "norm", "kmeans", "histogram"]          # compute-bound, equal-cost iterations
    IRREG = ["trisum", "mandelbrot", "spmv", "primes", "collatz", "cutoff"]
    MILD = ["spmv", "primes", "collatz", "cutoff"]
    uni = t3.loc[UNIFORM, "penalty_pct"]; mild = t3.loc[MILD, "penalty_pct"]
    n["NumStaticUniMin"] = f"{uni.min():.0f}"; n["NumStaticUniMax"] = f"{uni.max():.0f}"
    n["NumStaticMildMin"] = f"{mild.min():.0f}"; n["NumStaticMildMax"] = f"{mild.max():.0f}"
    for k in ["histogram", "kmeans", "norm"]:
        n[f"NumStatic{k.capitalize()}"] = f"{t3.loc[k, 'penalty_pct']:.0f}"
    pl = orig[orig.outcome.isin(["correct", "wrong"]) & (orig["style"] == "plain")]
    n["NumAdaptIrregular"] = f"{100 * (pl[pl.kernel.isin(IRREG)].sched == 'adaptive').mean():.0f}"
    n["NumAdaptUniform"] = f"{100 * (pl[pl.kernel.isin(UNIFORM)].sched == 'adaptive').mean():.0f}"
    n["NumStaticMandel"] = f"{t3.loc['mandelbrot', 'penalty_pct']:.0f}"; n["NumStaticTrisum"] = f"{t3.loc['trisum', 'penalty_pct']:.0f}"
    C = A[A.outcome == "correct"]
    for st in ["plain", "hw", "fb", "hwx"]:
        g = C[C["style"] == st]
        n[f"NumShortcut{st.capitalize()}"] = f"{100 * (g.cls == 'W2').mean():.0f}"
        n[f"NumShortcutN{st.capitalize()}"] = f"{(g.cls == 'W2').sum()} of {len(g)}"
    # feedback pairing
    P = A[A["style"] == "plain"].set_index(["model", "kernel", "sample"])
    F = A[A["style"] == "fb"].set_index(["model", "kernel", "sample"])
    j = P.join(F, lsuffix="_p", rsuffix="_f", how="inner")
    both = j[(j.outcome_p == "correct") & (j.outcome_f == "correct")]
    n["NumFbPairs"] = len(j); n["NumFbBoth"] = len(both)
    stat = both[both.sched_p != "adaptive"]
    n["NumFbStaticTotal"] = len(stat); n["NumFbStaticToAdapt"] = int((stat.sched_f == "adaptive").sum())
    pp = both[(both.cls_p != "W2") & (both.cls_f != "W2")]
    r = np.log(pp.S_self_f / pp.S_self_p); r3 = np.log(pp.vs_best_f / pp.vs_best_p)
    n["NumFbScaleRatio"] = f"{np.exp(r.mean()):.2f}"; n["NumFbScaleP"] = f"{stats.wilcoxon(r).pvalue:.2f}"
    n["NumFbBestRatio"] = f"{np.exp(r3.mean()):.2f}"; n["NumFbBestP"] = f"{stats.wilcoxon(r3).pvalue:.2f}"
    n["NumFbBroken"] = int(((j.outcome_p == "correct") & (j.outcome_f != "correct")).sum())
    n["NumFbFixed"] = int(((j.outcome_p != "correct") & (j.outcome_f == "correct")).sum())
    # reproducibility of serial times across nights
    r1 = pd.read_csv(f"{D}/timings_lpmoff.csv"); r2 = pd.read_csv(f"{D}/timings_run2.csv")
    s1 = r1[(r1.status == "ok") & (r1.variant == "serial")].groupby("kernel").time_s.median()
    s2 = r2[(r2.status == "ok") & (r2.variant == "serial")].groupby("kernel").time_s.median()
    dev = (s2 / s1 - 1).abs()
    n["NumReproWithinOne"] = int((dev <= 0.01).sum()); n["NumReproMaxDev"] = f"{100 * dev.max():.0f}"
    # outcome counts for the original (plain/hw) answers, and the W0 speed range
    oc = orig.outcome.value_counts()
    for k, name in [("compile_error", "Compile"), ("crash/timeout", "Crash"), ("wrong", "Wrong"), ("no_code", "NoCode")]:
        n[f"NumOrig{name}"] = int(oc.get(k, 0))
    w0 = A[(A.outcome == "correct") & (A.cls == "W0") & A["style"].isin(["plain", "hw"])]
    gms = w0.groupby(["model", "style"]).vs_best.apply(lambda s: np.exp(np.log(s).mean()) if len(s) >= 3 else np.nan).dropna()
    n["NumWzeroMin"] = f"{gms.min():.2f}"; n["NumWzeroMax"] = f"{gms.max():.2f}"
    ad = orig[orig.outcome.isin(["correct", "wrong"]) & (orig["style"] == "plain") & orig.model.isin(["gemma", "chatgpt", "deepseek", "claude"])]
    sh = ad.groupby("model").sched.apply(lambda s: 100 * (s == "adaptive").mean())
    n["NumAdaptPlainMin"] = f"{sh.min():.0f}"; n["NumAdaptPlainMax"] = f"{sh.max():.0f}"
    # shortcut growth under feedback, Low Power Mode effect, collection dates
    sp = float(n["NumShortcutPlain"]); sf = float(n["NumShortcutFb"])
    n["NumShortcutRatio"] = f"{sf / sp:.0f}" if sp else "n/a"
    def ser(f):
        d = pd.read_csv(f"{D}/{f}"); return d[d["mode"] == "default"].groupby("kernel").time_s.median()
    lpm = ser("pe_speed_lpmon.csv") / ser("pe_speed_lpmoff.csv")
    n["NumLpmMin"] = f"{lpm.min():.1f}"; n["NumLpmMax"] = f"{lpm.max():.1f}"
    import datetime as dt
    ts = [json.loads(l).get("time") for l in open(f"{D}/../llm_raw/manifest.jsonl")]; ts = [t for t in ts if t]
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    n["NumCollectFrom"] = dt.datetime.fromtimestamp(min(ts), ist).strftime("%-d"); n["NumCollectTo"] = dt.datetime.fromtimestamp(max(ts), ist).strftime("%-d %b %Y")
    # RQ3 schedule switching, paired by (kernel, sample): exact McNemar (binomial) test
    B = A[A.outcome.isin(["correct", "wrong"]) & A["style"].isin(["plain", "hw"])]
    for m in ["gemma", "claude", "chatgpt", "deepseek", "dsapi"]:
        g = B[B.model == m].pivot_table(index=["kernel", "sample"], columns="style", values="sched", aggfunc="first")
        g = g[["plain", "hw"]].dropna()
        up = int(((g.plain != "adaptive") & (g.hw == "adaptive")).sum()); dn = int(((g.plain == "adaptive") & (g.hw != "adaptive")).sum())
        tag = m.capitalize(); n[f"NumSwitchUp{tag}"] = up; n[f"NumSwitchDown{tag}"] = dn
        n[f"NumSwitchP{tag}"] = ("n/a" if up + dn == 0 else
                                 (r"\textless{}0.0001" if stats.binomtest(up, up + dn, 0.5).pvalue < 1e-4 else f"{stats.binomtest(up, up + dn, 0.5).pvalue:.3f}"))
    p = os.path.join(os.path.dirname(__file__), "..", "paper", "numbers.tex")
    with open(p, "a") as f:
        f.write("% --- extended numbers ---\n")
        for k, v in n.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("extended numbers:", n)


if __name__ == "__main__":
    extended_numbers()


def kernel_table():
    """paper/kernels_table.tex: kernel, cost class, input arguments, serial time (run 1 median)."""
    KDESC = {"saxpy": ("unif., mem.-bound", "y=ax+y, repeated"), "matmul": ("uniform", "dense matrix product"),
             "jacobi2d": ("unif., mem.-bound", "5-point stencil sweeps"), "trisum": ("irregular", "triangular row sums"),
             "mandelbrot": ("irregular", "escape-time iteration"), "spmv": ("irregular", "sparse mat.-vec., skewed rows"),
             "nbody": ("uniform", "all-pairs forces"), "histogram": ("uniform", "256-bin histogram, repeated"),
             "norm": ("uniform", "sum of sqrt$|x|$, repeated"), "prefix": ("loop-carried", "inclusive prefix sum"),
             "primes": ("irregular", "prime count, trial division"), "collatz": ("irregular", "Collatz step counts"),
             "conv2d": ("uniform", "2-D convolution, 15$\\times$15"), "kmeans": ("uniform", "Lloyd's k-means"),
             "cutoff": ("irregular", "neighbours within radius")}
    sizes = json.load(open(f"{D}/../sizes.json"))
    r1 = pd.read_csv(f"{D}/timings_lpmoff.csv"); r2 = pd.read_csv(f"{D}/timings_run2.csv")
    ser = r1[(r1.status == "ok") & (r1.variant == "serial")].groupby("kernel").time_s.median()
    ser["histogram"] = r2[(r2.status == "ok") & (r2.variant == "serial") & (r2.kernel == "histogram")].time_s.median()
    order = ["saxpy", "matmul", "jacobi2d", "trisum", "mandelbrot", "spmv", "nbody", "histogram", "norm",
             "prefix", "primes", "collatz", "conv2d", "kmeans", "cutoff"]
    lines = ["% generated by analysis/analyze.py - do not edit by hand",
             "\\begin{table}[t]\\centering\\caption{Kernels (serial time: median of five runs).}\\label{tab:kernels}",
             "\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{@{}lllr@{}}\\toprule", "Kernel & Computation & Cost & Serial (s) \\\\\\midrule"]
    for k in order:
        cls, what = KDESC[k]
        lines.append(f"{k} & {what} & {cls} & {ser[k]:.2f} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(os.path.dirname(__file__), "..", "paper", "kernels_table.tex"), "w").write("\n".join(lines) + "\n")
    print("kernel table written")


if __name__ == "__main__":
    kernel_table()
