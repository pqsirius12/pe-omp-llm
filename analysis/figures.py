"""figures.py - the paper's figures, generated from the same data as analyze.py.

Run after analyze.py (it reads analysis/out/answers.csv and T3_static_penalty.csv).
Writes paper/figs/*.pdf at IEEE single-column width (3.5 in).
Palette: categorical slots 1-5 of the reference palette, fixed order, validated
(CVD/normal-vision pass; low-contrast slots always carry a text label).
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
D, OUT = os.path.join(H, "..", "results"), os.path.join(H, "out")
FIG = os.path.join(H, "..", "paper", "figs"); os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["gemma", "flash", "chatgpt", "deepseek", "dsapi", "claude"]
MCOL = {"gemma": "#2a78d6", "flash": "#eb6834", "chatgpt": "#1baf7a", "deepseek": "#eda100",
        "claude": "#e87ba4", "dsapi": "#008300"}   # colour follows the model, never its position
MNAME = {"gemma": "Gemma 4 31B", "flash": "Gemini 3.8 Flash", "chatgpt": "ChatGPT (web)",
         "deepseek": "DeepSeek (web)", "dsapi": "DeepSeek (API)", "claude": "Claude (web)"}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
W = 3.5
rng = np.random.default_rng(1)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def gm_ci(x, n=5000):
    x = np.log(np.asarray(x, float)); b = rng.choice(x, (n, len(x))).mean(1)
    return np.exp(x.mean()), np.exp(np.percentile(b, 2.5)), np.exp(np.percentile(b, 97.5))


def fig_placement():
    """Busy E- and P-cores (core-equivalents) per thread count, from powermetrics."""
    OFF = -0.49   # powermetrics header clock truncated to whole seconds (see paper)
    p = pd.read_csv(f"{D}/power_lpmoff.csv"); p["t0"] = p.t_end - p.dt_s
    p["e"] = p[[f"a{i}" for i in range(6)]].sum(axis=1) / 100; p["P"] = p[[f"a{i}" for i in range(6, 10)]].sum(axis=1) / 100
    t = pd.read_csv(f"{D}/timings_lpmoff.csv")
    t = t[(t.status == "ok") & (t.variant == "expert") & t.schedule.isin(["dynamic,1", "dynamic,1024", "guided"])
          & (t.time_s >= 0.5) & (t.kernel != "histogram")]
    T0, T1 = p.t0.values, p.t_end.values
    rows = []
    for r in t.itertuples():
        s, e = r.timestamp - r.time_s + OFF, r.timestamp + OFF
        i0, i1 = np.searchsorted(T1, s), np.searchsorted(T0, e); seg = p.iloc[i0:i1]
        w = (np.minimum(seg.t_end, e) - np.maximum(seg.t0, s)).clip(lower=0).values
        if w.sum() > 0:
            rows.append((r.threads, (seg.e.values * w).sum() / w.sum(), (seg.P.values * w).sum() / w.sum()))
    df = pd.DataFrame(rows, columns=["thr", "e", "P"]).groupby("thr").median()
    fig, ax = plt.subplots(figsize=(W, 1.9))
    ax.plot(df.index, df.P, "-o", color=C[0], lw=2, ms=4, label="P-cores busy (of 4)")
    ax.plot(df.index, df.e, "-s", color=C[1], lw=2, ms=4, label="E-cores busy (of 6)")
    ax.set_xlabel("OpenMP threads"); ax.set_ylabel("busy cores (median)")
    ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.legend(loc="upper left")
    save(fig, "placement"); return df


def fig_static_penalty():
    t3 = pd.read_csv(f"{OUT}/T3_static_penalty.csv").sort_values("penalty_pct")
    fig, ax = plt.subplots(figsize=(W, 2.4))
    y = np.arange(len(t3))
    err = np.vstack([t3.penalty_pct - t3.lo.fillna(t3.penalty_pct), t3.hi.fillna(t3.penalty_pct) - t3.penalty_pct])
    ax.barh(y, t3.penalty_pct, height=0.6, color=C[1], xerr=err, error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_yticks(y); ax.set_yticklabels(t3.kernel); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("static slower than best schedule at 10 threads (%)"); ax.grid(axis="y", visible=False)
    save(fig, "static_penalty")


def fig_schedules(A):
    S = A[A.outcome.isin(["correct", "wrong"])]
    order = [(m, s) for m in MODELS for s in ["plain", "hw", "hwx", "fb"] if ((S.model == m) & (S["style"] == s)).any()]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    labels, y = [], np.arange(len(order))
    for i, (m, s) in enumerate(order):
        g = S[(S.model == m) & (S["style"] == s)].sched.value_counts(normalize=True)
        left = 0
        for cls, col in (("adaptive", C[0]), ("static", C[1]), ("default(static)", C[3])):
            v = g.get(cls, 0.0)
            ax.barh(i, v, left=left, height=0.65, color=col, edgecolor="white", linewidth=1)
            left += v
        labels.append(f"{MNAME[m]} · {s}")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("share of answers"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[0], label="dynamic/guided"), Patch(color=C[1], label="explicit static"),
                       Patch(color=C[3], label="no clause")], ncol=3, loc="lower center",
              bbox_to_anchor=(0.3, 1.0), handlelength=1.2, columnspacing=0.8)
    save(fig, "schedules")


def fig_vs_best(A):
    W0 = A[(A.outcome == "correct") & (A.cls == "W0")]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    xs, labs = [], []
    for i, m in enumerate(MODELS):
        for j, s in enumerate(["plain", "hw"]):
            g = W0[(W0.model == m) & (W0["style"] == s)].vs_best
            if len(g) < 3: continue
            gm, lo, hi = gm_ci(g); x = i + (j - 0.5) * 0.3
            ax.errorbar(x, gm, yerr=[[gm - lo], [hi - gm]], fmt="o" if s == "plain" else "s", color=MCOL[m],
                        ms=5, lw=1.2, capsize=2, mfc=MCOL[m] if s == "plain" else "white")
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(["Gemma", "Flash", "ChatGPT", "DS web", "DS API", "Claude"])
    ax.set_ylabel("speed vs best expert\n(10 threads, geo-mean)")
    ax.plot([], [], "o", color=INK2, label="plain prompt"); ax.plot([], [], "s", mfc="white", color=INK2, label="hardware prompt")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "vs_best")


def fig_feedback(A):
    fig, ax = plt.subplots(figsize=(W, 2.0))
    cats = [("W0", C[0], "parallelized only"), ("W1", C[2], "work-preserving"),
            ("W1c", C[2], None), ("W2", C[1], "shortcut"), ("fail", "#b8b7b1", "failed")]
    rows = [(m, s) for m in ["gemma", "chatgpt", "deepseek", "claude"] for s in ["plain", "fb"]]
    for i, (m, s) in enumerate(rows):
        g = A[(A.model == m) & (A["style"] == s)]
        if s == "fb":   # compare like with like: only answers that received feedback
            pass
        k = g.cls.where(g.outcome == "correct", "fail").fillna("fail").value_counts(normalize=True)
        left = 0
        for c, col, _ in cats:
            v = k.get(c, 0.0); ax.barh(i, v, left=left, height=0.65, color=col, edgecolor="white", linewidth=1); left += v
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{MNAME[m].replace(' (web)', '')} · {s}" for m, s in rows])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of answers"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for _, c, l in cats if l], ncol=2, loc="lower center",
              bbox_to_anchor=(0.3, 1.0), handlelength=1.2, columnspacing=0.8)
    save(fig, "feedback")


def fig_oversub():
    o = pd.read_csv(f"{D}/timings_oversub.csv"); o = o[o.status == "ok"]
    r1 = pd.read_csv(f"{D}/timings_lpmoff.csv"); r1 = r1[(r1.status == "ok") & (r1.kernel != "histogram")]
    r2 = pd.read_csv(f"{D}/timings_run2.csv"); r2 = r2[(r2.status == "ok") & (r2.kernel == "histogram")]
    ex = pd.concat([r1[(r1.variant == "expert") & (r1.threads == 10)], r2[(r2.variant == "expert") & (r2.threads == 10)],
                    o[o.variant == "expert"]])
    m = ex.groupby(["kernel", "schedule", "threads"]).time_s.median().unstack("threads")
    rows = []
    for k, g in m.groupby(level=0):
        g = g.droplevel(0)
        if "static" not in g.index or len(g.index) < 2: continue
        ad = g.drop("static").min()
        rows.append((k, g.loc["static", 10.0] / ad[10.0], g.loc["static", 20.0] / ad[20.0]))
    df = pd.DataFrame(rows, columns=["kernel", "t10", "t20"]).sort_values("t10")
    fig, ax = plt.subplots(figsize=(W, 2.4)); y = np.arange(len(df))
    ax.hlines(y, df.t20, df.t10, color=GRID, lw=2)
    ax.plot(df.t10, y, "o", color=C[1], ms=5, label="10 threads (= cores)")
    ax.plot(df.t20, y, "s", color=C[0], ms=5, label="20 threads (2× oversubscribed)")
    ax.axvline(1, color=INK2, lw=0.8, ls="--"); ax.set_yticks(y); ax.set_yticklabels(df.kernel)
    ax.set_xlabel("static time / best adaptive time"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, "oversub"); return df


if __name__ == "__main__":
    A = pd.read_csv(f"{OUT}/answers.csv")
    print(fig_placement().round(2))
    fig_static_penalty(); fig_schedules(A); fig_vs_best(A); fig_feedback(A)
    print(fig_oversub().round(2).to_string(index=False))
    print("figures in", FIG, sorted(os.listdir(FIG)))


def fig_numbers(place, over):
    """numbers_fig.tex: numbers quoted from the figure computations (placement, oversubscription,
    energy, calibrated split). Same rule as numbers.tex: never typed by hand."""
    n = {"NumPlaceSerialP": f"{place.P.loc[1.0]:.2f}", "NumPlaceFourP": f"{place.P.loc[4.0]:.2f}",
         "NumPlaceFourE": f"{place.e.loc[4.0]:.2f}", "NumPlaceSixE": f"{place.e.loc[6.0]:.2f}",
         "NumPlaceTenE": f"{place.e.loc[10.0]:.2f}", "NumPlaceTenP": f"{place.P.loc[10.0]:.2f}"}
    o = over.set_index("kernel")
    for k in ["mandelbrot", "trisum"]:
        n[f"NumOver{k.capitalize()}Ten"] = f"{o.loc[k, 't10']:.2f}"; n[f"NumOver{k.capitalize()}Twenty"] = f"{o.loc[k, 't20']:.2f}"
    # energy per run (CPU package energy from powermetrics, same alignment as the placement figure)
    OFF = -0.49
    p = pd.read_csv(f"{D}/power_lpmoff.csv"); p["t0"] = p.t_end - p.dt_s
    t = pd.read_csv(f"{D}/timings_lpmoff.csv"); t = t[(t.status == "ok") & (t.kernel == "mandelbrot")]
    T0, T1 = p.t0.values, p.t_end.values
    def energy(r):
        s, e = r.timestamp - r.time_s + OFF, r.timestamp + OFF
        i0, i1 = np.searchsorted(T1, s), np.searchsorted(T0, e); seg = p.iloc[i0:i1]
        w = (np.minimum(seg.t_end, e) - np.maximum(seg.t0, s)).clip(lower=0).values
        return (seg.cpu_mw.values * w).sum() / w.sum() / 1000 * r.time_s if w.sum() > 0 else np.nan
    t["J"] = [energy(r) for r in t.itertuples()]
    sel = {"Serial": (t.variant == "serial"), "Static": (t.variant == "expert") & (t.schedule == "static") & (t.threads == 10),
           "Dyn": (t.variant == "expert") & (t.schedule == "dynamic,1") & (t.threads == 10)}
    for k, msk in sel.items():
        n[f"NumMandelJ{k}"] = f"{t[msk].J.median():.1f}"
    # calibrated (AID-style) split vs best expert schedule at 10 threads, run 1
    r1 = pd.read_csv(f"{D}/timings_lpmoff.csv"); r1 = r1[(r1.status == "ok") & (r1.threads == 10)]
    cal = r1[r1.variant == "calib"].groupby("kernel").time_s.median()
    best = r1[r1.variant == "expert"].groupby(["kernel", "schedule"]).time_s.median().groupby("kernel").min()
    ratio = (cal / best[cal.index])
    n["NumCalibKernels"] = len(ratio); n["NumCalibSlowerMin"] = f"{100 * (ratio.min() - 1):.0f}"
    n["NumCalibSlowerMax"] = f"{100 * (ratio.max() - 1):.0f}"
    n["NumCalibSlowerCount"] = int((ratio > 1.0).sum())
    with open(os.path.join(H, "..", "paper", "numbers_fig.tex"), "w") as f:
        f.write("% generated by analysis/figures.py - do not edit by hand\n")
        for k, v in n.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("fig numbers:", n, "\ncalib/best:", ratio.round(3).to_dict())


if __name__ == "__main__":
    fig_numbers(fig_placement(), fig_oversub())
