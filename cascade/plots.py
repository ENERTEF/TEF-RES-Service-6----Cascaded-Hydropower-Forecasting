"""Figures for the backtest. Reads only output/results; safe to re-run alone:

    python -m cascade.plots
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# categorical slots, fixed order; ink and grid stay neutral
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e0", "#fcfcfb"
ACTUAL = "#3d3c39"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 110, "savefig.dpi": 150,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _note(fig, text):
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=8, color=INK2)


def _load(res, name):
    p = pd.read_parquet(res / f"preds_{name}.parquet")
    m = pd.read_csv(res / f"metrics_{name}.csv", index_col="model")
    s = pd.read_csv(res / f"metrics_by_step_{name}.csv")
    return p, m, s


# ------------------------------------------------------------------- figures
def data_overview(df, fig_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    q = df["inflow"].dropna()
    axes[0].plot(q.index, q, color=C[0], lw=0.8)
    axes[0].set_ylabel("m³/s")
    axes[0].set_title("Ilarionas natural inflow, daily mean (PPC)")
    a = df["asomata"].dropna()
    axes[1].plot(a.index, a, color=C[1], lw=0.5, alpha=0.45)
    axes[1].plot(a.index, a.rolling(30, min_periods=15).mean(), color=C[1], lw=1.6)
    axes[1].set_ylabel("MWh / day")
    axes[1].set_title("Asomata HPP production, daily (ENTSO-E) · thick line = 30-day mean")
    for ax in axes:
        ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), color=C[0], alpha=0.07, lw=0)
        ax.axvspan(pd.Timestamp("2025-01-01"), df.index.max(), color=C[2], alpha=0.07, lw=0)
    axes[0].text(pd.Timestamp("2023-02-01"), axes[0].get_ylim()[1] * 0.92, "Test A\n2023–24",
                 fontsize=8, color=INK2, va="top")
    axes[1].text(pd.Timestamp("2025-02-01"), axes[1].get_ylim()[1] * 0.92, "Test B\n(public data only)",
                 fontsize=8, color=INK2, va="top")
    _save(fig, fig_dir / "fig01_data_overview.png")


def series_vs_forecast(preds, model, steps, title, unit, path, window=None, note=None):
    fig, axes = plt.subplots(len(steps), 1, figsize=(11, 2.9 * len(steps)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, steps):
        f = preds[(preds["model"] == model) & (preds["step"] == h)].sort_values("date")
        if window:
            f = f[(f["date"] >= window[0]) & (f["date"] <= window[1])]
        if "q10" in f and f["q10"].notna().any():
            ax.fill_between(f["date"], f["q10"], f["q90"], color=C[0], alpha=0.18, lw=0,
                            label="80% interval")
        ax.plot(f["date"], f["y_true"], color=ACTUAL, lw=1.1, label="Actual")
        ax.plot(f["date"], f["y_pred"], color=C[0], lw=1.4, label="Forecast")
        ax.set_title(f"{title} · {h}-day-ahead", fontsize=10)
        ax.set_ylabel(unit)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    if note:
        _note(fig, note)
    _save(fig, path)


def _spread(ys, gap):
    """Nudge label positions apart so none sit closer than `gap`."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def by_horizon(by_step, models, metric, title, ylabel, path, muted=(), note=None):
    """Lines by lead time. `muted` series are drawn grey as context; every
    series gets a de-collided direct label, so no separate legend is needed."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ends, k = [], 0
    for m in models:
        s = by_step[by_step["model"] == m].sort_values("step")
        mute = m in muted
        color = MUTED if mute else C[k]
        k += 0 if mute else 1
        ax.plot(s["step"], s[metric], color=color, lw=1.3 if mute else 2.2, marker="o", ms=4,
                zorder=2 if mute else 3)
        ends.append((m, s[metric].iloc[-1], color))
    if metric == "MAE_skill":
        ax.axhline(0, color=INK2, lw=0.8)
    lo, hi = ax.get_ylim()
    ys = _spread([e[1] for e in ends], gap=(hi - lo) * 0.055)
    for (m, y0, color), y in zip(ends, ys):
        ax.annotate(m, (7, y0), xytext=(7.25, y), textcoords="data", va="center", fontsize=8.5,
                    color=INK if color != MUTED else INK2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(y - y0) > 1e-9 else None)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.8, 7.1)
    ax.set_ylim(min(lo, min(ys)) , max(hi, max(ys)))
    fig.subplots_adjust(right=0.62)
    if note:
        _note(fig, note)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bars(values, title, xlabel, path, highlight=None, fmt="{:.2f}", note=None):
    fig, ax = plt.subplots(figsize=(8.2, 0.48 * len(values) + 1.3))
    vals = values[::-1]
    colors = [C[0] if (highlight is None or k in highlight) else MUTED for k in vals.index]
    ax.barh(vals.index, vals.values, color=colors, height=0.62)
    for i, v in enumerate(vals.values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8.5, color=INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.12)
    if note:
        _note(fig, note)
    _save(fig, path)


def importance(res, fig_dir, top=15):
    imp = pd.read_csv(res / "importance_cascade.csv", index_col=0).mean(axis=1)

    def group(f):
        if f.startswith("aso_"):
            return "Asomata history"
        if f.startswith(("ilarionas", "polyfyto", "sfikia", "upstream")):
            return "Upstream units"
        if f.startswith(("inflow", "res_level")):
            return "Ilarionas inflow"
        if f.startswith(("price", "load", "res_ws")):
            return "Market"
        return "Calendar"
    groups = ["Asomata history", "Upstream units", "Ilarionas inflow", "Market", "Calendar"]
    gcol = dict(zip(groups, C))
    by_group = imp.groupby(imp.index.map(group)).sum().reindex(groups).fillna(0)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.45]})
    g = by_group[::-1]
    a0.barh(g.index, g.values * 100, color=[gcol[k] for k in g.index], height=0.62)
    for i, v in enumerate(g.values * 100):
        a0.text(v, i, f" {v:.0f}%", va="center", fontsize=9, color=INK)
    a0.set_title("By information layer")
    a0.set_xlabel("Share of split gain (%)")
    a0.margins(x=0.15)
    t = imp.sort_values(ascending=False).head(top)[::-1]
    a1.barh(t.index, t.values * 100, color=[gcol[group(f)] for f in t.index], height=0.62)
    a1.set_title(f"Top {top} features")
    a1.set_xlabel("Share of split gain (%)")
    for ax in (a0, a1):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    fig.suptitle("What the cascade model uses (Asomata, averaged over lead times)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    _note(fig, "Colour = information layer, same in both panels. Gain share measures use in the trees, not causal effect.")
    _save(fig, fig_dir / "fig07_feature_importance.png")


def example_cascade(df, p1, p2, fig_dir, n=3):
    """The service output for a few origins: inflow forecast feeding production forecast."""
    main1, main2 = "LightGBM", "LightGBM cascade: + inflow forecast"
    origins = [pd.Timestamp(o) for o in ("2023-03-12", "2023-11-26", "2024-04-07")][:n]
    fig, axes = plt.subplots(2, n, figsize=(11.5, 5.6), sharey="row")
    for j, o in enumerate(origins):
        for i, (preds, model, col, unit, hist_col) in enumerate(
                [(p1, main1, C[0], "Inflow (m³/s)", "inflow"),
                 (p2, main2, C[1], "Asomata (MWh/day)", "asomata")]):
            ax = axes[i, j]
            hist = df[hist_col].loc[o - pd.Timedelta(days=14):o + pd.Timedelta(days=7)]
            f = preds[(preds["model"] == model) & (preds["origin"] == o)].sort_values("date")
            ax.plot(hist.index, hist.values, color=ACTUAL, lw=1.1, label="Actual")
            if "q10" in f and f["q10"].notna().any():
                ax.fill_between(f["date"], f["q10"], f["q90"], color=col, alpha=0.2, lw=0, label="80% interval")
            ax.plot(f["date"], f["y_pred"], color=col, lw=2, marker="o", ms=3.5, label="Forecast")
            ax.axvline(o, color=MUTED, lw=0.8, ls="--")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.tick_params(axis="x", labelsize=8)
            if j == 0:
                ax.set_ylabel(unit)
            if i == 0:
                ax.set_title(f"Issued {o:%d %b %Y}", fontsize=10)
            if j == n - 1:
                ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Service output: 7-day cascade forecast (top: Ilarionas inflow → bottom: Asomata production)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, fig_dir / "fig09_example_cascade_forecasts.png")


def scatter(p2, model, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    for ax, h in zip(axes, (1, 7)):
        f = p2[(p2["model"] == model) & (p2["step"] == h)]
        ax.scatter(f["y_true"], f["y_pred"], s=9, color=C[0], alpha=0.45, lw=0)
        lim = max(f["y_true"].max(), f["y_pred"].max()) * 1.03
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.grid(axis="x", visible=True)
        ax.set_title(f"{h}-day-ahead", fontsize=10)
        ax.set_xlabel("Actual (MWh/day)")
    axes[0].set_ylabel("Forecast (MWh/day)")
    fig.suptitle(f"Asomata, test 2023–24 · {model}", x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, fig_dir / "fig10_production_scatter.png")


def make_all(df, res, fig_dir):
    res, fig_dir = Path(res), Path(fig_dir)
    data_overview(df, fig_dir)

    p1, m1, s1 = _load(res, "track1_inflow")
    series_vs_forecast(p1, "LightGBM", (1, 7), "Ilarionas inflow, test 2023–24", "m³/s",
                       fig_dir / "fig02_inflow_test.png",
                       note="LightGBM, honest arm: no future weather. Shaded: 80% quantile interval.")
    by_horizon(s1, ["LightGBM", "LightGBM + weather (oracle)", "Persistence", "Climatology"], "MAE",
               "Ilarionas inflow — error by lead time", "MAE (m³/s, lower is better)",
               fig_dir / "fig03_inflow_by_horizon.png", muted=("Persistence", "Climatology"),
               note="Oracle = observed weather used as a perfect forecast: an upper bound, not achievable live.")

    p2, m2, s2 = _load(res, "track2_production")
    window = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"))
    series_vs_forecast(p2, "LightGBM cascade: + inflow forecast", (1, 7), "Asomata production, Jan–Jun 2024",
                       "MWh/day", fig_dir / "fig04_production_test.png", window=window,
                       note="Cascade LightGBM. Shaded: 80% quantile interval.")
    models2 = list(m2.index)
    ens2 = [m for m in models2 if m.startswith("Ensemble")][0]
    by_horizon(s2, ["LightGBM cascade: + inflow forecast", ens2, "SARIMA", "Persistence", "Seasonal naive (week)"],
               "MAE_skill", "Asomata production — skill by lead time",
               "MAE skill vs seasonal naive", fig_dir / "fig05_production_skill_by_horizon.png",
               muted=("Persistence", "Seasonal naive (week)"), note="Skill = 1 − MAE / MAE(same weekday last week). Above 0 beats the naive forecast.")
    abl = ["LightGBM - Asomata only", "LightGBM + upstream units",
           "LightGBM cascade: + inflow forecast", "LightGBM cascade: + actual inflow (oracle)"]
    bars(m2.loc[abl, "MAE"], "Cascade ablation — what each information layer adds",
         "MAE, MWh/day (lower is better)", fig_dir / "fig06_cascade_ablation.png", fmt="{:.1f}",
         note="Test 2023–24, all lead times pooled. Oracle uses actual future inflow: an upper bound.")
    bars(m2["MAE"], "Asomata production — all models, test 2023–24", "MAE, MWh/day (lower is better)",
         fig_dir / "fig08_production_all_models.png", fmt="{:.1f}",
         highlight=["LightGBM cascade: + inflow forecast"] + [m for m in models2 if m.startswith("Ensemble")])
    importance(res, fig_dir)
    example_cascade(df, p1, p2, fig_dir)
    scatter(p2, "LightGBM cascade: + inflow forecast", fig_dir)

    p3, m3, s3 = _load(res, "track2_production_live")
    ens3 = [m for m in m3.index if m.startswith("Ensemble")][0]
    by_horizon(s3, ["LightGBM + upstream units", ens3, "SARIMA", "Persistence", "Seasonal naive (week)"],
               "MAE_skill", "Public-data-only configuration — test 2025 → 2026-09",
               "MAE skill vs seasonal naive", fig_dir / "fig11_live_skill_by_horizon.png",
               muted=("Persistence", "Seasonal naive (week)"),
               note="ENTSO-E only (no PPC inflow), trained through 2024. 2026 is the highest-production year in the record: "
                    "production averaged 2x any training year, which tree models cannot extrapolate to.")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "output"
    make_all(pd.read_parquet(out / "results" / "daily_data.parquet"), out / "results", out / "figures")
