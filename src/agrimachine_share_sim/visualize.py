import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# uv run agrimachine-share-sim-plot [--root results] [--out figures]
# Reads results/<theta_dist>/<label>/<algo>/<run_id>/eval_*.csv (latest run_id per scenario) and draws:
#   nonreport_rate_<theta>.png   : Pr(θ̃ = ∅ | φ = 1, θ) per θ, per group                (per theta_dist)
#   report_vs_true_<label>.png   : Pr(θ̃ | θ) (φ = 1, θ > 0), series = theta_dist         (per POSG scenario)
#   report_vs_true_0.25_{heatmap,bubble}.png : Pr(θ̃ | θ) for theta_dist = 0.25, x = θ, y = θ̃, one panel per POSG scenario
#   report_vs_true_0.25_deceiver_merged_heatmap.png : the same for Deceiver × {bid, scale}, θ̃ = 0.0 counted as ∅
#   expected_reward_<theta>.png  : mean Σw_i, Σv_i (stacked) + report cost (separate)   (per theta_dist)
#   reward_violin_<theta>.png    : distribution of Σ(w_i + v_i), Σw_i, Σv_i per group    (per theta_dist)
#   reward_violin_deceiver_<theta>.png : the same for Optimal, Deceiver × {bid, scale}      (per theta_dist)
#   fairness_violin_<theta>.png  : distribution of Π(w_i + v_i) per group               (per theta_dist)
#   summary.csv                  : the numbers behind the figures (table view)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Style (light-mode categorical slots; color follows the entity in every figure)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL, TEXT, MUTED, GRID, SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Groups on the x axis: (name, labels pooled into it, color). * × scale_to_win pools both strategy spaces
GROUPS = [
    ("Optimal", ("optimal",), BLUE),
    ("Withholder × bid", ("withholder_bid_to_win",), ORANGE),
    ("Deceiver × bid", ("deceiver_bid_to_win",), AQUA),
    ("* × scale", ("withholder_scale_to_win", "deceiver_scale_to_win"), YELLOW),
]
# Optimal vs. Deceiver under each aggregation rule (scale_to_win not pooled here)
DECEIVER_GROUPS = [
    ("Opt", ("optimal",), BLUE),
    ("Bid to Win", ("deceiver_bid_to_win",), AQUA),
    ("Scale to Win", ("deceiver_scale_to_win",), YELLOW),
]
POSG_LABELS = ("withholder_bid_to_win", "deceiver_bid_to_win", "withholder_scale_to_win", "deceiver_scale_to_win")
THETA_NAMES = {"0.25": "ample (0.25)", "0.50": "tight (0.50)", "0.75": "scarce (0.75)"}
THETA_COLORS = (BLUE, ORANGE, AQUA)
METRICS = [("sum_wv", r"$\sum_i (w_i + v_i)$", AQUA), ("sum_w", r"$\sum_i w_i$", BLUE), ("sum_v", r"$\sum_i v_i$", ORANGE)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Load
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def read_csv(path: str) -> dict:
    """{column: np.ndarray}; numeric columns become float"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {}
    for key in (rows[0] if rows else {}):
        values = [row[key] for row in rows]
        try:
            data[key] = np.asarray([float(v) if v != "" else np.nan for v in values])
        except ValueError:
            data[key] = np.asarray(values)
    return data

def load(root: str) -> dict:
    """{(theta_dist, label): {"episodes": ..., "cells": ...}} from the latest run of each scenario"""
    runs = {}
    for dirpath, _, files in os.walk(root):
        if "eval_episodes.csv" not in files:
            continue
        theta_dir, label, _algo, run_id = os.path.normpath(dirpath).split(os.sep)[-4:]
        key = (theta_dir.removeprefix("theta"), label)
        if key not in runs or run_id > runs[key][0]:
            runs[key] = (run_id, dirpath)
    return {key: {kind: read_csv(os.path.join(path, f"eval_{kind}.csv")) for kind in ("episodes", "cells")}
            for key, (_, path) in sorted(runs.items())}

def pooled(data: dict, theta: str, labels: tuple, kind: str) -> dict | None:
    parts = [data[(theta, label)][kind] for label in labels if (theta, label) in data]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

def reported_cells(cells: dict) -> np.ndarray:
    """Cells that a member could report on: φ = 1 and θ > 0 (θ = 0 forces θ̃ = 0)"""
    return (cells["phi"] == 1) & (cells["theta"] > 0)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Figures
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def plot_nonreport_rate(data: dict, theta: str, out: str, rows: list) -> None:
    groups = [(name, labels, color) for name, labels, color in GROUPS if labels != ("optimal",)]
    series = []
    for name, labels, color in groups:
        cells = pooled(data, theta, labels, "cells")
        if cells is not None:
            series.append((name, color, cells))
    if not series:
        return
    values = np.unique(np.concatenate([c["theta"][reported_cells(c)] for _, _, c in series]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.8 / len(series)
    for j, (name, color, cells) in enumerate(series):
        mask = reported_cells(cells)
        rates, counts = [], []
        for v in values:
            sel = mask & (cells["theta"] == v)
            rates.append(np.mean(cells["til_theta"][sel] == -1) if sel.any() else np.nan)
            counts.append(int(sel.sum()))
            rows.append({"figure": "nonreport_rate", "theta_dist": theta, "group": name, "theta": v,
                         "value": rates[-1], "n": counts[-1]})
        x = np.arange(len(values)) + (j - (len(series) - 1) / 2) * width
        ax.bar(x, rates, width * 0.92, color=color, label=name)
    ax.set_xticks(np.arange(len(values)), [f"{v:g}" for v in values])
    ax.set_xlabel(r"true value $\theta$")
    ax.set_ylabel(r"non-report rate  Pr($\tilde\theta = \emptyset$ | $\varphi = 1$)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Non-report rate — {THETA_NAMES.get(theta, theta)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, out, f"nonreport_rate_{theta}.png")

def plot_report_vs_true(data: dict, label: str, out: str, rows: list) -> None:
    thetas = [t for t in sorted({t for t, _ in data}) if (t, label) in data]
    if not thetas:
        return
    cells = {t: data[(t, label)]["cells"] for t in thetas}
    true_values = np.unique(np.concatenate([c["theta"][reported_cells(c)] for c in cells.values()]))
    report_values = np.unique(np.concatenate([[-1.0], true_values,
                                              *[c["til_theta"][reported_cells(c)] for c in cells.values()]]))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    offset = 0.28
    for j, t in enumerate(thetas):
        c = cells[t]
        mask = reported_cells(c)
        xs, ys, sizes = [], [], []
        for yi, v in enumerate(true_values):
            sel = mask & (c["theta"] == v)
            if not sel.any():
                continue
            for xi, r in enumerate(report_values):
                p = np.mean(c["til_theta"][sel] == r)
                rows.append({"figure": "report_vs_true", "theta_dist": t, "group": label, "theta": v,
                             "til_theta": r, "value": p, "n": int(sel.sum())})
                if p > 0:
                    xs.append(xi + (j - (len(thetas) - 1) / 2) * offset)
                    ys.append(yi)
                    sizes.append(320 * p)
        ax.scatter(xs, ys, s=sizes, color=THETA_COLORS[j % len(THETA_COLORS)], alpha=0.85,
                   edgecolors=SURFACE, linewidths=1.5, label=THETA_NAMES.get(t, t))
    ax.set_xticks(np.arange(len(report_values)), ["∅" if r == -1 else f"{r:g}" for r in report_values])
    ax.set_yticks(np.arange(len(true_values)), [f"{v:g}" for v in true_values])
    ax.set_xlim(-0.6, len(report_values) - 0.4)
    ax.set_ylim(-0.6, len(true_values) - 0.4)
    ax.grid(axis="both")
    ax.set_xlabel(r"report $\tilde\theta$")
    ax.set_ylabel(r"true value $\theta$")
    ax.set_title(f"Report vs. true value — {label.replace('_', ' ')}  (area ∝ Pr($\\tilde\\theta$ | $\\theta$))")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="theta_dist")
    for handle in legend.legend_handles:
        handle.set_sizes([60])
    _save(fig, out, f"report_vs_true_{label}.png")

def report_matrix(cells: dict, true_values: np.ndarray, report_values: np.ndarray) -> np.ndarray:
    """P[r, v] = Pr(θ̃ = report_values[r] | θ = true_values[v]) over cells with φ = 1, θ > 0 (NaN if unobserved)"""
    mask = reported_cells(cells)
    P = np.full((len(report_values), len(true_values)), np.nan)
    for v, value in enumerate(true_values):
        sel = mask & (cells["theta"] == value)
        if sel.any():
            P[:, v] = [np.mean(cells["til_theta"][sel] == r) for r in report_values]
    return P

def plot_report_vs_true_theta(data: dict, theta: str, out: str, kind: str, labels: tuple = POSG_LABELS,
                              titles: dict | None = None, merge_zero: bool = False, suffix: str = "") -> None:
    """One theta_dist, one panel per POSG scenario; x = true θ, y = report θ̃. kind = heatmap / bubble
    merge_zero: count θ̃ = 0.0 as ∅ (both are never assigned; see aggregation in core/model.py)"""
    labels = [label for label in labels if (theta, label) in data]
    if not labels:
        return
    cells = {label: data[(theta, label)]["cells"] for label in labels}
    if merge_zero:
        cells = {label: {**c, "til_theta": np.where(c["til_theta"] == 0.0, -1.0, c["til_theta"])}
                 for label, c in cells.items()}
    true_values = np.unique(np.concatenate([c["theta"][reported_cells(c)] for c in cells.values()]))
    report_values = np.unique(np.concatenate([[-1.0] if merge_zero else [-1.0, 0.0], true_values,
                                              *[c["til_theta"][reported_cells(c)] for c in cells.values()]]))
    # Heatmap: an extra narrow column holds the colorbar so that it does not cover the last panel
    ratios = [1] * len(labels) + ([0.06] if kind == "heatmap" else [])
    fig, grid = plt.subplots(1, len(ratios), figsize=(2.6 * len(labels) + 1.2, 3.6), squeeze=False,
                             gridspec_kw={"width_ratios": ratios})
    axes = grid[:, :len(labels)]
    for ax in axes[0][1:]:
        ax.sharey(axes[0][0])
        ax.tick_params(labelleft=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", [SURFACE, BLUE])
    for ax, label in zip(axes[0], labels):
        P = report_matrix(cells[label], true_values, report_values)
        if kind == "heatmap":
            image = ax.imshow(np.nan_to_num(P), cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
            for r in range(P.shape[0]):
                for v in range(P.shape[1]):
                    if P[r, v] > 0:
                        ax.text(v, r, f"{P[r, v]:.2f}", ha="center", va="center", fontsize=7,
                                color=SURFACE if P[r, v] > 0.55 else TEXT)
            ax.grid(False)
        else:
            rr, vv = np.nonzero(np.nan_to_num(P) > 0)
            ax.scatter(vv, rr, s=500 * P[rr, vv], color=BLUE, alpha=0.85, edgecolors=SURFACE, linewidths=1.5)
            ax.set_xlim(-0.6, len(true_values) - 0.4)
            ax.set_ylim(-0.6, len(report_values) - 0.4)
            ax.grid(axis="both")
        strategy, aggregation = label.split("_", 1)
        ax.set_title((titles or {}).get(label, f"{strategy.capitalize()} × {aggregation.replace('_', ' ')}"))
        ax.set_xticks(np.arange(len(true_values)), [f"{v:g}" for v in true_values])
        ax.set_xlabel(r"true value $\theta$")
    no_report = "∅ or 0" if merge_zero else "∅"
    axes[0][0].set_yticks(np.arange(len(report_values)), [no_report if r == -1 else f"{r:g}" for r in report_values])
    axes[0][0].set_ylabel(r"report $\tilde\theta$")
    note = "color" if kind == "heatmap" else "area"
    fig.suptitle(f"Report vs. true value — {THETA_NAMES.get(theta, theta)}  "
                 f"({note} = Pr($\\tilde\\theta$ | $\\theta$), $\\varphi = 1$)", color=TEXT)
    if kind == "heatmap":
        fig.colorbar(image, cax=grid[0][-1])
    _save(fig, out, f"report_vs_true_{theta}{suffix}_{kind}.png")

def plot_expected_reward(data: dict, theta: str, out: str, rows: list) -> None:
    groups = [(name, pooled(data, theta, labels, "episodes")) for name, labels, _ in GROUPS]
    groups = [(name, ep) for name, ep in groups if ep is not None]
    if not groups:
        return
    names = [name for name, _ in groups]
    mean = {k: np.array([np.mean(ep[k]) for _, ep in groups]) for k in ("sum_w", "sum_v", "sum_wv", "sum_cost")}
    for i, name in enumerate(names):
        for k in mean:
            rows.append({"figure": "expected_reward", "theta_dist": theta, "group": name, "metric": k,
                         "value": mean[k][i], "n": len(groups[i][1][k])})
    fig, (ax, ax_cost) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(names))
    ax.bar(x, mean["sum_w"], 0.6, color=BLUE, label=r"$\sum_i w_i$")
    ax.bar(x, mean["sum_v"], 0.6, color=ORANGE, label=r"$\sum_i v_i$")
    ax.scatter(x, mean["sum_wv"], s=40, marker="D", color=TEXT, zorder=3, label=r"$\sum_i (w_i + v_i)$")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [_tick(n) for n in names])
    ax.set_ylabel("mean per episode")
    ax.set_title("Reward decomposition (excl. report cost)")
    ax_cost.bar(x, mean["sum_cost"], 0.6, color=NEUTRAL)
    ax_cost.set_xticks(x, [_tick(n) for n in names])
    ax_cost.set_title(r"Report cost $\sum_i$ cost$_i$")
    fig.suptitle(THETA_NAMES.get(theta, theta), color=TEXT)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    _save(fig, out, f"expected_reward_{theta}.png")

def plot_reward_violin(data: dict, theta: str, out: str, groups: list = GROUPS, suffix: str = "",
                       ylim: tuple | None = None) -> None:
    groups = [(name, pooled(data, theta, labels, "episodes")) for name, labels, _ in groups]
    groups = [(name, ep) for name, ep in groups if ep is not None]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 0.8, 3.8))
    width = 0.8 / len(METRICS)
    for j, (key, label, color) in enumerate(METRICS):
        x = np.arange(len(groups)) + (j - (len(METRICS) - 1) / 2) * width
        _violin(ax, [ep[key] for _, ep in groups], x, width * 0.9, color)
        ax.bar([np.nan], [np.nan], color=color, label=label)   # legend entry
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(np.arange(len(groups)), [_tick(name) for name, _ in groups])
    ax.set_ylabel("value per episode")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"Ex-post reward decomposition — {THETA_NAMES.get(theta, theta)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, out, f"reward_violin{suffix}_{theta}.png")

def plot_fairness_violin(data: dict, theta: str, out: str, rows: list) -> None:
    groups = [(name, color, pooled(data, theta, labels, "episodes")) for name, labels, color in GROUPS]
    groups = [g for g in groups if g[2] is not None]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (name, color, ep) in enumerate(groups):
        _violin(ax, [ep["prod_wv"]], [i], 0.6, color)
        rows.append({"figure": "fairness", "theta_dist": theta, "group": name, "metric": "prod_wv",
                     "value": float(np.mean(ep["prod_wv"])), "n": len(ep["prod_wv"])})
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(np.arange(len(groups)), [_tick(name) for name, _, _ in groups])
    ax.set_ylabel(r"$\prod_i (w_i + v_i)$")
    ax.set_title(f"Ex-post fairness — {THETA_NAMES.get(theta, theta)}")
    _save(fig, out, f"fairness_violin_{theta}.png")

def _violin(ax, datasets: list, positions, width: float, color: str) -> None:
    datasets = [np.asarray(d, dtype=float) for d in datasets]
    # A constant sample has no density; draw it as a short line instead
    spread = [d for d in datasets if np.ptp(d) > 0]
    pos = [p for p, d in zip(positions, datasets) if np.ptp(d) > 0]
    if spread:
        parts = ax.violinplot(spread, pos, widths=width, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(SURFACE)
            body.set_alpha(0.9)
        parts["cmedians"].set_color(TEXT)
        parts["cmedians"].set_linewidth(1.2)
    for p, d in zip(positions, datasets):
        if np.ptp(d) == 0:
            ax.hlines(d[0], p - width / 2, p + width / 2, color=color, linewidth=2)

def _tick(name: str) -> str:
    return name if name.startswith("*") else name.replace(" × ", "\n× ")

def _save(fig, out: str, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(out, name)}")
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Main
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def plot_all(root: str = "results", out: str = "figures") -> None:
    data = load(root)
    if not data:
        raise FileNotFoundError(f"No eval_episodes.csv under {root}; run agrimachine-share-sim-evaluate first")
    os.makedirs(out, exist_ok=True)
    rows = []
    for theta in sorted({t for t, _ in data}):
        plot_nonreport_rate(data, theta, out, rows)
        plot_expected_reward(data, theta, out, rows)
        plot_reward_violin(data, theta, out)
        plot_reward_violin(data, theta, out, DECEIVER_GROUPS, "_deceiver", ylim=(-8, 8))
        plot_fairness_violin(data, theta, out, rows)
    for label in POSG_LABELS:
        plot_report_vs_true(data, label, out, rows)
    for kind in ("heatmap", "bubble"):
        plot_report_vs_true_theta(data, "0.25", out, kind)
    plot_report_vs_true_theta(data, "0.25", out, "heatmap", ("deceiver_bid_to_win", "deceiver_scale_to_win"),
                              {"deceiver_bid_to_win": "Bid to Win", "deceiver_scale_to_win": "Scale to Win"},
                              merge_zero=True, suffix="_deceiver_merged")

    fields = ["figure", "theta_dist", "group", "metric", "theta", "til_theta", "value", "n"]
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(prog="agrimachine-share-sim-plot")
    parser.add_argument("--root", type=str, default="results", help="Directory containing the runs")
    parser.add_argument("--out", type=str, default="figures", help="Output directory for the figures")
    args = parser.parse_args()
    plot_all(args.root, args.out)

if __name__ == "__main__":
    main()
