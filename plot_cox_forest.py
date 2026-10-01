#!/usr/bin/env python3
"""Forest plots of Cox hazard ratios from survival_analysis.py.

Requires: pip install pandas numpy matplotlib

  python plot_cox_forest.py --results survival_results/cox_results.csv
  python plot_cox_forest.py --results survival_er_her2/cox_results.csv \\
      --models adjusted interaction --title "ER+/HER2- patients"

Writes, next to the results file (or into --output-dir):
  cox_forest_all_models.png/.pdf   one panel per model, rows shared
  cox_forest_<model>.png/.pdf      one model with HR (95% CI), q and events

Hazard ratios are on a log scale; the dashed line at 1 means no effect.
Filled squares: FDR q < --q-threshold; hollow: not significant. Intervals
that run past the axis limits end in an arrow.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

SIGNIFICANT = "#2a78d6"
NOT_SIGNIFICANT = "#52514e"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
TICKS = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)

MODULE_NAMES = {
    "histology": "Histology",
    "tumor_morphology": "Tumor morphology",
    "immune_proximity": "Immune proximity",
    "tils_tsr": "TILs / tumor-stroma ratio",
    "necrosis": "Necrosis",
}
MODEL_TITLES = {
    "adjusted": "All patients, adjusted for histology",
    "interaction": "Interaction: HR in ILC / HR in IDC",
    "within_Lobular": "Within ILC",
    "within_Ductal": "Within IDC",
}


def feature_label(feature, transform):
    if feature == "lobular_vs_ductal":
        return "Lobular vs ductal"
    name = feature
    for prefix in ("wsi_area_weighted_", "wsi_", "WSI_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace("_", " ")
    if transform == "binary":
        return f"{name} (yes vs no)"
    return f"{name} ({'log, ' if str(transform).startswith('log') else ''}per SD)"


def build_rows(results, models):
    """Rows top to bottom: ('header', module, label) or ('feature', (module, feature), label)."""
    present = results.loc[results["model"].isin(models)]
    order = present.drop_duplicates(["module", "feature"])
    rows = []
    for module, group in order.groupby("module", sort=False):
        rows.append(("header", module, MODULE_NAMES.get(module, module)))
        for item in group.itertuples():
            rows.append(("feature", (module, item.feature), feature_label(item.feature, item.transform)))
    return rows


def axis_limits(data, max_range):
    finite = data.loc[np.isfinite(data["HR_CI95_low"]) & np.isfinite(data["HR_CI95_high"])]
    if finite.empty:
        return 0.5, 2.0
    low = max(min(finite["HR_CI95_low"].min(), 1.0) / 1.15, 1 / max_range)
    high = min(max(finite["HR_CI95_high"].max(), 1.0) * 1.15, max_range)
    return low, high


def draw_panel(ax, data, rows, q_threshold, max_range):
    by_key = {(r.module, r.feature): r for r in data.itertuples()}
    xlim = axis_limits(data, max_range)
    n = len(rows)

    for i, (kind, key, _) in enumerate(rows):
        y = n - 1 - i
        if kind == "header":
            if i:
                ax.axhline(y + 0.5, color=GRID, linewidth=1, zorder=0)
            continue
        row = by_key.get(key)
        if row is None or not np.isfinite(row.HR):
            continue
        significant = np.isfinite(row.q) and row.q < q_threshold
        color = SIGNIFICANT if significant else NOT_SIGNIFICANT

        low, high = row.HR_CI95_low, row.HR_CI95_high
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([max(low, xlim[0]), min(high, xlim[1])], [y, y],
                    color=color, linewidth=2, solid_capstyle="round", zorder=2)
            if low < xlim[0]:
                ax.plot(xlim[0], y, marker="<", markersize=7, color=color, zorder=2)
            if high > xlim[1]:
                ax.plot(xlim[1], y, marker=">", markersize=7, color=color, zorder=2)
        ax.plot(min(max(row.HR, xlim[0]), xlim[1]), y, marker="s", markersize=8,
                markerfacecolor=color if significant else "white",
                markeredgecolor=color, markeredgewidth=2, zorder=3)

    ax.axvline(1, color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator([t for t in TICKS if xlim[0] <= t <= xlim[1]]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Hazard ratio (log scale)", color=MUTED)


def label_rows(ax, rows):
    n = len(rows)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([label for _, _, label in rows])
    for tick, (kind, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if kind == "header" else "normal")
        tick.set_color(TEXT if kind == "header" else MUTED)


def legend(fig, q_threshold):
    handles = [
        Line2D([], [], color=SIGNIFICANT, marker="s", markersize=8, linewidth=2,
               label=f"FDR q < {q_threshold:g}"),
        Line2D([], [], color=NOT_SIGNIFICANT, marker="s", markersize=8, linewidth=2,
               markerfacecolor="white", markeredgewidth=2, label="Not significant"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01), labelcolor=TEXT)


def save(fig, output_dir, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(results, models, rows, args, output_dir):
    height = 0.3 * len(rows) + 1.8
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 3, height),
                             sharey=True, squeeze=False)
    for ax, model in zip(axes.flat, models):
        draw_panel(ax, results.loc[results["model"].eq(model)], rows,
                   args.q_threshold, args.max_range)
        ax.set_title(MODEL_TITLES.get(model, model), fontsize=10, color=TEXT)
    label_rows(axes[0, 0], rows)
    fig.suptitle(args.title, fontsize=13, color=TEXT)
    legend(fig, args.q_threshold)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save(fig, output_dir, "cox_forest_all_models")


def plot_one(results, model, args, output_dir):
    data = results.loc[results["model"].eq(model)]
    rows = build_rows(data, [model])
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(rows) + 1.8))
    draw_panel(ax, data, rows, args.q_threshold, args.max_range)
    label_rows(ax, rows)

    by_key = {(r.module, r.feature): r for r in data.itertuples()}
    columns = {"HR (95% CI)": 1.04, "q": 1.42, "events / n": 1.58}
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    n = len(rows)
    for header, x in columns.items():
        ax.text(x, n - 0.3, header, transform=transform, fontsize=9,
                fontweight="bold", color=TEXT, va="bottom")
    for i, (kind, key, _) in enumerate(rows):
        row = by_key.get(key) if kind == "feature" else None
        if row is None:
            continue
        y = n - 1 - i
        if np.isfinite(row.HR):
            hr_text = f"{row.HR:.2f} ({row.HR_CI95_low:.2f}-{row.HR_CI95_high:.2f})"
            q_text = f"{row.q:.2g}" if np.isfinite(row.q) else "-"
        else:
            hr_text, q_text = str(row.status), "-"
        for text, x in zip((hr_text, q_text, f"{row.events} / {row.n}"), columns.values()):
            ax.text(x, y, text, transform=transform, fontsize=9, color=MUTED, va="center")

    ax.set_title(f"{args.title}\n{MODEL_TITLES.get(model, model)}", fontsize=11, color=TEXT)
    legend(fig, args.q_threshold)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, output_dir, f"cox_forest_{model}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", required=True, type=Path,
                        help="cox_results.csv from survival_analysis.py")
    parser.add_argument("--models", nargs="+",
                        default=["adjusted", "interaction", "within_Lobular", "within_Ductal"])
    parser.add_argument("--modules", nargs="+", help="Restrict to these modules")
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--max-range", type=float, default=10.0,
                        help="Clip the axis to 1/R..R; longer intervals end in an arrow")
    parser.add_argument("--title", default="Cox proportional hazards: overall survival")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: the folder containing --results")
    args = parser.parse_args()

    try:
        results = pd.read_csv(args.results)
        required = {"module", "feature", "transform", "model", "HR",
                    "HR_CI95_low", "HR_CI95_high", "q", "n", "events", "status"}
        missing = required - set(results.columns)
        if missing:
            raise ValueError(f"{args.results}: missing columns {sorted(missing)}")
        if args.modules:
            results = results.loc[results["module"].isin(args.modules)]
        models = [m for m in args.models if m in set(results["model"])]
        if not models:
            raise ValueError(f"None of {args.models} found in {args.results}")

        output_dir = args.output_dir or args.results.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_all(results, models, build_rows(results, models), args, output_dir)
        for model in models:
            plot_one(results, model, args, output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    print(f"Saved forest plots for {', '.join(models)} to {output_dir}")


if __name__ == "__main__":
    main()
