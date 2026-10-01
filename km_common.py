"""Shared Kaplan-Meier plotting for km_tsr.py, km_necrosis.py and km_stil.py.

Each script supplies a feature, a rule that splits patients into two
categories, and the categories' colours; this module loads the slide-level
data, aggregates it per patient, joins survival (survival_analysis.load_survival
fills the follow-up time of censored patients) and draws three panels: all
patients, Lobular (ILC) and Ductal (IDC). Each panel has censor ticks, 95%
confidence bands, a number-at-risk table, the log-rank p-value and an
unadjusted Cox HR for the second category vs the first.

Requires: pip install pandas numpy scipy matplotlib lifelines
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D

from compare_groups_sensitivity import load_group
from slide_aggregation import aggregate_slides
from subset_analysis import GROUPS, SUBSET, receptor_status
from survival_analysis import DAYS_PER_MONTH, load_survival

# Reference category takes slot 1 (blue, solid); comparison slot 2 (orange, dashed).
REFERENCE_STYLE = ("#2a78d6", "-")
EXPOSED_STYLE = ("#eb6834", "--")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def add_common_arguments(parser, default_output):
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clinical", type=Path, default=Path("data/brca_tcga_clinical_data.csv"))
    parser.add_argument("--survival", type=Path, default=Path("data/TCGA_BRCA_survival.csv"))
    parser.add_argument("--patient-column", default="Patient ID")
    parser.add_argument("--subset", choices=["all", SUBSET], default="all")
    parser.add_argument("--aggregate", choices=["median", "mean", "auto", "largest", "single"],
                        default="median", help="How to combine a patient's WSIs")
    parser.add_argument("--max-months", type=float, default=180,
                        help="Right edge of the x-axis (follow-up is sparse later)")
    parser.add_argument("--landmark-months", type=float, default=60,
                        help="Time point for the survival estimate in the summary")
    parser.add_argument("--limit", type=float,
                        help="Follow-up limit: patients are censored at this time and the "
                             "x-axis ends there (unit: --limit-unit)")
    parser.add_argument("--limit-unit", choices=["days", "months"], default="months",
                        help="Unit of --limit; with days the plot is drawn in days")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(default_output))


def load_patients(args, stem, feature):
    """One row per patient: aggregated feature, histology, months, event."""
    keep = None
    if args.subset == SUBSET:
        status = receptor_status(args.clinical, args.patient_column,
                                 "ER Status By IHC", "HER2 IHC+ FISH (FGP)")
        keep = set(status.index[status.eq(SUBSET)])

    frames = []
    for group in GROUPS:
        slides = load_group(args.data_dir / f"{group}_master_{stem}.csv", [feature])
        if keep is not None:
            slides = slides.loc[slides["subject_id"].isin(keep)]
        patients = aggregate_slides(slides, [feature], "subject_id", args.aggregate)
        frames.append(patients.assign(histology=group))
    data = pd.concat(frames).join(load_survival(args.survival), how="inner")
    data = data.dropna(subset=[feature])

    # "time" is in the display unit: days when the limit is given in days.
    data["time"] = data["months"] * (DAYS_PER_MONTH if args.limit_unit == "days" and args.limit else 1)
    if args.limit:
        # Administrative censoring: deaths after the limit are not counted.
        beyond = data["time"] > args.limit
        data.loc[beyond, "event"] = 0
        data.loc[beyond, "time"] = args.limit
    return data


def time_axis(args):
    """(unit label, x-axis end, landmark) in the display unit."""
    if args.limit and args.limit_unit == "days":
        return "Days", args.limit, min(args.landmark_months * DAYS_PER_MONTH, args.limit)
    end = args.limit or args.max_months
    return "Months", end, min(args.landmark_months, end)


def tick_step(end, unit):
    """Tick spacing: months in 6/12/24/... steps, days in week/month/year steps."""
    steps = ((1, 3, 6, 12, 24, 36, 60, 120) if unit == "Months"
             else (7, 14, 30, 60, 90, 180, 365, 730, 1825))
    for step in steps:
        if end / step <= 12:
            return step
    return end / 10


def survival_at(kmf, months):
    """Survival estimate and 95% CI at a time point (NaN past last follow-up)."""
    if months > kmf.timeline.max():
        return np.nan, np.nan, np.nan
    estimate = kmf.survival_function_at_times(months).iloc[0]
    ci = kmf.confidence_interval_survival_function_
    before = ci.loc[ci.index <= months]
    return estimate, before.iloc[-1, 0], before.iloc[-1, 1]


def compare(frame, column, reference, exposed):
    """Log-rank p and unadjusted Cox HR for the exposed vs the reference category."""
    out = {"HR_comparison": f"{exposed} vs {reference}", "logrank_p": np.nan,
           "HR": np.nan, "HR_CI95_low": np.nan, "HR_CI95_high": np.nan}
    first = frame.loc[frame[column].eq(exposed)]
    second = frame.loc[frame[column].eq(reference)]
    if first.empty or second.empty or frame["event"].sum() == 0:
        return out
    out["logrank_p"] = logrank_test(first["time"], second["time"],
                                    first["event"], second["event"]).p_value
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxPHFitter().fit(
                frame.assign(exposed=frame[column].eq(exposed).astype(float))
                [["exposed", "time", "event"]],
                "time", "event",
            )
        row = model.summary.loc["exposed"]
        out.update({"HR": row["exp(coef)"],
                    "HR_CI95_low": row["exp(coef) lower 95%"],
                    "HR_CI95_high": row["exp(coef) upper 95%"]})
    except Exception:  # convergence failure, e.g. no events in one group
        pass
    return out


def draw_panel(ax, frame, title, max_time, landmark, column, categories, unit="Months"):
    """Plot one KM panel; categories maps label -> (colour, linestyle), reference first."""
    fitters, handles, labels, summary = [], [], [], []
    reference, exposed = list(categories)
    stats = compare(frame, column, reference, exposed)
    for category, (color, linestyle) in categories.items():
        group = frame.loc[frame[column].eq(category)]
        if group.empty:
            continue
        kmf = KaplanMeierFitter(label=category).fit(group["time"], group["event"])
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.12, show_censors=True,
            censor_styles={"marker": "|", "ms": 6, "mew": 1},
            color=color, linestyle=linestyle, linewidth=2,
        )
        fitters.append(kmf)
        events = int(group["event"].sum())
        handles.append(Line2D([], [], color=color, linestyle=linestyle, linewidth=2))
        labels.append(f"{category}  (n={len(group)}, events={events})")
        estimate, low, high = survival_at(kmf, landmark)
        at = f"OS_at_{landmark:g}_{unit.lower()}"
        summary.append({
            "panel": title, "category": category, "n": len(group), "events": events,
            f"median_OS_{unit.lower()}": kmf.median_survival_time_,
            at: estimate, f"{at}_CI95_low": low, f"{at}_CI95_high": high,
            **stats,
        })

    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, max_time + 1e-9, tick_step(max_time, unit)))
    ax.set_xlabel(f"{unit} from diagnosis", color=MUTED)
    ax.set_ylabel("Overall survival probability", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=8.5, labelcolor=TEXT)

    if np.isfinite(stats["logrank_p"]):
        text = f"Log-rank p = {stats['logrank_p']:.3g}"
        if np.isfinite(stats["HR"]):
            text += (f"\nHR {exposed.lower()} vs {reference.lower()} = {stats['HR']:.2f} "
                     f"({stats['HR_CI95_low']:.2f}-{stats['HR_CI95_high']:.2f})")
        ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=TEXT)

    if fitters:
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"], fontsize=8)
    return summary


def run(parser, args, stem, feature, classify, reference, exposed, name, title):
    """Load, split with classify(values) -> bool (True = exposed), plot and save."""
    categories = {reference: REFERENCE_STYLE, exposed: EXPOSED_STYLE}
    try:
        data = load_patients(args, stem, feature)
        data["category"] = np.where(classify(data[feature]), exposed, reference)

        panels = [
            ("All patients", data),
            ("Lobular (ILC)", data.loc[data["histology"].eq("Lobular")]),
            ("Ductal (IDC)", data.loc[data["histology"].eq("Ductal")]),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(17, 6))
        unit, end, landmark = time_axis(args)
        summary = []
        for ax, (panel_title, frame) in zip(axes, panels):
            summary += draw_panel(ax, frame, panel_title, end, landmark,
                                  "category", categories, unit)

        limit = f"; censored at {args.limit:g} {args.limit_unit}" if args.limit else ""
        fig.suptitle(args.title or f"{title} ({args.subset} patients{limit})",
                     fontsize=13, color=TEXT)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.27, wspace=0.28)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output_dir / f"{name}.{suffix}", dpi=300,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
        summary = pd.DataFrame(summary)
        summary.insert(0, "limit", f"{args.limit:g} {args.limit_unit}" if args.limit else "none")
        summary.insert(0, "subset", args.subset)
        summary.to_csv(args.output_dir / f"{name}_summary.csv", index=False)

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    pd.set_option("display.width", 200)
    print(summary.drop(columns=["subset", "limit"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nSaved {args.output_dir / (name + '.png')} and {name}_summary.csv")
