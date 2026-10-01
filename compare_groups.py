#!/usr/bin/env python3
"""
Compare features between independent patient groups A and B.

Install:
    pip install pandas numpy scipy

Example:
pip install pandas numpy scipy

python compare_groups.py \
  --group-a group_A_features.csv \
  --group-b group_B_features.csv \
  --features feature_column_1 feature_column_2 \
  --output-dir comparison_results

Replace feature_column_1 and feature_column_2 with actual CSV column names.

Method:
- Aggregate each feature across WSIs within each patient (default: median;
  see --aggregate and slide_aggregation.py for alternatives).
- Compare patient-level values using Welch's two-sided t-test.
- Report mean difference (A minus B) and a Welch 95% confidence interval.
- Apply Benjamini–Hochberg FDR correction across tested features.

This is an unadjusted comparison; it does not account for confounders.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from slide_aggregation import (
    AXIS_LABELS, METHODS, aggregate_slides, count_multi_slide,
    extensive_features,
)


def load_group(path, features, subject_column):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if subject_column not in df:
        if "sample_id" not in df:
            raise ValueError(f"{path}: requires subject_id or sample_id")

        samples = df["sample_id"].astype("string").str.strip().str.upper()
        valid = samples.str.fullmatch(
            r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", na=False
        )
        if not valid.all():
            raise ValueError(f"{path}: invalid TCGA sample IDs")

        df[subject_column] = samples.str.rsplit("-", n=1).str[0]

    df[subject_column] = (
        df[subject_column].astype("string").str.strip().str.upper()
    )
    if (
        df[subject_column].isna().any()
        or df[subject_column].eq("").any()
    ):
        raise ValueError(f"{path}: missing subject IDs")

    if df.empty:
        raise ValueError(f"{path}: no data rows")

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing features: {sorted(missing)}")

    if "wsi_name" in df:
        named = df.loc[
            df["wsi_name"].notna()
            & df["wsi_name"].astype(str).str.strip().ne("")
        ]
        if named.duplicated([subject_column, "wsi_name"]).any():
            raise ValueError(f"{path}: duplicate subject/WSI rows")

    for feature in features:
        original = df[feature]
        numeric = pd.to_numeric(original, errors="coerce")

        nonblank = (
            original.notna()
            & original.astype(str).str.strip().ne("")
        )
        if (nonblank & numeric.isna()).any():
            raise ValueError(f"{path}: {feature!r} contains nonnumeric values")

        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"{path}: {feature!r} contains infinite values")

        df[feature] = numeric

    # Slide-level rows; aggregated to one row per patient in main().
    return df


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))

    if len(valid):
        order = valid[np.argsort(p[valid])]
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(
            1.0, np.minimum.accumulate(ranked[::-1])[::-1]
        )

    return adjusted


def compare_feature(feature, group_a, group_b):
    a = group_a[feature].dropna().to_numpy(dtype=float)
    b = group_b[feature].dropna().to_numpy(dtype=float)

    result = {
        "feature": feature,
        "n_A": len(a),
        "n_B": len(b),
        "missing_A": len(group_a) - len(a),
        "missing_B": len(group_b) - len(b),
        "mean_A": a.mean() if len(a) else np.nan,
        "mean_B": b.mean() if len(b) else np.nan,
        "median_A": np.median(a) if len(a) else np.nan,
        "median_B": np.median(b) if len(b) else np.nan,
        "mean_difference_A_minus_B": (
            a.mean() - b.mean() if len(a) and len(b) else np.nan
        ),
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "p_value": np.nan,
        "status": "insufficient_patients",
    }

    if len(a) < 2 or len(b) < 2:
        return result

    va = a.var(ddof=1) / len(a)
    vb = b.var(ddof=1) / len(b)
    se_squared = va + vb

    if se_squared <= 0:
        result["status"] = "zero_variance_in_both_groups"
        return result

    degrees_freedom = se_squared**2 / (
        va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)
    )

    margin = stats.t.ppf(0.975, degrees_freedom) * np.sqrt(se_squared)
    difference = result["mean_difference_A_minus_B"]

    result.update({
        "CI95_low": difference - margin,
        "CI95_high": difference + margin,
        "p_value": stats.ttest_ind(a, b, equal_var=False).pvalue,
        "status": "ok",
    })

    return result
def plot_results(
    a, b, results, output_dir,
    y_label=AXIS_LABELS["median"], title="Lobular vs Ductal",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from textwrap import fill

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#377EB8", "#E68632"]
    rng = np.random.default_rng(42)
    features = results["feature"].tolist()

    def label(feature):
        return fill(feature.replace("wsi_", "").replace("_", " "), width=38)

    # Patient-level distributions: each dot represents one patient.
    ncols = 3
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(15, 4.5 * nrows),
        squeeze=False,
    )

    indexed_results = results.set_index("feature")

    for ax, feature in zip(axes.flat, features):
        values = [
            a[feature].dropna().to_numpy(dtype=float),
            b[feature].dropna().to_numpy(dtype=float),
        ]

        for position, (group_values, color) in enumerate(
            zip(values, colors), start=1
        ):
            if not len(group_values):
                continue

            box = ax.boxplot(
                group_values,
                positions=[position],
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            box["boxes"][0].set_facecolor(color)
            box["boxes"][0].set_alpha(0.35)

            ax.scatter(
                rng.normal(position, 0.055, len(group_values)),
                group_values,
                s=15,
                alpha=0.5,
                color=color,
                edgecolors="none",
            )

        q = indexed_results.loc[feature, "FDR_q"]
        q_text = f"FDR q = {q:.3g}" if np.isfinite(q) else "Not testable"

        ax.set_title(f"{label(feature)}\n{q_text}", fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([
            f"Lobular\nn = {len(values[0])}",
            f"Ductal\nn = {len(values[1])}",
        ])
        ax.set_xlim(0.5, 2.5)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes.flat[len(features):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "patient_distributions.png", dpi=300)
    fig.savefig(output_dir / "patient_distributions.pdf")
    plt.close(fig)

    # Separate axes preserve each feature's original units.
    valid = results.loc[
        np.isfinite(results["mean_difference_A_minus_B"])
        & np.isfinite(results["CI95_low"])
        & np.isfinite(results["CI95_high"])
    ].copy()

    if valid.empty:
        print("No valid confidence intervals available for a forest plot.")
        return

    fig, axes = plt.subplots(
        len(valid), 1,
        figsize=(12, max(4, 1.05 * len(valid))),
        squeeze=False,
    )

    for ax, (_, row) in zip(axes.flat, valid.iterrows()):
        difference = row["mean_difference_A_minus_B"]
        low = row["CI95_low"]
        high = row["CI95_high"]
        significant = row["FDR_q"] < 0.05
        color = "#B2182B" if significant else "#666666"

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.errorbar(
            difference,
            0,
            xerr=[[difference - low], [high - difference]],
            fmt="o",
            color=color,
            capsize=4,
        )
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_ylabel(
            label(row["feature"]),
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )

        # Include zero and leave space around the confidence interval.
        left, right = min(low, 0), max(high, 0)
        span = right - left
        padding = 0.15 * span if span > 0 else 1
        ax.set_xlim(left - padding, right + padding)

        ax.text(
            1.02, 0.5,
            f"q = {row['FDR_q']:.3g}",
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)

    fig.suptitle(
        f"{title}\n"
        "Mean difference: Lobular − Ductal\n"
        "Welch 95% confidence intervals; red indicates FDR q < 0.05\n"
        "Each panel has its own scale and original feature units",
        fontsize=12,
    )
    axes[-1, 0].set_xlabel(
        "Negative: higher in Ductal    |    Positive: higher in Lobular"
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.92])
    fig.savefig(output_dir / "mean_differences.png", dpi=300)
    fig.savefig(output_dir / "mean_differences.pdf")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Compare patient-level features from two group CSV files."
    )
    parser.add_argument("--group-a", required=True, type=Path)
    parser.add_argument("--group-b", required=True, type=Path)
    parser.add_argument("--features", required=True, nargs="+")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("comparison_results")
    )
    parser.add_argument(
        "--aggregate", choices=METHODS, default="median",
        help="How to combine multiple WSIs per patient (default: median)",
    )
    parser.add_argument(
        "--size-column",
        help="Tissue-size column for --aggregate largest (auto-detected)",
    )
    parser.add_argument(
        "--title", default="Lobular vs Ductal",
        help='Plot title, e.g. "Immune proximity: Lobular vs Ductal"',
    )
    args = parser.parse_args()

    try:
        features = list(dict.fromkeys(args.features))
        if set(features) & {args.subject_column, "sample_id", "wsi_name"}:
            raise ValueError("Identifier columns cannot be tested as features.")

        slides_a = load_group(args.group_a, features, args.subject_column)
        slides_b = load_group(args.group_b, features, args.subject_column)
        print(
            "Patients with more than one WSI:",
            f"A={count_multi_slide(slides_a, args.subject_column)},",
            f"B={count_multi_slide(slides_b, args.subject_column)}",
        )
        if args.aggregate == "auto":
            print("Summed across WSIs:", extensive_features(features) or "none")

        a, b = (
            aggregate_slides(
                slides, features, args.subject_column,
                args.aggregate, args.size_column,
            )
            for slides in (slides_a, slides_b)
        )

        overlap = set(a.index) & set(b.index)
        if overlap:
            raise ValueError(
                f"Patients appear in both groups: {sorted(overlap)[:10]}"
            )

        results = pd.DataFrame([
            compare_feature(feature, a, b)
            for feature in features
        ])
        results["FDR_q"] = bh_adjust(results["p_value"])
        results["significant_FDR_0.05"] = results["FDR_q"].lt(0.05)
        results["aggregation"] = args.aggregate
        results = results.sort_values("FDR_q", na_position="last")

        paths = {
            "results": args.output_dir / "feature_comparisons.csv",
            "a": args.output_dir / "group_A_patient_features.csv",
            "b": args.output_dir / "group_B_patient_features.csv",
        }
        plot_results(
            a, b, results, args.output_dir,
            AXIS_LABELS[args.aggregate], args.title,
        )
        inputs = {args.group_a.resolve(), args.group_b.resolve()}
        if any(path.resolve() in inputs for path in paths.values()):
            raise ValueError("Output paths must not overwrite input files.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(paths["results"], index=False)
        a.reset_index().to_csv(paths["a"], index=False)
        b.reset_index().to_csv(paths["b"], index=False)

        print(f"Independent patients: A={len(a)}, B={len(b)}")
        print(f"Features tested: {results['p_value'].notna().sum()}")
        print(
            "Features with FDR q < 0.05:",
            results["significant_FDR_0.05"].sum(),
        )
        print(f"Results: {paths['results']}")

        

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()