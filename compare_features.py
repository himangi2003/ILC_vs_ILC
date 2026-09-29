#!/usr/bin/env python3
"""
Compare features between independent patient groups A and B.

Install:
    pip install pandas numpy scipy

Example:
    python compare_groups.py \
        --features-csv tils_tsr_features.csv \
        --group-a group_A.txt \
        --group-b group_B.txt \
        --features feature_column_1 feature_column_2 \
        --output-dir comparison_results

Replace feature_column_1 and feature_column_2 with actual CSV column names.

Method:
- Aggregate each feature to its median across WSIs within each patient.
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

    # One observation per patient: median across that patient's WSIs.
    return df.groupby(subject_column)[features].median()


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
    args = parser.parse_args()

    try:
        features = list(dict.fromkeys(args.features))
        if set(features) & {args.subject_column, "sample_id", "wsi_name"}:
            raise ValueError("Identifier columns cannot be tested as features.")

        a = load_group(args.group_a, features, args.subject_column)
        b = load_group(args.group_b, features, args.subject_column)

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
        results = results.sort_values("FDR_q", na_position="last")

        paths = {
            "results": args.output_dir / "feature_comparisons.csv",
            "a": args.output_dir / "group_A_patient_features.csv",
            "b": args.output_dir / "group_B_patient_features.csv",
        }
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