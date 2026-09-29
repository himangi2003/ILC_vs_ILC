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


def read_ids(path):
    ids = {
        line.strip().upper()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise ValueError(f"No subject IDs found in {path}")
    return ids


def bh_adjust(p_values):
    """Benjamini–Hochberg adjustment, excluding untestable features."""
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


def compare_feature(feature, a, b, requested_a, requested_b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    result = {
        "feature": feature,
        "n_A": len(a),
        "n_B": len(b),
        "missing_A": requested_a - len(a),
        "missing_B": requested_b - len(b),
        "mean_A": a.mean() if len(a) else np.nan,
        "mean_B": b.mean() if len(b) else np.nan,
        "median_A": np.median(a) if len(a) else np.nan,
        "median_B": np.median(b) if len(b) else np.nan,
        "mean_difference_A_minus_B": np.nan,
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "p_value": np.nan,
        "status": "insufficient_patients",
    }

    if len(a) and len(b):
        result["mean_difference_A_minus_B"] = a.mean() - b.mean()

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", required=True, type=Path)
    parser.add_argument("--group-a", required=True, type=Path)
    parser.add_argument("--group-b", required=True, type=Path)
    parser.add_argument(
        "--features", required=True, nargs="+",
        help="Exact numeric feature column names to compare",
    )
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_results"))
    args = parser.parse_args()

    try:
        group_a = read_ids(args.group_a)
        group_b = read_ids(args.group_b)

        overlap = group_a & group_b
        if overlap:
            raise ValueError(
                f"Subjects appear in both groups: {sorted(overlap)[:10]}"
            )

        df = pd.read_csv(args.features_csv)
        df.columns = df.columns.str.strip()

        # Derive TCGA patient IDs if the CSV only has sample IDs.
        if args.subject_column not in df:
            if args.sample_column not in df:
                raise ValueError(
                    "CSV must contain a subject-ID or sample-ID column."
                )
            samples = df[args.sample_column].astype("string").str.strip().str.upper()
            if not samples.str.fullmatch(
                r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", na=False
            ).all():
                raise ValueError("Cannot derive subject IDs: invalid TCGA sample IDs.")
            df[args.subject_column] = samples.str.rsplit("-", n=1).str[0]

        df[args.subject_column] = (
            df[args.subject_column].astype("string").str.strip().str.upper()
        )
        if (
            df[args.subject_column].isna().any()
            or df[args.subject_column].eq("").any()
        ):
            raise ValueError("Feature CSV contains missing subject IDs.")

        features = list(dict.fromkeys(args.features))
        missing_columns = set(features) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing feature columns: {sorted(missing_columns)}")

        if args.subject_column in features:
            raise ValueError("Subject ID cannot be tested as a feature.")

        # Keep only requested subjects; each remaining row should represent a WSI.
        df = df.loc[
            df[args.subject_column].isin(group_a | group_b)
        ].copy()

        if df.empty:
            raise ValueError("No group subject IDs match the feature CSV.")

        if "wsi_name" in df:
            named = df.loc[
                df["wsi_name"].notna()
                & df["wsi_name"].astype(str).str.strip().ne("")
            ]
            if named.duplicated([args.subject_column, "wsi_name"]).any():
                raise ValueError(
                    "Duplicate subject/WSI rows found. Resolve duplicates first."
                )

        for feature in features:
            original = df[feature]
            numeric = pd.to_numeric(original, errors="coerce")
            nonblank = (
                original.notna()
                & original.astype(str).str.strip().ne("")
            )
            if (nonblank & numeric.isna()).any():
                raise ValueError(f"{feature!r} contains nonnumeric values.")
            if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
                raise ValueError(f"{feature!r} contains infinite values.")
            df[feature] = numeric

        # Each patient receives equal weight, regardless of WSI count.
        patient = df.groupby(args.subject_column)[features].median()
        a = patient.reindex(sorted(group_a))
        b = patient.reindex(sorted(group_b))

        results = pd.DataFrame([
            compare_feature(
                feature,
                a[feature].dropna(),
                b[feature].dropna(),
                len(group_a),
                len(group_b),
            )
            for feature in features
        ])

        results["FDR_q"] = bh_adjust(results["p_value"])
        results["significant_FDR_0.05"] = results["FDR_q"].lt(0.05)
        results = results.sort_values("FDR_q", na_position="last")

        # Include requested patients without data in the patient-level export.
        patient_export = pd.concat([
            a.assign(group="A"),
            b.assign(group="B"),
        ]).rename_axis("subject_id").reset_index()

        present = set(patient.index)
        missing_subjects = pd.DataFrame(
            [
                {"subject_id": sid, "group": label}
                for label, ids in [("A", group_a), ("B", group_b)]
                for sid in sorted(ids - present)
            ],
            columns=["subject_id", "group"],
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "feature_comparisons.csv", index=False)
        patient_export.to_csv(
            args.output_dir / "patient_level_features.csv", index=False
        )
        missing_subjects.to_csv(
            args.output_dir / "subjects_without_feature_rows.csv", index=False
        )

        print(f"Group A: {len(group_a & present)}/{len(group_a)} patients found")
        print(f"Group B: {len(group_b & present)}/{len(group_b)} patients found")
        print(f"Features tested: {results['p_value'].notna().sum()}")
        print(
            "Features with FDR q < 0.05:",
            results["significant_FDR_0.05"].sum(),
        )
        print(f"Results saved in: {args.output_dir}")

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()