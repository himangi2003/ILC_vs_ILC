#!/usr/bin/env python3
"""Collapse WSI-level rows to one row per patient.

Some patients have several slides (WSIs) of the same tumor. Each patient must
contribute one observation to a between-group test, so slide rows are
aggregated before testing. Methods:

  median  Median across a patient's slides (for two slides, equals the mean).
  mean    Mean across a patient's slides.
  auto    Sum size-like features (areas, perimeters, counts), because the
          patient's total tissue is the meaningful quantity; median for all
          other features (ratios, densities, shape, distances).
  largest Keep only the slide with the most tissue (see --size-column).
  single  Keep only patients with exactly one slide (drops multi-slide
          patients entirely); a sensitivity check.

Note for "auto": ratio features are medians of per-slide ratios, not ratios
of summed areas, so they are not recomputed from the summed columns.

Used as a module by compare_groups.py and compare_groups_sensitivity.py, or
on its own to write a patient-level CSV:

  python slide_aggregation.py \\
      --input data/Lobular_master_with_tumor_morphology.csv \\
      --output Lobular_tumor_morphology_patients.csv \\
      --method auto --keep-columns OS_event OS_days

Without --features, every numeric column whose name starts with wsi_, WSI_,
n_, total_ or mean_ is aggregated. --keep-columns carries patient-level
columns (e.g. survival) through unchanged; they must agree across a
patient's slides.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

METHODS = ("median", "mean", "auto", "largest", "single")

# Columns that measure how much tissue was on the slide, in preference order.
SIZE_CANDIDATES = (
    "wsi_cluster_area_mm2",
    "total_cluster_area_mm2",
    "total_master_area_mm2",
)

# Size-like (extensive) features: absolute areas, perimeters and counts.
EXTENSIVE = re.compile(
    r"area(_\w+)?_(um2|mm2)$|perimeter_(um|mm)$|(^|_)n_[a-z]\w*$",
    re.IGNORECASE,
)

FEATURE_PREFIXES = ("wsi_", "WSI_", "n_", "total_", "mean_")
SAMPLE_ID = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}"

AXIS_LABELS = {
    "median": "Patient median across WSIs",
    "mean": "Patient mean across WSIs",
    "auto": "Patient value (sum or median across WSIs)",
    "largest": "Value on patient's largest WSI",
    "single": "Single-WSI patients only",
}


def extensive_features(features):
    return [feature for feature in features if EXTENSIVE.search(feature)]


def resolve_size_column(df, size_column=None):
    if size_column is None:
        size_column = next((c for c in SIZE_CANDIDATES if c in df), None)
        if size_column is None:
            raise ValueError(
                "No tissue-size column found for --aggregate largest; "
                f"tried {list(SIZE_CANDIDATES)}. Pass --size-column."
            )
    if size_column not in df:
        raise ValueError(f"Size column {size_column!r} not found.")
    return size_column


def aggregate_slides(df, features, subject_column, method, size_column=None):
    """Return a DataFrame indexed by subject_column with one row per patient."""
    if method not in METHODS:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    grouped = df.groupby(subject_column)

    if method == "median":
        return grouped[features].median()

    if method == "mean":
        return grouped[features].mean()

    if method == "auto":
        result = grouped[features].median()
        summed = extensive_features(features)
        if summed:
            # min_count=1 keeps all-missing patients missing rather than 0.
            result[summed] = grouped[summed].sum(min_count=1)
        return result

    if method == "single":
        counts = grouped.size()
        kept = df.loc[df[subject_column].isin(counts.index[counts.eq(1)])]
        return kept.set_index(subject_column)[features].sort_index()

    size_column = resolve_size_column(df, size_column)
    size = pd.to_numeric(df[size_column], errors="coerce")
    if size.isna().any():
        raise ValueError(f"{size_column!r} has missing or nonnumeric values.")
    chosen = size.groupby(df[subject_column]).idxmax()
    return df.loc[chosen.to_numpy()].set_index(subject_column)[features].sort_index()


def count_multi_slide(df, subject_column):
    counts = df.groupby(subject_column).size()
    return int(counts.gt(1).sum())


def load_slides(path, subject_column):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if df.empty:
        raise ValueError(f"{path}: no data rows")

    if subject_column not in df:
        if "sample_id" not in df:
            raise ValueError(f"{path}: requires {subject_column} or sample_id")
        samples = df["sample_id"].astype("string").str.strip().str.upper()
        if not samples.str.fullmatch(SAMPLE_ID, na=False).all():
            raise ValueError(f"{path}: invalid TCGA sample IDs")
        df[subject_column] = samples.str.rsplit("-", n=1).str[0]

    df[subject_column] = df[subject_column].astype("string").str.strip().str.upper()
    if df[subject_column].isna().any() or df[subject_column].eq("").any():
        raise ValueError(f"{path}: missing subject IDs")

    if "wsi_name" in df:
        named = df.loc[df["wsi_name"].notna()]
        if named.duplicated([subject_column, "wsi_name"]).any():
            raise ValueError(f"{path}: duplicate subject/WSI rows")
    return df


def numeric_features(df, features):
    for feature in features:
        original = df[feature]
        numeric = pd.to_numeric(original, errors="coerce")
        nonblank = original.notna() & original.astype(str).str.strip().ne("")
        if (nonblank & numeric.isna()).any():
            raise ValueError(f"{feature!r} contains nonnumeric values")
        df[feature] = numeric
    return df


def default_features(df):
    """Numeric feature columns, identified by name prefix."""
    return [
        column for column in df
        if column.startswith(FEATURE_PREFIXES)
        and column != "wsi_name"
        and pd.to_numeric(df[column], errors="coerce").notna().any()
        and not (
            df[column].notna()
            & pd.to_numeric(df[column], errors="coerce").isna()
        ).any()
    ]


def patient_columns(df, columns, subject_column):
    """Carry patient-level columns through; they must agree across slides."""
    grouped = df.groupby(subject_column)
    conflicting = [c for c in columns if grouped[c].nunique(dropna=False).gt(1).any()]
    if conflicting:
        raise ValueError(f"--keep-columns differ across a patient's slides: {conflicting}")
    return grouped[columns].first()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path, help="Slide-level CSV")
    parser.add_argument("--output", required=True, type=Path, help="Patient-level CSV")
    parser.add_argument("--method", choices=METHODS, default="median")
    parser.add_argument("--features", nargs="+", help="Columns to aggregate (default: auto-detected)")
    parser.add_argument("--keep-columns", nargs="+", default=[],
                        help="Patient-level columns to copy through, e.g. OS_event OS_days")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--size-column", help="Tissue-size column for --method largest")
    args = parser.parse_args()

    if args.output.resolve() == args.input.resolve():
        parser.error("Output must not overwrite the input file.")

    try:
        df = load_slides(args.input, args.subject_column)
        features = list(dict.fromkeys(args.features or default_features(df)))
        if not features:
            raise ValueError("No feature columns found; pass --features.")
        missing = set(features + args.keep_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {sorted(missing)}")
        if set(features) & set(args.keep_columns):
            raise ValueError("A column cannot be both a feature and a kept column.")
        df = numeric_features(df, features)

        patients = aggregate_slides(
            df, features, args.subject_column, args.method, args.size_column
        )
        slide_counts = df.groupby(args.subject_column).size().rename("n_wsi")
        result = patients.join(slide_counts)
        if args.keep_columns:
            result = result.join(
                patient_columns(df, args.keep_columns, args.subject_column)
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.reset_index().to_csv(args.output, index=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    print(f"Slides: {len(df)}; patients: {df[args.subject_column].nunique()}; "
          f"with more than one WSI: {count_multi_slide(df, args.subject_column)}")
    print(f"Method: {args.method}; features aggregated: {len(features)}")
    if args.method == "auto":
        print("Summed across WSIs:", extensive_features(features) or "none")
    print(f"Saved {len(result)} patient rows to {args.output}")


if __name__ == "__main__":
    main()
