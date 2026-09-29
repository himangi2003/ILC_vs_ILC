import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_FEATURES = [
    "n_clusters",
    "n_clusters_with_tumor",
    "wsi_cluster_area_mm2",
    "wsi_tumor_area_mm2",
    "wsi_tumor_fraction_of_cluster",
    "wsi_tumor_perimeter_mm",
    "wsi_tumor_boundary_density_per_mm",
    "wsi_tumor_n_islands",
    "wsi_tumor_patch_density_per_mm2",
    "wsi_area_weighted_tumor_boundary_fractal_dimension",
    "wsi_area_weighted_tumor_compactness_mean",
    "wsi_area_weighted_tumor_solidity_mean",
    "wsi_area_weighted_tumor_elongation_mean",
    "wsi_area_weighted_tumor_largest_patch_index",
    "wsi_area_weighted_tumor_island_nnd_median_um",
    "wsi_area_weighted_tumor_island_gap_median_um",
    "wsi_area_weighted_tumor_spread_frac",
]


def load_group(path, features):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "subject_id" not in df:
        if "sample_id" not in df:
            raise ValueError(f"{path}: requires subject_id or sample_id")

        samples = df["sample_id"].astype("string").str.strip().str.upper()
        if not samples.str.fullmatch(
            r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", na=False
        ).all():
            raise ValueError(f"{path}: invalid TCGA sample IDs")

        df["subject_id"] = samples.str.rsplit("-", n=1).str[0]

    df["subject_id"] = (
        df["subject_id"].astype("string").str.strip().str.upper()
    )
    if df.empty or df["subject_id"].isna().any() or df["subject_id"].eq("").any():
        raise ValueError(f"{path}: empty data or missing subject IDs")

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing features: {sorted(missing)}")

    if "wsi_name" in df:
        named = df.loc[
            df["wsi_name"].notna()
            & df["wsi_name"].astype(str).str.strip().ne("")
        ]
        if named.duplicated(["subject_id", "wsi_name"]).any():
            raise ValueError(f"{path}: duplicate subject/WSI rows")

    for feature in features:
        original = df[feature]
        numeric = pd.to_numeric(original, errors="coerce")

        nonblank = (
            original.notna()
            & original.astype(str).str.strip().ne("")
        )
        if (nonblank & numeric.isna()).any():
            raise ValueError(f"{path}: nonnumeric values in {feature}")

        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"{path}: infinite values in {feature}")

        df[feature] = numeric

    # Each patient contributes once, regardless of their number of WSIs.
    return df.groupby("subject_id")[features].median()


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


def bootstrap_mean_difference(a, b, repetitions, rng):
    """
    Resample patients independently within each group.
    Return percentile CI for mean(A) - mean(B).
    """
    estimates = np.empty(repetitions)

    # Work in batches to limit memory usage.
    for start in range(0, repetitions, 200):
        size = min(200, repetitions - start)

        sampled_a = rng.choice(a, size=(size, len(a)), replace=True)
        sampled_b = rng.choice(b, size=(size, len(b)), replace=True)

        estimates[start:start + size] = (
            sampled_a.mean(axis=1) - sampled_b.mean(axis=1)
        )

    return np.quantile(estimates, [0.025, 0.975])


def analyze_feature(feature, group_a, group_b, repetitions, rng):
    a = group_a[feature].dropna().to_numpy(dtype=float)
    b = group_b[feature].dropna().to_numpy(dtype=float)

    row = {
        "feature": feature,
        "n_A": len(a),
        "n_B": len(b),
        "missing_A": len(group_a) - len(a),
        "missing_B": len(group_b) - len(b),
        "mean_A": a.mean() if len(a) else np.nan,
        "mean_B": b.mean() if len(b) else np.nan,
        "median_A": np.median(a) if len(a) else np.nan,
        "median_B": np.median(b) if len(b) else np.nan,
        "mean_difference_A_minus_B": np.nan,
        "bootstrap_CI95_low": np.nan,
        "bootstrap_CI95_high": np.nan,
        "cliffs_delta": np.nan,
        "MWU_p": np.nan,
        "Welch_p": np.nan,
        "status": "insufficient_patients",
    }

    if len(a) < 2 or len(b) < 2:
        return row

    row["mean_difference_A_minus_B"] = a.mean() - b.mean()

    # Zero variance in both groups prevents a useful mean-based CI.
    if a.var(ddof=1) == 0 and b.var(ddof=1) == 0:
        row["status"] = "constant_within_both_groups"
        return row

    # Asymptotic calculation includes correction for ties.
    u, p = stats.mannwhitneyu(
        a, b,
        alternative="two-sided",
        method="asymptotic",
    )
    row["MWU_p"] = p

    # P(A > B) - P(A < B); ties contribute zero.
    row["cliffs_delta"] = 2 * u / (len(a) * len(b)) - 1

    row["Welch_p"] = stats.ttest_ind(a, b, equal_var=False).pvalue

    low, high = bootstrap_mean_difference(a, b, repetitions, rng)
    row["bootstrap_CI95_low"] = low
    row["bootstrap_CI95_high"] = high
    row["status"] = "ok"

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Patient-level sensitivity analysis: Lobular vs Ductal"
    )
    parser.add_argument("--group-a", required=True, type=Path)
    parser.add_argument("--group-b", required=True, type=Path)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lobular_vs_ductal_sensitivity"),
    )
    args = parser.parse_args()

    if args.bootstrap < 1000:
        parser.error("Use at least 1000 bootstrap repetitions.")

    try:
        features = list(dict.fromkeys(args.features))
        if set(features) & {"subject_id", "sample_id", "wsi_name"}:
            raise ValueError("Identifiers cannot be tested as features.")

        a = load_group(args.group_a, features)
        b = load_group(args.group_b, features)

        overlap = set(a.index) & set(b.index)
        if overlap:
            raise ValueError(
                f"Patients appear in both groups: {sorted(overlap)[:10]}"
            )

        seeds = np.random.SeedSequence(args.seed).spawn(len(features))
        results = pd.DataFrame([
            analyze_feature(
                feature, a, b, args.bootstrap, np.random.default_rng(seed)
            )
            for feature, seed in zip(features, seeds)
        ])

        # Each method is corrected across the specified feature family.
        results["MWU_FDR_q"] = bh_adjust(results["MWU_p"])
        results["Welch_FDR_q"] = bh_adjust(results["Welch_p"])

        results["MWU_FDR_significant"] = results["MWU_FDR_q"].lt(0.05)
        results["Welch_FDR_significant"] = results["Welch_FDR_q"].lt(0.05)

        results["bootstrap_CI_excludes_zero"] = (
            results["bootstrap_CI95_low"].gt(0)
            | results["bootstrap_CI95_high"].lt(0)
        )
        results["both_tests_FDR_significant"] = (
            results["MWU_FDR_significant"]
            & results["Welch_FDR_significant"]
        )

        results = results.sort_values("MWU_FDR_q", na_position="last")

        outputs = {
            "results": args.output_dir / "sensitivity_results.csv",
            "a": args.output_dir / "group_A_patient_features.csv",
            "b": args.output_dir / "group_B_patient_features.csv",
        }
        inputs = {args.group_a.resolve(), args.group_b.resolve()}
        if any(path.resolve() in inputs for path in outputs.values()):
            raise ValueError("Output files must not overwrite input files.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(outputs["results"], index=False)
        a.reset_index().to_csv(outputs["a"], index=False)
        b.reset_index().to_csv(outputs["b"], index=False)

        print(f"Patients: Lobular={len(a)}, Ductal={len(b)}")
        print(
            "Mann–Whitney FDR-significant features:",
            results["MWU_FDR_significant"].sum(),
        )
        print(
            "Significant with both methods:",
            results["both_tests_FDR_significant"].sum(),
        )
        print(f"Results saved to: {outputs['results']}")

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()