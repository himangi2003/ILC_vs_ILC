#!/usr/bin/env python3
"""Lobular vs Ductal comparison restricted to ER+/HER2- patients.

ILC is mostly ER+/HER2-, whereas IDC also includes HER2+ and triple-negative
tumors. Repeating the comparison within ER+/HER2- patients shows whether a
difference reflects histology or receptor subtype.

ER+/HER2- means "ER Status By IHC" is Positive and "HER2 IHC+ FISH (FGP)" is
Negative (identical to "ER/HER2 status - FGP II" in the TCGA clinical file).

Every module is analysed twice, on all patients and on the ER+/HER2- subset,
with the same patient-level tests as compare_groups_sensitivity.py
(Mann-Whitney, Welch, Cliff's delta, bootstrap CI; BH-FDR within module and
subset). Fisher exact tests are added for TSR stroma-high (ratio < 1) and
any necrosis (area > --necrosis-min-area).

  python subset_analysis.py --data-dir data \\
      --clinical data/brca_tcga_clinical_data.csv --output-dir er_her2_subset

Outputs: feature_results.csv, fisher_tests.csv, patient_counts.csv.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from compare_groups_sensitivity import analyze_feature, bh_adjust, load_group
from slide_aggregation import METHODS, aggregate_slides

SUBSET = "ER+/HER2-"
GROUPS = ("Lobular", "Ductal")

# module: (file stem, default aggregation, features)
MODULES = {
    "tumor_morphology": ("with_tumor_morphology", "auto", [
        "wsi_tumor_area_mm2",
        "wsi_tumor_fraction_of_cluster",
        "wsi_tumor_perimeter_mm",
        "wsi_tumor_boundary_density_per_mm",
        "wsi_tumor_n_islands",
        "wsi_tumor_patch_density_per_mm2",
        "wsi_area_weighted_tumor_boundary_fractal_dimension",
        "wsi_area_weighted_tumor_island_nnd_median_um",
        "wsi_area_weighted_tumor_island_gap_median_um",
    ]),
    "immune_proximity": ("immune_proximity", "auto", [
        "wsi_til_area_total_mm2",
        "wsi_til_fraction_intratumoral",
        "wsi_til_contact_fraction",
        "wsi_til_pct_within_50um",
        "wsi_til_pct_within_100um",
        "wsi_til_pct_within_200um",
        "wsi_til_extratumoral_distance_aw_median_um",
    ]),
    "tils_tsr": ("Tils_Tsr_proximity", "median", [
        "WSI_tumor_stroma_ratio",
        "WSI_sTIL_pct",
        "WSI_neighborhood_intratumoral_TIL_pct",
        "WSI_inter_tumor_sTILs_pct",
    ]),
    "necrosis": ("necrosis_proximity", "median", [
        "wsi_necrosis_area_um2",
        "wsi_necrosis_perimeter_um",
        "wsi_necrosis_fraction_of_cluster",
    ]),
}


def receptor_status(path, patient_column, er_column, her2_column):
    """Return a Series of 'ER+/HER2-', 'other' or 'unknown' indexed by patient."""
    clinical = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clinical.columns = clinical.columns.str.strip()
    missing = {patient_column, er_column, her2_column} - set(clinical.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    er = clinical[er_column].str.strip()
    her2 = clinical[her2_column].str.strip()
    status = pd.Series("unknown", index=clinical.index)
    status[er.isin(["Positive", "Negative"]) & her2.isin(["Positive", "Negative"])] = "other"
    status[er.eq("Positive") & her2.eq("Negative")] = SUBSET

    result = pd.DataFrame({
        "subject_id": clinical[patient_column].str.strip().str.upper(),
        "status": status,
    }).drop_duplicates()
    if result["subject_id"].duplicated().any():
        raise ValueError("Conflicting ER/HER2 status across samples of one patient.")
    return result.set_index("subject_id")["status"]


def fisher_test(test, subset, a_yes, a_no, b_yes, b_no):
    """Fisher exact test with a Woolf 95% CI for the odds ratio (Lobular vs Ductal)."""
    odds, p = stats.fisher_exact([[a_yes, a_no], [b_yes, b_no]])
    low = high = np.nan
    if min(a_yes, a_no, b_yes, b_no) > 0:
        se = math.sqrt(1 / a_yes + 1 / a_no + 1 / b_yes + 1 / b_no)
        low = math.exp(math.log(odds) - 1.96 * se)
        high = math.exp(math.log(odds) + 1.96 * se)
    return {
        "test": test, "subset": subset,
        "Lobular_yes": a_yes, "Lobular_n": a_yes + a_no,
        "Lobular_pct": 100 * a_yes / (a_yes + a_no) if a_yes + a_no else np.nan,
        "Ductal_yes": b_yes, "Ductal_n": b_yes + b_no,
        "Ductal_pct": 100 * b_yes / (b_yes + b_no) if b_yes + b_no else np.nan,
        "odds_ratio": odds, "OR_CI95_low": low, "OR_CI95_high": high, "p_value": p,
    }


def binary_counts(a, b, condition):
    a, b = condition(a.dropna()), condition(b.dropna())
    return int(a.sum()), int((~a).sum()), int(b.sum()), int((~b).sum())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clinical", type=Path, default=Path("data/brca_tcga_clinical_data.csv"))
    parser.add_argument("--patient-column", default="Patient ID")
    parser.add_argument("--er-column", default="ER Status By IHC")
    parser.add_argument("--her2-column", default="HER2 IHC+ FISH (FGP)")
    parser.add_argument("--aggregate", choices=METHODS,
                        help="Override the per-module aggregation (morphology/immune: auto; "
                             "TILs/necrosis: median). 'largest' fails for necrosis.")
    parser.add_argument("--necrosis-min-area", type=float, default=0.0,
                        help="Necrosis area (um2) above which a patient counts as having necrosis")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("er_her2_subset"))
    args = parser.parse_args()

    if args.bootstrap < 1000:
        parser.error("Use at least 1000 bootstrap repetitions.")

    try:
        status = receptor_status(
            args.clinical, args.patient_column, args.er_column, args.her2_column
        )
        subsets = {"all": None, SUBSET: set(status.index[status.eq(SUBSET)])}

        feature_rows, fisher_rows, counts = [], [], []
        stream = 0
        for module, (stem, default_method, features) in MODULES.items():
            method = args.aggregate or default_method
            slides = [
                load_group(args.data_dir / f"{group}_master_{stem}.csv", features)
                for group in GROUPS
            ]

            for subset, keep in subsets.items():
                a, b = (
                    aggregate_slides(
                        df if keep is None else df.loc[df["subject_id"].isin(keep)],
                        features, "subject_id", method,
                    )
                    for df in slides
                )
                for group, patients in zip(GROUPS, (a, b)):
                    by_status = status.reindex(patients.index).fillna("not_in_clinical")
                    counts.append({
                        "module": module, "subset": subset, "group": group,
                        "patients": len(patients),
                        **by_status.value_counts().to_dict(),
                    })

                results = []
                for feature in features:
                    stream += 1
                    rng = np.random.default_rng([args.seed, stream])
                    results.append(analyze_feature(feature, a, b, args.bootstrap, rng))
                results = pd.DataFrame(results)
                results["MWU_FDR_q"] = bh_adjust(results["MWU_p"])
                results["Welch_FDR_q"] = bh_adjust(results["Welch_p"])
                results.insert(0, "aggregation", method)
                results.insert(0, "subset", subset)
                results.insert(0, "module", module)
                feature_rows.append(results)

                if module == "tils_tsr":
                    fisher_rows.append(fisher_test(
                        "TSR stroma-high (ratio < 1)", subset,
                        *binary_counts(a["WSI_tumor_stroma_ratio"], b["WSI_tumor_stroma_ratio"],
                                       lambda s: s < 1),
                    ))
                if module == "necrosis":
                    fisher_rows.append(fisher_test(
                        f"Any necrosis (area > {args.necrosis_min_area:g} um2)", subset,
                        *binary_counts(a["wsi_necrosis_area_um2"], b["wsi_necrosis_area_um2"],
                                       lambda s: s > args.necrosis_min_area),
                    ))

        results = pd.concat(feature_rows, ignore_index=True)
        fisher = pd.DataFrame(fisher_rows)
        counts = pd.DataFrame(counts).fillna(0)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "feature_results.csv", index=False)
        fisher.to_csv(args.output_dir / "fisher_tests.csv", index=False)
        counts.to_csv(args.output_dir / "patient_counts.csv", index=False)

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    pd.set_option("display.width", 200)
    print("Patients per subset (first module):")
    print(counts.loc[counts["module"].eq("tumor_morphology"),
                     ["subset", "group", "patients"]].to_string(index=False))

    print("\nCliff's delta (Lobular vs Ductal) and Mann-Whitney FDR q:")
    summary = results.pivot(index=["module", "feature"], columns="subset",
                            values=["cliffs_delta", "MWU_FDR_q"])
    print(summary.to_string(float_format=lambda x: f"{x:.3g}"))

    print("\nFisher exact tests:")
    print(fisher[["test", "subset", "Lobular_pct", "Ductal_pct", "odds_ratio",
                  "OR_CI95_low", "OR_CI95_high", "p_value"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
