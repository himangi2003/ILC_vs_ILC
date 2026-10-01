#!/usr/bin/env python3
"""Kaplan-Meier overall survival by an immune-proximity feature.

Requires km_common.py and: pip install pandas numpy scipy matplotlib lifelines

Patients are split into high / low on one feature (--feature, default
wsi_til_contact_fraction). Without --cutoff the split is the median of the
patients analysed (all, or the ER+/HER2- subset, both histologies pooled),
so the same cut-off applies in every panel. Panels: all patients, Lobular
(ILC), Ductal (IDC).

  python km_immune.py --output-dir km_immune_contact
  python km_immune.py --feature wsi_til_pct_within_50um --output-dir km_immune_50um
  python km_immune.py --feature wsi_til_area_total_mm2 --aggregate auto \\
      --output-dir km_immune_til_area
  python km_immune.py --cutoff 0.1 --output-dir km_immune_contact_0.1

Outputs: km_immune.png/.pdf and km_immune_summary.csv (in --output-dir).
"""
import argparse

from km_common import add_common_arguments, run

FEATURES = {
    "wsi_til_contact_fraction": "TIL-tumor contact fraction",
    "wsi_til_pct_within_50um": "% TILs within 50 um of tumor",
    "wsi_til_pct_within_100um": "% TILs within 100 um of tumor",
    "wsi_til_pct_within_200um": "% TILs within 200 um of tumor",
    "wsi_til_fraction_intratumoral": "Intratumoral TIL fraction",
    "wsi_til_area_total_mm2": "Total TIL area (mm2)",
    "wsi_til_extratumoral_distance_aw_median_um": "Extratumoral TIL distance (um)",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, "km_immune")
    parser.add_argument("--feature", choices=FEATURES, default="wsi_til_contact_fraction")
    parser.add_argument("--cutoff", type=float,
                        help="High when the feature is at or above this (default: cohort median)")
    args = parser.parse_args()

    label = FEATURES[args.feature]
    chosen = {}

    def classify(values):
        chosen["cutoff"] = args.cutoff if args.cutoff is not None else values.median()
        return values >= chosen["cutoff"]

    split = f">= {args.cutoff:g}" if args.cutoff is not None else ">= cohort median"
    run(parser, args, "immune_proximity", args.feature, classify,
        reference="Low", exposed="High", name="km_immune",
        title=f"Overall survival by {label} (high: {split})")
    print(f"Cut-off used for {args.feature}: {chosen['cutoff']:.4g}")


if __name__ == "__main__":
    main()
