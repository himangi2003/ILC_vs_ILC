#!/usr/bin/env python3
"""Kaplan-Meier overall survival by necrosis AREA (high vs low/none).

Requires km_common.py and: pip install pandas numpy scipy matplotlib lifelines

km_necrosis.py compares any necrosis vs none. This script splits on how much
necrosis there is. Because a third or more of patients have zero area, the
default cut-off is the median area among necrosis-positive patients
(--split positive-median); 'High' is at or above it and 'Low/none' is below
it, including zero. --split median uses the median of all patients, and
--cutoff sets a value in um2. --feature wsi_necrosis_fraction_of_cluster
splits on the fraction of tissue instead, which does not grow with the
amount of tissue on the slide.

  python km_necrosis_area.py --limit 120 --output-dir km_necrosis_area
  python km_necrosis_area.py --cutoff 100000 --limit 120 --output-dir km_necrosis_area_100k
  python km_necrosis_area.py --feature wsi_necrosis_fraction_of_cluster --limit 120 \\
      --output-dir km_necrosis_fraction

Outputs: km_necrosis_area.png/.pdf and km_necrosis_area_summary.csv.
"""
import argparse

from km_common import add_common_arguments, run

FEATURES = {
    "wsi_necrosis_area_um2": ("necrosis area", "um²"),
    "wsi_necrosis_fraction_of_cluster": ("necrosis fraction", ""),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, "km_necrosis_area")
    parser.add_argument("--feature", choices=FEATURES, default="wsi_necrosis_area_um2")
    parser.add_argument("--split", choices=["positive-median", "median"],
                        default="positive-median",
                        help="Default cut-off when --cutoff is not given")
    parser.add_argument("--cutoff", type=float, help="High when the value is at or above this")
    args = parser.parse_args()

    label, unit = FEATURES[args.feature]
    chosen = {}

    def classify(values):
        if args.cutoff is not None:
            chosen["cutoff"] = args.cutoff
        elif args.split == "positive-median":
            chosen["cutoff"] = values[values > 0].median()
        else:
            chosen["cutoff"] = values.median()
        # Zero is never 'High', even if the cut-off itself is zero.
        return (values >= chosen["cutoff"]) & (values > 0)

    if args.cutoff is not None:
        split = f">= {args.cutoff:g} {unit}".strip()
    else:
        split = ">= median of necrosis-positive patients" if args.split == "positive-median" \
            else ">= cohort median"
    run(parser, args, "necrosis_proximity", args.feature, classify,
        reference="Low/none", exposed="High", name="km_necrosis_area",
        title=f"Overall survival by {label} (high: {split})")
    print(f"Cut-off used for {args.feature}: {chosen['cutoff']:.4g} {unit}")


if __name__ == "__main__":
    main()
