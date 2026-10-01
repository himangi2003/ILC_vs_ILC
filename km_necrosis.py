#!/usr/bin/env python3
"""Kaplan-Meier overall survival by tumor necrosis.

Requires km_common.py and: pip install pandas numpy scipy matplotlib lifelines

A patient has necrosis when their aggregated wsi_necrosis_area_um2 is above
--min-area (default 0: any necrosis). Raise it, e.g. --min-area 10000, to
ignore tiny detections that may be noise. Panels: all patients, Lobular
(ILC), Ductal (IDC). --aggregate largest is unavailable: the necrosis file
has no tissue-size column.

  python km_necrosis.py --output-dir km_necrosis
  python km_necrosis.py --min-area 10000 --output-dir km_necrosis_10k
  python km_necrosis.py --subset ER+/HER2- --output-dir km_necrosis_er_her2

Outputs: km_necrosis.png/.pdf and km_necrosis_summary.csv.
"""
import argparse

from km_common import add_common_arguments, run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, "km_necrosis")
    parser.add_argument("--min-area", type=float, default=0.0,
                        help="Necrosis when aggregated area (um2) is above this")
    args = parser.parse_args()
    run(parser, args, "necrosis_proximity", "wsi_necrosis_area_um2",
        lambda values: values > args.min_area,
        reference="No necrosis", exposed="Necrosis", name="km_necrosis",
        title=f"Overall survival by necrosis (area > {args.min_area:g} um²)")


if __name__ == "__main__":
    main()
