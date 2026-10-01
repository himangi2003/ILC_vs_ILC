#!/usr/bin/env python3
"""Kaplan-Meier overall survival by tumor-stroma ratio (TSR) category.

Requires km_common.py and: pip install pandas numpy scipy matplotlib lifelines

A patient is stroma-high when their aggregated WSI_tumor_stroma_ratio is
below --threshold (default 1, i.e. more than 50% stroma), else stroma-low.
Panels: all patients, Lobular (ILC), Ductal (IDC).

  python km_tsr.py --output-dir km_tsr
  python km_tsr.py --subset ER+/HER2- --output-dir km_tsr_er_her2

Outputs: km_tsr.png/.pdf and km_tsr_summary.csv.
"""
import argparse

from km_common import add_common_arguments, run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, "km_tsr")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Stroma-high when tumor-stroma ratio is below this")
    args = parser.parse_args()
    run(parser, args, "Tils_Tsr_proximity", "WSI_tumor_stroma_ratio",
        lambda values: values < args.threshold,
        reference="Stroma-low", exposed="Stroma-high", name="km_tsr",
        title=f"Overall survival by tumor-stroma ratio (stroma-high: ratio < {args.threshold:g})")


if __name__ == "__main__":
    main()
