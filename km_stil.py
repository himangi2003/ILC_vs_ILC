#!/usr/bin/env python3
"""Kaplan-Meier overall survival by stromal TILs (sTIL %).

Requires km_common.py and: pip install pandas numpy scipy matplotlib lifelines

A patient is sTIL-high when their aggregated WSI_sTIL_pct is at or above
--cutoff (default 10%, the common low/high split for stromal TILs).
Panels: all patients, Lobular (ILC), Ductal (IDC).

  python km_stil.py --output-dir km_stil
  python km_stil.py --cutoff 30 --output-dir km_stil_30
  python km_stil.py --subset ER+/HER2- --output-dir km_stil_er_her2

Outputs: km_stil.png/.pdf and km_stil_summary.csv.
"""
import argparse

from km_common import add_common_arguments, run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, "km_stil")
    parser.add_argument("--cutoff", type=float, default=10.0,
                        help="sTIL-high when sTIL %% is at or above this")
    args = parser.parse_args()
    run(parser, args, "Tils_Tsr_proximity", "WSI_sTIL_pct",
        lambda values: values >= args.cutoff,
        reference="sTIL-low", exposed="sTIL-high", name="km_stil",
        title=f"Overall survival by stromal TILs (sTIL-high: >= {args.cutoff:g}%)")


if __name__ == "__main__":
    main()
