#!/usr/bin/env python3
"""Kaplan-Meier overall survival by a tumor-morphology feature.

Requires km_common.py and: pip install pandas numpy scipy matplotlib lifelines

Patients are split into high / low on one feature (--feature, default
wsi_tumor_boundary_density_per_mm). Without --cutoff the split is the median
of the patients analysed (all, or the ER+/HER2- subset, both histologies
pooled), so the same cut-off applies in every panel. Panels: all patients,
Lobular (ILC), Ductal (IDC).

Size features (area, perimeter, island count) grow with the amount of tissue
on the slide; prefer the ratio and shape features, or use --aggregate auto
to sum size features across a patient's slides.

  python km_morphology.py --output-dir km_morph_boundary
  python km_morphology.py --feature wsi_tumor_fraction_of_cluster --output-dir km_morph_fraction
  python km_morphology.py --feature wsi_tumor_area_mm2 --aggregate auto --output-dir km_morph_area

Outputs: km_morphology.png/.pdf and km_morphology_summary.csv (in --output-dir).
"""
import argparse

from km_common import add_common_arguments, run

FEATURES = {
    "wsi_tumor_boundary_density_per_mm": "tumor boundary density",
    "wsi_tumor_fraction_of_cluster": "tumor fraction",
    "wsi_tumor_patch_density_per_mm2": "tumor patch density",
    "wsi_area_weighted_tumor_boundary_fractal_dimension": "boundary fractal dimension",
    "wsi_area_weighted_tumor_island_nnd_median_um": "island nearest-neighbour distance",
    "wsi_area_weighted_tumor_island_gap_median_um": "island gap",
    "wsi_tumor_area_mm2": "tumor area",
    "wsi_tumor_perimeter_mm": "tumor perimeter",
    "wsi_tumor_n_islands": "number of tumor islands",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, "km_morphology")
    parser.add_argument("--feature", choices=FEATURES,
                        default="wsi_tumor_boundary_density_per_mm")
    parser.add_argument("--cutoff", type=float,
                        help="High when the feature is at or above this (default: cohort median)")
    args = parser.parse_args()

    chosen = {}

    def classify(values):
        chosen["cutoff"] = args.cutoff if args.cutoff is not None else values.median()
        return values >= chosen["cutoff"]

    split = f">= {args.cutoff:g}" if args.cutoff is not None else ">= cohort median"
    run(parser, args, "with_tumor_morphology", args.feature, classify,
        reference="Low", exposed="High", name="km_morphology",
        title=f"Overall survival by {FEATURES[args.feature]} (high: {split})")
    print(f"Cut-off used for {args.feature}: {chosen['cutoff']:.4g}")


if __name__ == "__main__":
    main()
