#!/usr/bin/env python3
"""Run every Kaplan-Meier script for every feature with the same settings.

Requires km_common.py, km_tsr.py, km_stil.py, km_necrosis.py, km_immune.py,
km_morphology.py and their dependencies in the same folder.

Plots (one folder each under --output-root):
  TSR category, sTIL, necrosis, 7 immune-proximity features and 9
  tumor-morphology features (median split for the last two groups).

  python km_all.py --output-root km_120_months          # default: 120 months
  python km_all.py --subset ER+/HER2- --output-root km_120_months_er_her2
  python km_all.py --limit 60 --output-root km_60_months

All summaries are also combined into <output-root>/km_all_summary.csv.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from km_immune import FEATURES as IMMUNE
from km_morphology import FEATURES as MORPHOLOGY

HERE = Path(__file__).resolve().parent
SIZE_FEATURES = {"wsi_til_area_total_mm2", "wsi_tumor_area_mm2",
                 "wsi_tumor_perimeter_mm", "wsi_tumor_n_islands"}


def jobs():
    """(output folder, script, extra arguments, summary file stem)."""
    yield "tsr", "km_tsr.py", [], "km_tsr"
    yield "stil", "km_stil.py", [], "km_stil"
    yield "necrosis", "km_necrosis.py", [], "km_necrosis"
    for script, features, stem in (("km_immune.py", IMMUNE, "km_immune"),
                                   ("km_morphology.py", MORPHOLOGY, "km_morphology")):
        for feature in features:
            # Sum size features across a patient's slides; median otherwise.
            extra = ["--feature", feature]
            if feature in SIZE_FEATURES:
                extra += ["--aggregate", "auto"]
            yield f"{stem.split('_')[1]}_{feature}", script, extra, stem


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=float, default=120,
                        help="Follow-up limit (default 120)")
    parser.add_argument("--limit-unit", choices=["days", "months"], default="months",
                        help="Unit of --limit (default months)")
    parser.add_argument("--subset", choices=["all", "ER+/HER2-"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    common = ["--limit", f"{args.limit:g}", "--limit-unit", args.limit_unit,
              "--subset", args.subset, "--data-dir", args.data_dir]
    summaries, failed = [], []
    for folder, script, extra, stem in jobs():
        output = args.output_root / folder
        print(f"--> {folder}")
        result = subprocess.run(
            [sys.executable, str(HERE / script), *common, *extra, "--output-dir", str(output)],
            capture_output=True, text=True,
        )
        if result.returncode:
            failed.append(folder)
            print(result.stderr.strip().splitlines()[-1] if result.stderr else "failed")
            continue
        summary = pd.read_csv(output / f"{stem}_summary.csv")
        summary.insert(0, "plot", folder)
        summaries.append(summary)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(args.output_root / "km_all_summary.csv", index=False)
        print(f"\nCombined summary: {args.output_root / 'km_all_summary.csv'}")
    print(f"Plots made: {len(summaries)}; failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
