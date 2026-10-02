#!/usr/bin/env python3
"""Stage-adjusted Cox models for every feature: does stage explain the association?

Requires survival_analysis.py, subset_analysis.py, compare_groups_sensitivity.py,
slide_aggregation.py and: pip install pandas numpy scipy lifelines

For each feature (subset_analysis.MODULES plus TSR stroma-high and any
necrosis) and each cohort (All, ILC, IDC), three nested Cox models are fitted
on the SAME patients (known age and AJCC stage), so their HRs are comparable:

  unadjusted   OS ~ feature                  (+ lobular in the All cohort)
  age          OS ~ feature + age            (+ lobular)
  age_stage    OS ~ feature + age + stage    (+ lobular)

attenuation_pct = how much of the unadjusted log-HR the stage-adjusted model
removes (100% = fully explained by age and stage; negative = HR grew).

Stage is coded I / II / III-IV (--stage-coding 3level) or I-II vs III-IV
(binary). Within ILC (few deaths) binary coding is always used to save
parameters; the 'stage_coding' column records what each model used.

Continuous features are per SD (log1p first if skewed), as in
survival_analysis.py; HR > 1 means worse survival. BH-FDR q-values are
computed within each cohort x model. Follow-up is censored at --limit months
(default 120, matching the KM plots; 0 = no limit).

  python cox_stage_adjusted.py --output-dir cox_stage
  python cox_stage_adjusted.py --subset ER+/HER2- --output-dir cox_stage_er_her2

Outputs: cox_stage_adjusted.csv (long; plot_cox_forest.py can read it, e.g.
--models IDC_age_stage ILC_age_stage) and cox_stage_summary.csv (one row
per feature and cohort with the three HRs side by side).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_groups_sensitivity import bh_adjust, load_group
from slide_aggregation import aggregate_slides
from subset_analysis import GROUPS, MODULES, SUBSET, receptor_status
from survival_analysis import fit_term, load_covariates, load_survival, standardise

MODELS = ("unadjusted", "age", "age_stage")
COHORTS = {"All": None, "ILC": 1.0, "IDC": 0.0}


def load_module(args, stem, method, features, keep):
    patients = []
    for group in GROUPS:
        slides = load_group(args.data_dir / f"{group}_master_{stem}.csv", features)
        if keep is not None:
            slides = slides.loc[slides["subject_id"].isin(keep)]
        patients.append(aggregate_slides(slides, features, "subject_id", method)
                        .assign(lobular=float(group == "Lobular")))
    return pd.concat(patients)


def model_frame(data, model, cohort, stage_coding):
    """Feature x, covariates for the model, months and event."""
    frame = data[["x", "months", "event"]].copy()
    if cohort == "All":
        frame["lobular"] = data["lobular"]
    if model in ("age", "age_stage"):
        frame["age_per_10y"] = data["age"] / 10
    if model == "age_stage":
        if stage_coding == "binary":
            frame["stage_III_IV"] = data["stage"].eq("III-IV").astype(float)
        else:
            frame["stage_II"] = data["stage"].eq("II").astype(float)
            frame["stage_III_IV"] = data["stage"].eq("III-IV").astype(float)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clinical", type=Path, default=Path("data/brca_tcga_clinical_data.csv"))
    parser.add_argument("--survival", type=Path, default=Path("data/TCGA_BRCA_survival.csv"))
    parser.add_argument("--patient-column", default="Patient ID")
    parser.add_argument("--subset", choices=["all", SUBSET], default="all")
    parser.add_argument("--stage-coding", choices=["3level", "binary"], default="3level")
    parser.add_argument("--limit", type=float, default=120,
                        help="Censor follow-up at this many months (0 = no limit)")
    parser.add_argument("--necrosis-min-area", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("cox_stage"))
    args = parser.parse_args()

    try:
        survival = load_survival(args.survival)
        if args.limit:
            beyond = survival["months"] > args.limit
            survival.loc[beyond, "event"] = 0
            survival.loc[beyond, "months"] = args.limit
        covariates = load_covariates(args.clinical, args.patient_column)
        keep = None
        if args.subset == SUBSET:
            status = receptor_status(args.clinical, args.patient_column,
                                     "ER Status By IHC", "HER2 IHC+ FISH (FGP)")
            keep = set(status.index[status.eq(SUBSET)])

        rows = []
        for module, (stem, method, features) in MODULES.items():
            data = (load_module(args, stem, method, features, keep)
                    .join(survival, how="inner").join(covariates)
                    .dropna(subset=["age", "stage"]))

            tests = [(feature, "continuous") for feature in features]
            if module == "tils_tsr":
                ratio = data["WSI_tumor_stroma_ratio"]
                data["TSR_stroma_high"] = ratio.lt(1).astype(float).where(ratio.notna())
                tests.append(("TSR_stroma_high", "binary"))
            if module == "necrosis":
                area = data["wsi_necrosis_area_um2"]
                data["any_necrosis"] = area.gt(args.necrosis_min_area).astype(float).where(area.notna())
                tests.append(("any_necrosis", "binary"))

            for feature, kind in tests:
                if kind == "binary":
                    x, transform = data[feature], "binary"
                else:
                    x, transform = standardise(data[feature])
                    if x is None:
                        continue
                featured = data.assign(x=x).dropna(subset=["x"])

                for cohort, lobular in COHORTS.items():
                    subset = featured if lobular is None else featured.loc[featured["lobular"].eq(lobular)]
                    coding = "binary" if cohort == "ILC" else args.stage_coding
                    for model in MODELS:
                        frame = model_frame(subset, model, cohort, coding)
                        rows.append({
                            "module": module, "feature": feature, "transform": transform,
                            "cohort": cohort, "adjustment": model,
                            "model": f"{cohort}_{model}",
                            "stage_coding": coding if model == "age_stage" else "",
                            **fit_term(frame, "x"),
                        })

        results = pd.DataFrame(rows)
        results["q"] = np.nan
        for _, index in results.groupby("model").groups.items():
            results.loc[index, "q"] = bh_adjust(results.loc[index, "p"])
        results.insert(0, "limit_months", args.limit or np.nan)
        results.insert(0, "subset", args.subset)

        # Side by side: unadjusted vs age vs age+stage, plus attenuation.
        wide = results.pivot_table(
            index=["module", "feature", "cohort"], columns="adjustment",
            values=["HR", "HR_CI95_low", "HR_CI95_high", "p", "q"], aggfunc="first",
        )
        wide.columns = [f"{value}_{model}" for value, model in wide.columns]
        counts = (results.loc[results["adjustment"].eq("age_stage")]
                  .set_index(["module", "feature", "cohort"])[["n", "events"]])
        wide = counts.join(wide).reset_index()
        log_unadj = np.log(wide["HR_unadjusted"])
        wide["attenuation_pct"] = 100 * (log_unadj - np.log(wide["HR_age_stage"])) / log_unadj
        order = {m: i for i, m in enumerate(MODULES)}
        wide = wide.sort_values(["cohort", "module"], key=lambda s: s.map(order) if s.name == "module" else s)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "cox_stage_adjusted.csv", index=False)
        wide.to_csv(args.output_dir / "cox_stage_summary.csv", index=False)

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    columns = ["feature", "n", "events", "HR_unadjusted", "HR_age", "HR_age_stage",
               "HR_CI95_low_age_stage", "HR_CI95_high_age_stage", "p_age_stage",
               "q_age_stage", "attenuation_pct"]
    for cohort in COHORTS:
        print(f"\n=== {cohort} ===")
        print(wide.loc[wide["cohort"].eq(cohort), columns]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nSaved {args.output_dir / 'cox_stage_adjusted.csv'} and cox_stage_summary.csv")


if __name__ == "__main__":
    main()
