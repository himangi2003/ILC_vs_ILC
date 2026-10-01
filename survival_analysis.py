#!/usr/bin/env python3
"""Cox proportional-hazards analysis of spatial features and overall survival.

Requires: pip install pandas numpy scipy matplotlib lifelines

For every feature of every module (feature lists and per-module WSI
aggregation from subset_analysis.py), three Cox models are fitted:

  adjusted      OS ~ feature + lobular + covariates       (all patients)
  interaction   OS ~ feature * lobular + covariates       (does the feature's
                prognostic effect differ between ILC and IDC?)
  within_<grp>  OS ~ feature + within-covariates          (ILC or IDC only)

Continuous features are standardised (per SD); nonnegative, right-skewed
features (skewness > 1) are log1p-transformed first. Binary features are
TSR stroma-high (ratio < 1) and any necrosis. HR > 1 means worse survival.
For the interaction model, HR is the ratio of the feature's HR in ILC to
that in IDC. BH-FDR q-values are computed within each model type. A
Schoenfeld-residual test of proportional hazards is reported per term.

Survival: OS_days is only filled for deaths in TCGA_BRCA_survival.csv, so
censored patients take followup_days_to_follow_up (or
diagnosis_days_to_last_follow_up). Covariates come from the cBioPortal
clinical file: age at diagnosis and AJCC stage (I, II, III-IV).

  python survival_analysis.py --data-dir data --output-dir survival_results
  python survival_analysis.py --subset ER+/HER2- --output-dir survival_er_her2
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

from compare_groups_sensitivity import bh_adjust, load_group
from slide_aggregation import aggregate_slides
from subset_analysis import GROUPS, MODULES, SUBSET, receptor_status

DAYS_PER_MONTH = 30.4375
AGE_COLUMN = "Diagnosis Age"
STAGE_COLUMN = "Neoplasm Disease Stage American Joint Committee on Cancer Code"


def load_survival(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    required = {"tcga_case_id", "OS_event", "OS_days", "followup_days_to_follow_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    # OS_days is filled only for deaths; censored patients use last follow-up.
    days = pd.to_numeric(df["OS_days"], errors="coerce")
    for column in ("followup_days_to_follow_up", "diagnosis_days_to_last_follow_up"):
        if column in df:
            days = days.fillna(pd.to_numeric(df[column], errors="coerce"))

    survival = pd.DataFrame({
        "subject_id": df["tcga_case_id"].str.strip().str.upper(),
        "event": pd.to_numeric(df["OS_event"], errors="coerce"),
        "months": days / DAYS_PER_MONTH,
    }).dropna().drop_duplicates()
    survival = survival.loc[survival["event"].isin([0, 1]) & survival["months"].gt(0)]
    if survival["subject_id"].duplicated().any():
        raise ValueError(f"{path}: conflicting survival records for one patient")
    return survival.set_index("subject_id")


def load_covariates(path, patient_column):
    clinical = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clinical.columns = clinical.columns.str.strip()
    missing = {patient_column, AGE_COLUMN, STAGE_COLUMN} - set(clinical.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    roman = clinical[STAGE_COLUMN].str.strip().str.upper().str.extract(
        r"^STAGE (IV|III|II|I)", expand=False
    )
    covariates = pd.DataFrame({
        "subject_id": clinical[patient_column].str.strip().str.upper(),
        "age": pd.to_numeric(clinical[AGE_COLUMN], errors="coerce"),
        "stage": roman.map({"I": "I", "II": "II", "III": "III-IV", "IV": "III-IV"}),
    }).drop_duplicates()
    if covariates["subject_id"].duplicated().any():
        raise ValueError(f"{path}: conflicting age/stage records for one patient")
    return covariates.set_index("subject_id")


def standardise(values):
    """Return (z-scored values, transform label), or (None, reason) if unusable."""
    values = values.astype(float)
    observed = values.dropna()
    if observed.nunique() < 2:
        return None, "constant"
    transform = "z"
    if observed.min() >= 0 and observed.skew() > 1:
        values = np.log1p(values)
        transform = "log1p+z"
    return (values - values.mean()) / values.std(), transform


def design(data, terms, covariates):
    """Model frame with the requested terms, covariates and outcome; complete cases."""
    frame = data[terms].copy()
    if "age" in covariates:
        frame["age_per_10y"] = data["age"] / 10
    if "stage" in covariates:
        known = data["stage"].notna()
        frame["stage_II"] = data["stage"].eq("II").astype(float).where(known)
        frame["stage_III_IV"] = data["stage"].eq("III-IV").astype(float).where(known)
    frame["months"] = data["months"]
    frame["event"] = data["event"]
    return frame.dropna()


def fit_term(frame, term):
    result = {
        "n": len(frame), "events": int(frame["event"].sum()),
        "HR": np.nan, "HR_CI95_low": np.nan, "HR_CI95_high": np.nan,
        "p": np.nan, "PH_test_p": np.nan, "status": "ok",
    }
    if result["events"] < 5 or frame[term].nunique() < 2:
        result["status"] = "too_few_events_or_constant"
        return result
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxPHFitter().fit(frame, "months", "event")
            row = model.summary.loc[term]
            result.update({
                "HR": row["exp(coef)"],
                "HR_CI95_low": row["exp(coef) lower 95%"],
                "HR_CI95_high": row["exp(coef) upper 95%"],
                "p": row["p"],
            })
            try:
                ph = proportional_hazard_test(model, frame, time_transform="rank")
                result["PH_test_p"] = ph.summary.loc[term, "p"]
            except Exception:
                pass
    except Exception as exc:  # lifelines raises several convergence error types
        result["status"] = f"fit_failed: {type(exc).__name__}"
    return result


def km_plot(data, output_dir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("All patients: Lobular vs Ductal", data, "lobular", {1: "Lobular", 0: "Ductal"})]
    if "TSR_stroma_high" in data:
        for value, name in ((1, "Lobular"), (0, "Ductal")):
            panels.append((
                f"{name}: TSR stroma-high vs low",
                data.loc[data["lobular"].eq(value)],
                "TSR_stroma_high", {1: "Stroma-high", 0: "Stroma-low"},
            ))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for ax, (panel_title, frame, column, labels) in zip(axes.flat, panels):
        frame = frame.dropna(subset=[column, "months", "event"])
        groups = {value: frame.loc[frame[column].eq(value)] for value in labels}
        for value, label in labels.items():
            group = groups[value]
            if len(group):
                KaplanMeierFitter().fit(
                    group["months"], group["event"],
                    label=f"{label} (n={len(group)}, events={int(group['event'].sum())})",
                ).plot_survival_function(ax=ax, ci_show=False)
        if all(len(g) for g in groups.values()):
            first, second = groups.values()
            p = logrank_test(first["months"], second["months"],
                             first["event"], second["event"]).p_value
            panel_title += f"\nlog-rank p = {p:.3g}"
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Months")
        ax.set_ylabel("Overall survival")
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "kaplan_meier.png", dpi=300)
    fig.savefig(output_dir / "kaplan_meier.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clinical", type=Path, default=Path("data/brca_tcga_clinical_data.csv"))
    parser.add_argument("--survival", type=Path, default=Path("data/TCGA_BRCA_survival.csv"))
    parser.add_argument("--patient-column", default="Patient ID")
    parser.add_argument("--subset", choices=["all", SUBSET], default="all")
    parser.add_argument("--covariates", nargs="*", choices=["age", "stage"],
                        default=["age", "stage"],
                        help="Adjustment for the adjusted and interaction models")
    parser.add_argument("--within-covariates", nargs="*", choices=["age", "stage"],
                        default=["age"],
                        help="Adjustment within ILC/IDC (few ILC events: keep this small)")
    parser.add_argument("--necrosis-min-area", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("survival_results"))
    args = parser.parse_args()

    try:
        survival = load_survival(args.survival)
        covariates = load_covariates(args.clinical, args.patient_column)
        keep = None
        if args.subset == SUBSET:
            status = receptor_status(args.clinical, args.patient_column,
                                     "ER Status By IHC", "HER2 IHC+ FISH (FGP)")
            keep = set(status.index[status.eq(SUBSET)])

        rows, km_data = [], None
        for module, (stem, method, features) in MODULES.items():
            patients = []
            for group in GROUPS:
                slides = load_group(args.data_dir / f"{group}_master_{stem}.csv", features)
                if keep is not None:
                    slides = slides.loc[slides["subject_id"].isin(keep)]
                patients.append(
                    aggregate_slides(slides, features, "subject_id", method)
                    .assign(lobular=float(group == "Lobular"))
                )
            data = pd.concat(patients).join(survival, how="inner").join(covariates)

            tests = [(feature, "continuous") for feature in features]
            if module == "tils_tsr":
                ratio = data["WSI_tumor_stroma_ratio"]
                data["TSR_stroma_high"] = ratio.lt(1).astype(float).where(ratio.notna())
                tests.append(("TSR_stroma_high", "binary"))
            if module == "necrosis":
                area = data["wsi_necrosis_area_um2"]
                data["any_necrosis"] = area.gt(args.necrosis_min_area).astype(float).where(area.notna())
                tests.append(("any_necrosis", "binary"))

            if module == next(iter(MODULES)):
                # Histology alone, on the first module's patients.
                frame = design(data, ["lobular"], args.covariates)
                rows.append({"module": "histology", "feature": "lobular_vs_ductal",
                             "transform": "binary", "model": "adjusted",
                             **fit_term(frame, "lobular")})
            if module == "tils_tsr":
                km_data = data

            for feature, kind in tests:
                if kind == "binary":
                    x, transform = data[feature], "binary"
                else:
                    x, transform = standardise(data[feature])
                    if x is None:
                        continue
                frame = data.assign(x=x, x_by_lobular=x * data["lobular"])
                label = {"module": module, "feature": feature, "transform": transform}

                rows.append({**label, "model": "adjusted",
                             **fit_term(design(frame, ["x", "lobular"], args.covariates), "x")})
                rows.append({**label, "model": "interaction",
                             **fit_term(design(frame, ["x", "lobular", "x_by_lobular"],
                                               args.covariates), "x_by_lobular")})
                for value, group in ((1.0, "Lobular"), (0.0, "Ductal")):
                    within = frame.loc[frame["lobular"].eq(value)]
                    rows.append({**label, "model": f"within_{group}",
                                 **fit_term(design(within, ["x"], args.within_covariates), "x")})

        results = pd.DataFrame(rows)
        results["q"] = np.nan
        for model, index in results.groupby("model").groups.items():
            results.loc[index, "q"] = bh_adjust(results.loc[index, "p"])
        results.insert(0, "subset", args.subset)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "cox_results.csv", index=False)
        km_plot(km_data, args.output_dir, f"Overall survival ({args.subset} patients)")

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    pd.set_option("display.width", 220)
    fmt = lambda x: f"{x:.3g}"
    columns = ["module", "feature", "n", "events", "HR", "HR_CI95_low", "HR_CI95_high", "p", "q"]
    for model in ["adjusted", "interaction", "within_Lobular", "within_Ductal"]:
        print(f"\n=== {model} ===")
        print(results.loc[results["model"].eq(model), columns + ["PH_test_p"]]
              .to_string(index=False, float_format=fmt))
    print(f"\nResults: {args.output_dir / 'cox_results.csv'}")
    print(f"Kaplan-Meier: {args.output_dir / 'kaplan_meier.png'}")


if __name__ == "__main__":
    main()
