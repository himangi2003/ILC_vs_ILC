import argparse
import re
from pathlib import Path

import pandas as pd


WSI_PATTERN = re.compile(
    r"^(?P<subject>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})"
    r"-(?P<sample_code>\d{2})[A-Z]?-.+$",
    re.IGNORECASE,
)


def load_table(path, id_column, key, prefix):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    if id_column not in df.columns:
        raise ValueError(f"{path}: missing column {id_column!r}")

    df[id_column] = df[id_column].str.strip().str.upper()
    if df[id_column].eq("").any():
        raise ValueError(f"{path}: contains blank IDs")

    df = df.drop_duplicates()

    duplicated = df[id_column].duplicated(keep=False)
    if duplicated.any():
        ids = df.loc[duplicated, id_column].unique().tolist()
        raise ValueError(
            f"{path}: conflicting records for {ids[:10]}. "
            "Resolve these before merging."
        )

    # Prefix source columns while retaining a separate merge key.
    result = df.add_prefix(prefix + "__")
    result[key] = df[id_column]
    return result


def load_wsi_names(path):
    records = []
    seen = set()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name in seen:
            continue

        match = WSI_PATTERN.fullmatch(name)
        if not match:
            raise ValueError(
                f"{path}, line {line_number}: expected a full TCGA WSI "
                f"folder name, got {name!r}"
            )

        subject_id = match.group("subject").upper()
        sample_id = f"{subject_id}-{match.group('sample_code')}"

        records.append({
            "subject_id": subject_id,
            "sample_id": sample_id,
            "wsi_name": name,
        })
        seen.add(name)

    if not records:
        raise ValueError("No WSI names found in the text file.")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Merge WSI names from TXT with clinical and survival CSVs."
    )
    parser.add_argument("--wsi-txt", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--survival", required=True, type=Path)
    parser.add_argument("--clinical-id", default="Sample ID")
    parser.add_argument("--survival-id", default="subject_id")
    parser.add_argument("--output", default="master.csv", type=Path)
    args = parser.parse_args()

    input_paths = {
        args.wsi_txt.resolve(),
        args.clinical.resolve(),
        args.survival.resolve(),
    }
    if args.output.resolve() in input_paths:
        parser.error("Output must not overwrite an input file.")

    try:
        wsi = load_wsi_names(args.wsi_txt)

        clinical = load_table(
            args.clinical, args.clinical_id, "sample_id", "clinical"
        )
        survival = load_table(
            args.survival, args.survival_id, "subject_id", "survival"
        )

        master = (
            wsi.merge(
                clinical,
                on="sample_id",
                how="left",
                validate="many_to_one",
                indicator="clinical_match",
            )
            .merge(
                survival,
                on="subject_id",
                how="left",
                validate="many_to_one",
                indicator="survival_match",
            )
            .sort_values(["subject_id", "sample_id", "wsi_name"])
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(args.output, index=False)

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    print(f"Saved {len(master)} WSI rows to {args.output}")
    print(f"Unique subjects: {master['subject_id'].nunique()}")
    print(
        "WSIs missing clinical data:",
        (master["clinical_match"] == "left_only").sum(),
    )
    print(
        "WSIs missing survival data:",
        (master["survival_match"] == "left_only").sum(),
    )


if __name__ == "__main__":
    main()