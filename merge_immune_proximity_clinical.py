#!/usr/bin/env python3
"""Create a immune proximity-only dataframe from previously extracted CSV files.

Requires pandas. Optionally join survival data using --survival.
Use --features-csv to add survival to an already-created immune proximity CSV.
  python merge_immune_proximity_clinical.py --input extracted_csvs \
      --output immune_proximity_features.csv

Recursively finds immune_proximity_wsi_summary.csv and WSI-prefixed versions.
Output starts with wsi_name and sample_id, followed by all source feature
columns. Identifier collisions are retained under feature__<column>.
WSI identity comes from filename, extraction_report.csv, a WSI folder, or a
summary column (--wsi-column, default wsi_name). Unrecoverable names stay blank;
the feature row is still included. No arbitrary WSI names are generated.
Multi-row summaries require a distinct WSI name on every row. Cluster CSVs
are not aggregated. Missing/invalid files and names are listed in a report.
"""
import argparse
import re
from pathlib import Path
import pandas as pd

SUMMARY = 'immune_proximity_wsi_summary.csv'
SAMPLE = re.compile(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}$', re.I)
WSI = re.compile(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2})[A-Z]?-.+$', re.I)


def read_csv(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names after whitespace removal.')
    return frame


def build_dataframe(input_dir, wsi_column='wsi_name'):
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Input folder does not exist: {root}')
    report_names = {}
    report_path = root / 'extraction_report.csv'
    if report_path.is_file():
        report = read_csv(report_path)
        if {'sample_id', 'wsi_folder', 'filename', 'destination'}.issubset(report.columns):
            for _, row in report.iterrows():
                if row['filename'] == SUMMARY and row['destination']:
                    basename = row['destination'].replace(chr(92), '/').rsplit('/', 1)[-1]
                    key = (row['sample_id'].strip().upper(), basename)
                    value = row['wsi_folder'].strip()
                    if key in report_names and report_names[key] != value:
                        raise ValueError(f'Conflicting WSI names in extraction report: {key}')
                    report_names[key] = value

    files = sorted(p for p in root.rglob('*.csv') if p.name == SUMMARY or p.name.endswith('__' + SUMMARY))
    if not files:
        raise ValueError(f'No {SUMMARY} files found anywhere inside {root}. Check --input.')
    rows, issues = [], []
    for source in files:
        relatives = source.relative_to(root).parts[:-1]
        sample_id = ''
        folder_wsi = ''
        # Nearest enclosing sample/WSI folder, including input itself if selected.
        for name in reversed((root.name,) + relatives):
            if not sample_id and SAMPLE.fullmatch(name):
                sample_id = name.upper()
            match = WSI.fullmatch(name)
            if match and not folder_wsi:
                folder_wsi = name
                if not sample_id:
                    sample_id = match.group(1).upper()
        prefix = source.name[:-len('__' + SUMMARY)] if source.name.endswith('__' + SUMMARY) else ''
        inferred_name = prefix or folder_wsi
        if not sample_id and WSI.fullmatch(inferred_name):
            sample_id = WSI.fullmatch(inferred_name).group(1).upper()
        reported_name = report_names.get((sample_id, source.name), '')
        if reported_name and inferred_name and reported_name != inferred_name:
            issues.append([str(source), 'identifier_conflict', 'Report and filename/folder disagree.'])
            continue
        inferred_name = reported_name or inferred_name
        try:
            frame = read_csv(source)
            if frame.empty:
                raise ValueError('Summary contains no data rows.')
            if len(frame) > 1:
                if wsi_column not in frame or frame[wsi_column].str.strip().eq('').any():
                    raise ValueError(f'{len(frame)} rows but no complete {wsi_column} column; attach an example to adapt this layout.')
                if frame[wsi_column].str.strip().duplicated().any():
                    raise ValueError('Multiple rows for the same WSI; expected WSI-level summary features.')
            pending = []
            for _, values in frame.iterrows():
                csv_name = str(values.get(wsi_column, '')).strip()
                name = inferred_name or csv_name
                # A filename or extraction-report identity describes one WSI only.
                if len(frame) > 1 and inferred_name and csv_name != inferred_name:
                    raise ValueError('Multi-row WSI names conflict with file/report identity.')
                sample = sample_id
                name_match = WSI.fullmatch(name)
                if name_match:
                    from_name = name_match.group(1).upper()
                    if sample and sample != from_name:
                        raise ValueError('Sample folder disagrees with WSI identity.')
                    sample = from_name
                if not sample:
                    sample = str(values.get('sample_id', values.get('Sample ID', ''))).strip().upper()
                if not SAMPLE.fullmatch(sample):
                    raise ValueError('Cannot determine TCGA sample_id from folder, WSI name, or CSV.')
                record = {'wsi_name': name, 'sample_id': sample, 'subject_id': sample.rsplit('-', 1)[0]}
                for column, value in values.items():
                    target = 'feature__' + column if column in {'wsi_name', 'sample_id', 'subject_id', 'source_file'} else column
                    if target in record:
                        raise ValueError(f'Feature column name collision: {target}')
                    record[target] = value
                record['source_file'] = str(source)
                pending.append(record)
            rows.extend(pending)
            if any(not row['wsi_name'] for row in pending):
                issues.append([str(source), 'missing_wsi_name', 'Features included; keep extraction_report.csv to recover the original WSI name.'])
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            issues.append([str(source), 'invalid_summary', str(exc)])
    result = pd.DataFrame(rows)
    if not result.empty:
        named = result.loc[result['wsi_name'].ne('')]
        if named.duplicated(['sample_id', 'wsi_name']).any():
            raise ValueError('Multiple files map to the same WSI. Remove duplicate/stale copies before combining.')
        columns = ['wsi_name', 'sample_id', 'subject_id']
        features = [c for c in result if c not in columns + ['source_file']]
        result = result[columns + features + ['source_file']]
    return result, pd.DataFrame(issues, columns=['source_file', 'status', 'detail']), len(files)


def add_subject_id(df):
    df = df.copy()
    if 'sample_id' not in df:
        raise ValueError('Immune proximity CSV must contain sample_id.')
    df['sample_id'] = df['sample_id'].str.strip().str.upper()
    if not df['sample_id'].str.fullmatch(SAMPLE).all():
        raise ValueError('Invalid sample_id in immune proximity CSV.')
    subjects = df['sample_id'].str.rsplit('-', n=1).str[0]
    if 'subject_id' in df and not df['subject_id'].str.strip().str.upper().eq(subjects).all():
        raise ValueError('Existing subject_id disagrees with sample_id.')
    df['subject_id'] = subjects
    first = [c for c in ['wsi_name', 'sample_id', 'subject_id'] if c in df]
    return df[first + [c for c in df if c not in first]]


def merge_survival(df, survival_path):
    df = add_subject_id(df)
    survival = read_csv(Path(survival_path))
    if 'tcga_case_id' not in survival:
        raise ValueError('Survival CSV must contain tcga_case_id.')
    survival['tcga_case_id'] = survival['tcga_case_id'].str.strip().str.upper()
    if not survival['tcga_case_id'].str.fullmatch(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}').all():
        raise ValueError('Survival CSV contains invalid or blank tcga_case_id values.')
    survival = survival.drop_duplicates()
    if survival['tcga_case_id'].duplicated().any():
        raise ValueError('Conflicting survival records for the same tcga_case_id.')
    if 'survival_match' in df or 'survival_match' in survival:
        raise ValueError('Input already contains survival_match; use the immune proximity-only CSV.')
    # Stop instead of silently replacing existing feature columns.
    overlaps = set(df.columns) & set(survival.columns)
    if overlaps:
        raise ValueError(f'Columns already present in both inputs: {sorted(overlaps)}. Use immune proximity-only input or rename conflicting features.')
    merged = df.merge(survival, left_on='subject_id', right_on='tcga_case_id',
                      how='left', validate='many_to_one', indicator='survival_match')
    merged['survival_match'] = merged['survival_match'].map({
        'both': 'matched', 'left_only': 'missing_survival', 'right_only': 'unused'})
    assert len(merged) == len(df)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='Extracted CSV folder')
    inputs.add_argument('--features-csv', type=Path, help='Existing immune proximity dataframe CSV')
    parser.add_argument('--survival', type=Path, help='Optional survival CSV containing tcga_case_id')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wsi-column', default='wsi_name')
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = output.with_name(output.stem + '_issues.csv')
    sources = {p.resolve() for p in (args.features_csv, args.survival) if p is not None}
    if output in sources or report_path in sources:
        parser.error('Output files must not overwrite an input CSV.')
    if args.input and (output == args.input.resolve() or args.input.resolve() in output.parents):
        parser.error('Save the output outside the extracted input folder.')
    try:
        if args.features_csv:
            df = add_subject_id(read_csv(args.features_csv))
            if 'wsi_name' not in df:
                raise ValueError('Immune proximity CSV must contain wsi_name.')
            issues = pd.DataFrame(columns=['source_file', 'status', 'detail'])
        else:
            df, issues, file_count = build_dataframe(args.input, args.wsi_column)
            print(f'Found {file_count} summary files.')
        output.parent.mkdir(parents=True, exist_ok=True)
        if not args.features_csv:
            issues.to_csv(report_path, index=False)
        if df.empty:
            parser.error(f'No feature rows available. See {report_path} when extracting from a folder.')
        if args.survival:
            df = merge_survival(df, args.survival)
        df.to_csv(output, index=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f'Saved {len(df)} WSI rows: {output}')
    if args.survival:
        print(df['survival_match'].value_counts().to_string())
    if not args.features_csv:
        print(f'Issues: {len(issues)}; report: {report_path}')


if __name__ == '__main__':
    main()
