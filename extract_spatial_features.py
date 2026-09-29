#!/usr/bin/env python3
"""Copy eight selected ViSpace spatial-feature CSV files for TCGA sample IDs (Python 3, no packages).

Usage:
  python extract_spatial_features.py --input ViSpace_output \
      --ids sample_ids.txt --output extracted_spatial_features

IDs: one TCGA sample ID per line, e.g. TCGA-A8-A09Z-01.
Matches the two-digit sample code, ignoring the slide's vial letter (01Z).
Copies the two selected CSVs from each of cluster_tils_tsr_score,
immune_proximity, necrosis_feature, and tumor_morphology into OUTPUT/SAMPLE_ID.
For multiple matching slides, filenames are prefixed with the WSI folder name.
Automatically detects spatial_feature_results or spatial_features_results.
If both exist for a slide, select one explicitly with --results-folder.
Existing destinations are skipped; sources are never moved or changed.
"""
import argparse
import csv
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

ID_PATTERN = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", re.I)
WSI_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2})[A-Z]?(?:-|\.|$)", re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='ViSpace_output directory')
    parser.add_argument('--ids', required=True, type=Path, help='Text file: one sample ID per line')
    parser.add_argument('--output', required=True, type=Path, help='Destination directory')
    parser.add_argument('--results-folder', default='auto',
                        choices=['auto', 'spatial_features_results', 'spatial_feature_results'])
    args = parser.parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        parser.error(f'Input directory does not exist: {root}')
    if output == root or root in output.parents or output in root.parents:
        parser.error('Input and output must be separate, non-overlapping directories.')
    try:
        lines = args.ids.read_text(encoding='utf-8-sig').splitlines()
    except OSError as exc:
        parser.error(str(exc))
    ids = []
    for number, line in enumerate(lines, 1):
        value = line.strip().upper()
        if not value or value.startswith('#') or value == 'SAMPLE_ID':
            continue
        if not ID_PATTERN.fullmatch(value):
            parser.error(f'Invalid sample ID at line {number}: {line!r}; use one ID per line.')
        if value not in ids:
            ids.append(value)
    if not ids:
        parser.error('No sample IDs found.')

    slides = defaultdict(list)
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            match = WSI_PATTERN.match(folder.name)
            if match:
                slides[match.group(1).upper()].append(folder)

    output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    selected_files = {
        'cluster_tils_tsr_score': ('tils_tsr_by_cluster.csv', 'tils_tsr_wsi_summary.csv'),
        'immune_proximity': ('immune_proximity_by_cluster.csv', 'immune_proximity_wsi_summary.csv'),
        'necrosis_feature': ('necrosis_feature_by_cluster.csv', 'necrosis_feature_wsi_summary.csv'),
        'tumor_morphology': ('tumor_core_features_by_cluster.csv', 'tumor_core_wsi_summary.csv'),
    }
    results_dirs = {}
    for sample_id in ids:
        for folder in slides.get(sample_id, []):
            if args.results_folder == 'auto':
                candidates = [folder / name for name in
                              ('spatial_feature_results', 'spatial_features_results')
                              if (folder / name).is_dir()]
                if len(candidates) > 1:
                    parser.error(f'Both result-folder spellings exist in {folder}; use --results-folder.')
                results_dirs[folder] = candidates[0] if candidates else folder / 'spatial_feature_results'
            else:
                results_dirs[folder] = folder / args.results_folder
    for sample_id in ids:
        incomplete = False
        matches = slides.get(sample_id, [])
        sample_output = output / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        if not matches:
            incomplete = True
            rows.append([sample_id, '', '', 'missing_wsi_folder', '', '', ''])
        for folder in matches:
            for subfolder, filename in ((sub, name) for sub, names in selected_files.items() for name in names):
                source = results_dirs[folder] / subfolder / filename
                output_name = f'{folder.name}__{filename}' if len(matches) > 1 else filename
                destination = sample_output / output_name
                error = ''
                if not source.is_file():
                    status = 'missing_file'
                    incomplete = True
                elif destination.exists() or destination.is_symlink():
                    status = 'skipped_existing'
                else:
                    try:
                        # Stage each file to avoid leaving a partial destination on failure.
                        with tempfile.TemporaryDirectory(prefix='.copy-', dir=sample_output) as temporary:
                            staged = Path(temporary) / filename
                            shutil.copy2(source, staged)
                            staged.rename(destination)
                        status = 'copied'
                    except (OSError, shutil.Error) as exc:
                        status, error = 'copy_error', str(exc)
                rows.append([sample_id, folder.name, filename, status, str(source), str(destination), error])
        if incomplete:
            missing.append(sample_id)

    with (output / 'extraction_report.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'wsi_folder', 'filename', 'status', 'source', 'destination', 'error'])
        writer.writerows(rows)
    (output / 'missing_sample_ids.txt').write_text(''.join(f'{sid}\n' for sid in missing), encoding='utf-8')
    counts = defaultdict(int)
    for row in rows:
        counts[row[3]] += 1
    print(f'Requested samples: {len(ids)}; samples with missing source files/folders: {len(missing)}')
    for status, count in sorted(counts.items()):
        print(f'{status}: {count}')
    print(f'Reports saved in: {output}')
    return 1 if counts['copy_error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
