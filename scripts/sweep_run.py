#!/usr/bin/env python3
"""
Launch batch_run.bash once per row of a parameter-sweep CSV.

CSV columns (first line is a header and is skipped):
    vmec_input, aspect, iota, helicity_m, helicity_n
Relative vmec_input paths are resolved against the code root (e.g. inputs/vmec/input.nfp2_torus), so the CSV can
live anywhere. Blank rows and rows starting with '#' are skipped. Every row is checked before anything is
submitted. Run folders go to runs/ under the code root (batch_run.bash decides that), from any working directory.
Standard library only, so it runs with the login node's python3.

Usage: python3 scripts/sweep_run.py <csv> [--dry-run]
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
BATCH_RUN = SCRIPT_DIR / 'batch_run.bash'


def read_rows(csv_path):
    """Return (line_no, fields) for each data row."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for fields in reader:
            fields = [x.strip() for x in fields]
            if not any(fields) or fields[0].startswith('#'):
                continue
            rows.append((reader.line_num, fields))
    return rows


def check_row(fields):
    """Return (args for batch_run.bash, None) or (None, reason the row is bad)."""
    if len(fields) != 5:
        return None, f'expected 5 columns, got {len(fields)}'
    vmec_input, aspect, iota, m, n = fields

    input_path = Path(vmec_input)
    if not input_path.is_absolute():
        input_path = CODE_ROOT / input_path
    if not input_path.is_file():
        return None, f'VMEC input not found: {input_path}'

    for name, value, kind in [('aspect', aspect, float), ('iota', iota, float), ('m', m, int), ('n', n, int)]:
        try:
            kind(value)
        except ValueError:
            return None, f'{name} is not {"a number" if kind is float else "an integer"}: {value!r}'

    # Pass the CSV strings through unchanged so folder names keep e.g. "6", not "6.0"
    return [str(input_path), aspect, iota, m, n], None


def main():
    parser = argparse.ArgumentParser(description='Run batch_run.bash for every row of a parameter-sweep CSV.')
    parser.add_argument('csv', help='CSV with columns vmec_input, aspect, iota, helicity_m, helicity_n')
    parser.add_argument('--dry-run', action='store_true', help='print the commands without submitting anything')
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    rows = read_rows(csv_path)

    runs, errors = [], []
    for line_no, fields in rows:
        run_args, error = check_row(fields)
        if error:
            errors.append(f'line {line_no}: {error}')
        else:
            runs.append(run_args)

    if errors:
        print(f'{csv_path.name} has {len(errors)} bad row(s); nothing submitted:', file=sys.stderr)
        for error in errors:
            print('  ' + error, file=sys.stderr)
        sys.exit(1)
    if not runs:
        sys.exit(f'{csv_path.name} has no data rows')

    failed = []
    for k, run_args in enumerate(runs, start=1):
        vmec_input, aspect, iota, m, n = run_args
        print(f'[{k}/{len(runs)}] {vmec_input} aspect={aspect} iota={iota} m={m} n={n}', flush=True)
        command = ['bash', str(BATCH_RUN)] + run_args
        if args.dry_run:
            print('  ' + ' '.join(command))
            continue

        if subprocess.run(command).returncode != 0:
            failed.append(k)
        if k < len(runs):
            time.sleep(1)  # folder names are timestamped to the second; keep them distinct

    if args.dry_run:
        print(f'dry run: {len(runs)} row(s), nothing submitted')
    else:
        print(f'submitted {len(runs) - len(failed)}/{len(runs)} row(s)')
        if failed:
            print(f'failed rows (in sweep order): {failed}', file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
