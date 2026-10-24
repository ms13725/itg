#!/usr/bin/env python3
"""
Regenerate the plots of finished sweeps without re-optimizing: submit replot.slurm once per optim-* folder.

Each folder's aspect, iota, m and n come from its run_params.txt (written by batch_run.bash), or, for folders made
before that file existed, aspect and iota from the folder name (optim-<name>-<aspect>-<iota>-<YYYYMMDD_HHMMSS>) and
m and n from the "QS on (m=.., n=..)" line run_optimization.py prints into the optim_*.out logs. Every folder is
checked before anything is submitted. Standard library only, so it runs with the login node's python3.

Usage: python3 scripts/replot_runs.py <optim_dir> [<optim_dir> ...] [--dry-run]
       python3 scripts/replot_runs.py --all [--dry-run]
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
RUNS_DIR = CODE_ROOT / 'runs'
REPLOT = SCRIPT_DIR / 'replot.slurm'

# Anchored on the timestamp, so a negative iota (optim-x-8--0.42-...) still parses
FOLDER_RE = re.compile(r'^optim-.+-(?P<aspect>[^-]+)-(?P<iota>-?[^-]+)-\d{8}_\d{6}$')
QS_RE = re.compile(r'QS on \(m=(-?\d+), n=(-?\d+)\)')


def params_from_file(params_file):
    """Return [aspect, iota, m, n] from a run_params.txt."""
    with open(params_file, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        fields = [x.strip() for x in next(reader, [])]
    if len(fields) != 5:
        raise ValueError(f'{params_file.name}: expected 5 columns, got {len(fields)}')
    return fields[1:]


def params_from_name_and_logs(optim_dir):
    """Return [aspect, iota, m, n] from the folder name and the slurm logs."""
    match = FOLDER_RE.match(optim_dir.name)
    if not match:
        raise ValueError('no run_params.txt, and the folder name is not optim-<name>-<aspect>-<iota>-<timestamp>')

    helicities = set()
    for log in optim_dir.glob('optim_*.out'):
        helicities.update(QS_RE.findall(log.read_text(errors='replace')))
    if len(helicities) != 1:
        found = ', '.join(f'(m={m}, n={n})' for m, n in sorted(helicities)) or 'none'
        raise ValueError(f'expected one QS helicity in the optim_*.out logs, found {found}')
    m, n = helicities.pop()
    return [match['aspect'], match['iota'], m, n]


def check_folder(optim_dir):
    """Return (args for replot.slurm, None) or (None, reason the folder can't be replotted)."""
    if not optim_dir.is_dir():
        return None, 'not a directory'

    n_runs = sum(1 for d in optim_dir.glob('t[1-5]_*qs') if (d / 'wout_final.nc').is_file())
    if n_runs == 0:
        return None, 'no t<N>_qs/ or t<N>_noqs/ run with a wout_final.nc'

    params_file = optim_dir / 'run_params.txt'
    try:
        if params_file.is_file():
            params = params_from_file(params_file)
        else:
            params = params_from_name_and_logs(optim_dir)
    except ValueError as e:
        return None, str(e)

    aspect, iota, m, n = params
    for name, value, kind in [('aspect', aspect, float), ('iota', iota, float), ('m', m, int), ('n', n, int)]:
        try:
            kind(value)
        except ValueError:
            return None, f'{name} is not {"a number" if kind is float else "an integer"}: {value!r}'

    return [str(optim_dir)] + params, None


def main():
    parser = argparse.ArgumentParser(description='Regenerate the plots of finished optim-* sweep folders.')
    parser.add_argument('optim_dirs', nargs='*', help='optim-* folders to replot')
    parser.add_argument('--all', action='store_true', help='replot every optim-* folder in runs/')
    parser.add_argument('--dry-run', action='store_true', help='print the commands without submitting anything')
    args = parser.parse_args()

    if args.all == bool(args.optim_dirs):
        parser.error('give either optim-* folders or --all')
    if args.all:
        optim_dirs = sorted(d for d in RUNS_DIR.glob('optim-*') if d.is_dir())
        if not optim_dirs:
            sys.exit(f'no optim-* folders in {RUNS_DIR}')
    else:
        optim_dirs = [Path(d).resolve() for d in args.optim_dirs]

    jobs, errors = [], []
    for optim_dir in optim_dirs:
        replot_args, error = check_folder(optim_dir)
        if error:
            errors.append(f'{optim_dir.name}: {error}')
        else:
            jobs.append(replot_args)

    if errors:
        print(f'{len(errors)} folder(s) can\'t be replotted; nothing submitted:', file=sys.stderr)
        for error in errors:
            print('  ' + error, file=sys.stderr)
        sys.exit(1)

    failed = []
    for k, replot_args in enumerate(jobs, start=1):
        optim_dir, aspect, iota, m, n = replot_args
        print(f'[{k}/{len(jobs)}] {Path(optim_dir).name} aspect={aspect} iota={iota} m={m} n={n}', flush=True)
        command = ['sbatch', str(REPLOT)] + replot_args
        if args.dry_run:
            print('  ' + ' '.join(command))
            continue

        # Submit from the optim-* folder so the replot_<jobid>.out/.err logs land there
        if subprocess.run(command, cwd=optim_dir).returncode != 0:
            failed.append(k)

    if args.dry_run:
        print(f'dry run: {len(jobs)} folder(s), nothing submitted')
    else:
        print(f'submitted {len(jobs) - len(failed)}/{len(jobs)} folder(s)')
        if failed:
            print(f'failed folders (in order above): {failed}', file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
