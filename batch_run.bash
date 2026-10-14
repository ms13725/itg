#!/bin/bash

usage() {
    cat << EOF
Usage: $(basename "$0") <vmec_input> <aspect> <iota> <helicity_m> <helicity_n>
       $(basename "$0") -h | --help

Launch the full ITG turbulence optimization sweep for one VMEC input file.
Submits a 10-task Slurm job array (batch_optim.slurm), one task per run:
  tasks 0-4  turbulence targets 1-5, no quasisymmetry    -> t<N>_noqs/
  tasks 5-9  turbulence targets 1-5, with quasisymmetry  -> t<N>_qs/
Each task runs run_optimization.py, then plot_total_obj.py, plot_specific_{qs,noqs}.py
and make_paraview.py. The plots are moved into the run's plots/ subdirectory.

Arguments:
  vmec_input   VMEC input file (e.g. vmecInputs/input.nfp4_QH_warm_start)
  aspect       target aspect ratio (float)
  iota         target mean rotational transform (float)
  helicity_m   QS helicity m (int), used only by the 5 QS runs (e.g. 1)
  helicity_n   QS helicity n (int), used only by the 5 QS runs (QA: 0, QH: -1)

Output (created in the current directory):
  optim-<vmec_input name without "input.">/   (e.g. optim-nfp4_QH_warm_start/)
    optim_<jobid>_<task>.out/.err    slurm logs
    t1_noqs/ ... t5_noqs/  t1_qs/ ... t5_qs/
      <simsopt output, wout_final.nc>
      plots/

Example (placeholder values):
  bash $(basename "$0") vmecInputs/input.nfp4_QH_warm_start <aspect> <iota> 1 -1
EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if [ "$#" -ne 5 ]; then
    echo "error: expected 5 arguments, got $#" >&2
    echo >&2
    usage >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "error: VMEC input '$1' not found" >&2
    exit 1
fi

FILENAME=$1
ASPECT=$2
IOTA=$3
HELICITY_M=$4
HELICITY_N=$5

ABS_PATH=$(realpath "$FILENAME")
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# VMEC names its output files from everything after the first "input." in the
# input file's full path, so no directory in the run path may contain "input."
RUN_NAME=$(basename "$FILENAME")
RUN_NAME=${RUN_NAME#input.}
BASE_DIR="optim-${RUN_NAME}"

if [[ "$(pwd)/${BASE_DIR}" == *input.* ]]; then
    echo "error: run path '$(pwd)/${BASE_DIR}' contains 'input.', which breaks VMEC output file naming" >&2
    exit 1
fi

mkdir -p "${BASE_DIR}"

echo "made directory: ${BASE_DIR}"

cd "${BASE_DIR}"
sbatch "${SCRIPT_DIR}/batch_optim.slurm" "$ABS_PATH" "$ASPECT" "$IOTA" "$HELICITY_M" "$HELICITY_N"
