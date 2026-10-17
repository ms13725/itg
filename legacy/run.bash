#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <filename> <aspect> <iota> <m> <n>"
    exit 1
fi

FILENAME=$1
ASPECT=$2
IOTA=$3
HELICITY_M=$4
HELICITY_N=$5

ABS_PATH=$(realpath "$FILENAME")

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
BASE_DIR="optim-${DATETIME}"
mkdir -p "${BASE_DIR}/plots"

echo "made directory: ${BASE_DIR}"

cd "${BASE_DIR}"
sbatch ../itg_run.slurm "$ABS_PATH" "$ASPECT" "$IOTA" "$HELICITY_M" "$HELICITY_N"
