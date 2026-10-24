#!/bin/bash
#SBATCH --mem=20GB
#SBATCH --account=torch_pr_292_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=batch_optim
#SBATCH --mail-type=END
#SBATCH --mail-user=ms13725@nyu.edu
#SBATCH --array=0-9
#SBATCH --output=optim_%A_%a.out
#SBATCH --error=optim_%A_%a.err

# Array tasks 0-4: targets 1-5 without QS; tasks 5-9: targets 1-5 with QS.

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <filename> <aspect> <iota> <m> <n>"
    exit 1
fi

FILENAME=$1
ASPECT=$2
IOTA=$3
HELICITY_M=$4
HELICITY_N=$5

CODE_DIR=/scratch/ms13725/codes
SRC_DIR=${CODE_DIR}/src

TASK=$SLURM_ARRAY_TASK_ID
TARGET=$(( TASK % 5 + 1 ))
if (( TASK < 5 )); then
    QS_TAG=noqs
    QS_ARGS=""
    PLOT_ARGS=""
else
    QS_TAG=qs
    QS_ARGS="--helicity-m $HELICITY_M --helicity-n $HELICITY_N"
    PLOT_ARGS="$HELICITY_M $HELICITY_N"
fi

# nfp is fixed by the input file, so read it from there ("NFP = 0004" -> 4)
NFP=$(grep -iE '^[[:space:]]*NFP[[:space:]]*=' "$FILENAME" | head -1 | sed -E 's/^[^=]*=[[:space:]]*([0-9]+).*/\1/')
NFP=${NFP:+$((10#$NFP))}
NFP=${NFP:-unknown}

RUN_DIR="t${TARGET}_${QS_TAG}"
mkdir -p "${RUN_DIR}/plots"
cd "${RUN_DIR}"

echo "array task ${TASK}: target ${TARGET}, ${QS_TAG}, nfp ${NFP}, running in $(pwd)"

module purge && module load singularity-ce/4.5.1

singularity exec \
--overlay /scratch/ms13725/simsopt.ext3:ro \
--overlay /share/apps/images/intel-oneapi-2022.1.2.sqf:ro \
--overlay /share/apps/images/openmpi-4.1.2-ubuntu-22.04.1.sqf:ro \
/share/apps/images/ubuntu-22.04.1.sif \
/bin/bash << EOF

echo "in singularity env"

unset -f which
if [[ -e /ext3/apps/openmpi/4.1.2/env.sh ]]; then source /ext3/apps/openmpi/4.1.2/env.sh; fi
if [[ -e /ext3/env.sh ]]; then source /ext3/env.sh; fi

mpiexec -n $SLURM_NTASKS python ${SRC_DIR}/run_optimization.py "$FILENAME" "$ASPECT" "$IOTA" --target ${TARGET} ${QS_ARGS} < /dev/null
bash ${CODE_DIR}/scripts/postprocess.bash ${TARGET} ${ASPECT} ${IOTA} ${NFP} ${PLOT_ARGS}
EOF
