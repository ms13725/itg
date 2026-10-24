#!/bin/bash
#SBATCH --mem=8GB
#SBATCH --account=torch_pr_292_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=replot
#SBATCH --output=replot_%j.out
#SBATCH --error=replot_%j.err

# Regenerate the plots of every run in one optim-* folder without re-optimizing: runs postprocess.bash in each
# t<N>_noqs/ and t<N>_qs/ that has a wout_final.nc. Usually submitted by replot_runs.py from inside the optim-* folder,
# so the logs land there.

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <optim_dir> <aspect> <iota> <m> <n>"
    exit 1
fi

OPTIM_DIR=$(realpath "$1")
ASPECT=$2
IOTA=$3
HELICITY_M=$4
HELICITY_N=$5

CODE_DIR=/scratch/ms13725/codes
SRC_DIR=${CODE_DIR}/src

module purge && module load singularity-ce/4.5.1

# Run a command string inside the same singularity env as batch_optim.slurm
run_in_env() {
singularity exec \
--overlay /scratch/ms13725/simsopt.ext3:ro \
--overlay /share/apps/images/intel-oneapi-2022.1.2.sqf:ro \
--overlay /share/apps/images/openmpi-4.1.2-ubuntu-22.04.1.sqf:ro \
/share/apps/images/ubuntu-22.04.1.sif \
/bin/bash << EOF
unset -f which
if [[ -e /ext3/apps/openmpi/4.1.2/env.sh ]]; then source /ext3/apps/openmpi/4.1.2/env.sh; fi
if [[ -e /ext3/env.sh ]]; then source /ext3/env.sh; fi
$1
EOF
}

echo "replotting ${OPTIM_DIR} (aspect ${ASPECT}, iota ${IOTA}, m ${HELICITY_M}, n ${HELICITY_N})"

for RUN_DIR in "${OPTIM_DIR}"/t[1-5]_noqs "${OPTIM_DIR}"/t[1-5]_qs; do
    [ -d "${RUN_DIR}" ] || continue
    if [ ! -e "${RUN_DIR}/wout_final.nc" ]; then
        echo "skipping ${RUN_DIR} (no wout_final.nc)"
        continue
    fi

    RUN_NAME=$(basename "${RUN_DIR}")   # e.g. t3_qs
    TARGET=${RUN_NAME:1:1}
    if [[ "${RUN_NAME}" == *_qs ]]; then
        PLOT_ARGS="${HELICITY_M} ${HELICITY_N}"
    else
        PLOT_ARGS=""
    fi

    # nfp comes from the run's own wout_final.nc, so the VMEC input file isn't needed
    run_in_env "cd \"${RUN_DIR}\" && \
NFP=\$(PYTHONPATH=\"${SRC_DIR}\" python -c 'from wout_info import read_nfp; print(read_nfp() or \"unknown\")') && \
bash \"${CODE_DIR}/scripts/postprocess.bash\" ${TARGET} ${ASPECT} ${IOTA} \$NFP ${PLOT_ARGS}"
done

echo "done replotting ${OPTIM_DIR}"
