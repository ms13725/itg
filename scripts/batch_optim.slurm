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
    PLOT_SCRIPT=plot_specific_noqs.py
    PLOT_ARGS=""
else
    QS_TAG=qs
    QS_ARGS="--helicity-m $HELICITY_M --helicity-n $HELICITY_N"
    PLOT_SCRIPT=plot_specific_qs.py
    PLOT_ARGS="$HELICITY_M $HELICITY_N"
fi

# nfp is fixed by the input file, so read it from there ("NFP = 0004" -> 4)
NFP=$(grep -iE '^[[:space:]]*NFP[[:space:]]*=' "$FILENAME" | head -1 | sed -E 's/^[^=]*=[[:space:]]*([0-9]+).*/\1/')
NFP=${NFP:+$((10#$NFP))}
NFP=${NFP:-unknown}

# Suffix for the plot files moved into plots/, e.g. nfp4_3qs
SUFFIX="nfp${NFP}_${TARGET}${QS_TAG}"

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
echo "starting plotting"
python ${SRC_DIR}/plot_total_obj.py ${TARGET}
python ${SRC_DIR}/${PLOT_SCRIPT} ${ASPECT} ${IOTA} ${TARGET} ${PLOT_ARGS}
python ${SRC_DIR}/make_paraview.py
python ${SRC_DIR}/plot_surfaces.py ${TARGET}
# Move plots into plots/, renamed as <stem>_${SUFFIX}.<ext>
for pair in total_obj.pdf:total_obj targets.pdf:targets objectives_paraview.vts:paraview boozer.pdf:boozer heatmap.pdf:heatmap plot3.pdf:flux_surfaces; do
    src=\${pair%%:*}
    stem=\${pair##*:}
    ext=\${src##*.}
    if [[ -e "\$src" ]]; then
        mv "\$src" "plots/\${stem}_${SUFFIX}.\${ext}"
    else
        echo "skipping \$src (not found)"
    fi
done
EOF
