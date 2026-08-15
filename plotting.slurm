#!/bin/bash
#SBATCH --mem=8GB
#SBATCH --account=torch_pr_292_courant
#SBATCH --job-name=gen_plots
#SBATCH --mail-type=END
#SBATCH --mail-user=ms13725@nyu.edu
#SBATCH --output=/scratch/ms13725/codes/nfp4Run/gen_plots_%j.out
#SBATCH --error=/scratch/ms13725/codes/nfp4Run/gen_plots_%j.err

# Validate argument count
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <filename> <aspect> <iota> <prev job_id>"
    exit 1
fi

FILENAME=$1
PREFIX="${FILENAME#input.}"
ASPECT=$2
IOTA=$3
JOB_ID=$4

cd /scratch/ms13725/codes/nfp4Run/

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
BASE_DIR="run-${DATETIME}"

# Construct target directory hierarchy
mkdir -p "${BASE_DIR}/vmecInputs"
mkdir -p "${BASE_DIR}/vmecOutputs"
mkdir -p "${BASE_DIR}/logs"
mkdir -p "${BASE_DIR}/plots"

# Partition files into designated subdirectories using the prefix argument
mv ${FILENAME}_* "${BASE_DIR}/vmecInputs/"
mv wout_${PREFIX}_* "${BASE_DIR}/vmecOutputs/"
mv jac_log* objective_* residuals_* "${BASE_DIR}/logs/"

# Relocate standard error, standard output (using job_id), and persistent run files
mv *${JOB_ID}.err *${JOB_ID}.out parvmecinfo.txt threed1.${PREFIX}* "${BASE_DIR}/"


module purge && module load singularity-ce/4.3.3

singularity exec \
--overlay /scratch/ms13725/simsopt.ext3:rw \
--overlay /share/apps/images/intel-oneapi-2022.1.2.sqf:ro \
--overlay /share/apps/images/openmpi-4.1.2-ubuntu-22.04.1.sqf:ro \
/share/apps/images/ubuntu-22.04.1.sif \
/bin/bash << EOF

unset -f which
if [[ -e /ext3/apps/openmpi/4.1.2/env.sh ]]; then source /ext3/apps/openmpi/4.1.2/env.sh; fi
if [[ -e /ext3/env.sh ]]; then source /ext3/env.sh; fi

cd ${BASE_DIR}/logs/
python /scratch/ms13725/codes/nfp4Run/plot_total_obj.py
python /scratch/ms13725/codes/nfp4Run/plot_specific.py ${ASPECT} ${IOTA}

mv total_obj.pdf targets.pdf ../plots

cd ../vmecOutputs
python /scratch/ms13725/codes/nfp4Run/make_paraview.py ${PREFIX}
mv objectives_heat_map.vts ../plots

EOF