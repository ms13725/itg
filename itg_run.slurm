#!/bin/bash
#SBATCH --mem=48GB
#SBATCH --account=torch_pr_292_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=ITGRun
#SBATCH --mail-type=END
#SBATCH --mail-user=ms13725@nyu.edu
#SBATCH --output=/scratch/ms13725/codes/nfp4Run/nfp4_run_%j.out
#SBATCH --error=/scratch/ms13725/codes/nfp4Run/nfp4_run_%j.err


# Validate argument count
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <filename> <aspect> <iota>"
    exit 1
fi

FILENAME=$1
ASPECT=$2
IOTA=$3

cd /scratch/ms13725/codes/nfp4Run/

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

mpiexec -n $SLURM_NTASKS python run_optimization.py "$FILENAME" "$ASPECT" "$IOTA"

EOF
