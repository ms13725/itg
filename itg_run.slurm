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
#SBATCH --output=optim_%j.out
#SBATCH --error=optim_%j.err


# Validate argument count
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <filename> <aspect> <iota>"
    exit 1
fi

FILENAME=$1
ASPECT=$2
IOTA=$3

module purge && module load singularity-ce/4.3.3

singularity exec \
--overlay /scratch/ms13725/simsopt.ext3:rw \
--overlay /share/apps/images/intel-oneapi-2022.1.2.sqf:ro \
--overlay /share/apps/images/openmpi-4.1.2-ubuntu-22.04.1.sqf:ro \
/share/apps/images/ubuntu-22.04.1.sif \
/bin/bash << EOF

echo "in singularity env"
echo "$PWD"

unset -f which
if [[ -e /ext3/apps/openmpi/4.1.2/env.sh ]]; then source /ext3/apps/openmpi/4.1.2/env.sh; fi
if [[ -e /ext3/env.sh ]]; then source /ext3/env.sh; fi

mpiexec -n $SLURM_NTASKS python /scratch/ms13725/codes/run_optimization.py "$FILENAME" "$ASPECT" "$IOTA" < /dev/null
echo "starting plotting"
python /scratch/ms13725/codes/plot_total_obj.py
python /scratch/ms13725/codes/plot_specific.py ${ASPECT} ${IOTA}
python /scratch/ms13725/codes/make_paraview.py
mv total_obj.pdf targets.pdf objectives_heat_map.vts plots/
EOF