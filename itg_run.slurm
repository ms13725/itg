#!/bin/bash
#SBATCH --mem=48GB
#SBATCH --account=torch_pr_292_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --job-name=ITGRun
#SBATCH --mail-type=END
#SBATCH --mail-user=ms13725@nyu.edu
#SBATCH --output=/scratch/ms13725/codes/july22/itg_run_%j.out
#SBATCH --error=/scratch/ms13725/codes/july22/itg_run_%j.err

cd /scratch/ms13725/codes/july22/

/scratch/ms13725/run-simsopt.bash mpiexec -n $SLURM_NTASKS python run_optimization.py 
