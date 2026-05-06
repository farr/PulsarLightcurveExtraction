#!/bin/bash
#SBATCH --partition=genx
#SBATCH --cpus-per-task=4
#SBATCH --time=7-0
#SBATCH --job-name=J0740_sampling
#SBATCH --output=logs/J0740_sampling_%j.out
#SBATCH --error=logs/J0740_sampling_%j.err

mkdir -p logs

unbuffer julia --project=. scripts/J0740_sampling_script.jl "$@" >> J0740_sampling_${SLURM_JOB_ID}.out 2>> J0740_sampling_${SLURM_JOB_ID}.err
