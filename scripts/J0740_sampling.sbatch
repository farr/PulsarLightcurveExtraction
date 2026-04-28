#!/bin/bash
#SBATCH --partition=genx
#SBATCH --cpus-per-task=4
#SBATCH --time=7-0
#SBATCH --job-name=J0740_sampling
#SBATCH --output=logs/J0740_sampling_%j.out
#SBATCH --error=logs/J0740_sampling_%j.err

mkdir -p logs

julia --threads 4 --gcthreads 1 --project=. scripts/J0740_sampling_script.jl "$@"
