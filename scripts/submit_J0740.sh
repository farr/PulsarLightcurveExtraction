#!/bin/bash
#SBATCH --job-name=J0740_sampling
#SBATCH --partition=genx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-0
#SBATCH --output=logs/J0740_%j.out
#SBATCH --error=logs/J0740_%j.err

mkdir -p logs

julia --project=. scripts/J0740_sampling_script.jl
