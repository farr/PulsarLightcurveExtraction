#!/bin/bash
#SBATCH --job-name=J0740_sampling
#SBATCH --partition=cca
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=7-0
#SBATCH --output=logs/J0740_%j_submit.out
#SBATCH --error=logs/J0740_%j_submit.err

mkdir -p logs

julia --threads $SLURM_CPUS_PER_TASK --project=. scripts/J0740_sampling_script.jl --n-segments 10 > logs/J0740_10.log 2>&1 &
julia --threads $SLURM_CPUS_PER_TASK --project=. scripts/J0740_sampling_script.jl --n-segments 100 > logs/J0740_100.log 2>&1 &
julia --threads $SLURM_CPUS_PER_TASK --project=. scripts/J0740_sampling_script.jl --n-segments 1000 > logs/J0740_1000.log 2>&1 &
julia --threads $SLURM_CPUS_PER_TASK --project=. scripts/J0740_sampling_script.jl > logs/J0740.log 2>&1 &

wait