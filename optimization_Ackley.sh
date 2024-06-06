#!/bin/bash
#SBATCH --job-name=Ackley
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=1G
#SBATCH --partition=small
#SBATCH --time=01:00:00

module load LUMI/22.12 partition/C
#module load cpeGNU/22.12
module load cray-python/3.10.10
#module load parallel/20230322
export PYTHONPATH="$PWD/src":$PYTHONPATH


python3 scripts/run_optimization_Ackley.py
