#!/bin/bash
#SBATCH --job-name=grassmind
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=1G
#SBATCH --partition=small
#SBATCH --time=01:00:00

module load LUMI/22.12 partition/C
module load cpeGNU/22.12
module load cray-python/3.10.10
module load parallel/20230322
export PYTHONPATH="$PWD/src":$PYTHONPATH

# Create temporary directory to be used as a work directory
TMPDIR=`mktemp -d /tmp/grassdt.XXXXXXXXXX`
#echo $TMPDIR
trap "rm -rf $TMPDIR" EXIT

python3 scripts/run_optimization_grassland.py \
    par_file_templates/GCEF_EM_amb_2013_2022.tag \
    task_fileset \
    'observation_data/GCEF_{key}_EM_amb_2013_2022_{plot}.txt' \
    "$TMPDIR"
