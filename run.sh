#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 14
#SBATCH --partition cpu
#SBATCH --mem 20GB
#SBATCH --exclude c01

cd ${PWD}
source activate
conda activate base
./run_rigid_scan.sh ${1} 
