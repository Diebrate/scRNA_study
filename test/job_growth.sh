#!/bin/bash
#SBATCH --account=def-kdhkdh
#SBATCH --mem=1024M
#SBATCH --time=24:00:00
#SBATCH --job-name=otsim-growth
#SBATCH --output=%x-%J.out

module load python/3.8.2
module load scipy-stack
source ENV/bin/activate

python simcomp_batch.py ${1} ${2}
