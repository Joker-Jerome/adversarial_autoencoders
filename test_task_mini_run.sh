#!/bin/bash

#SBATCH --job-name=test_task_mini.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaolong.yu@yale.edu
#SBATCH --mem-per-cpu=48g
#SBATCH -t 10:00:00
#SBATCH --array=0

/ycga-gpfs/apps/hpc/software/dSQ/0.92/dSQBatch.py test_task_mini.txt
