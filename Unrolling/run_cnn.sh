#!/bin/bash

#SBATCH --job-name=cnn_benchmark
#SBATCH --account=project_2016196   # REPLACE THIS with your project (e.g., project_200xxxx)
#SBATCH --partition=gputest             # 'gputest' is best for quick tests (max 15 mins)
#SBATCH --time=00:05:00                 # 5 minutes is plenty for this run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1               # Request 1 A100 GPU
#SBATCH --mem=4G                        # 4 Gigabytes of memory

# Load the same modules used during compilation
module load gcc/10.4.0 cuda/12.6.1

# Run the executable created by your Makefile
srun ./cnn_example
