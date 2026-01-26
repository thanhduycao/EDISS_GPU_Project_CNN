#!/bin/bash
#SBATCH --job-name=cnn_benchmark
#SBATCH --account=project_2016790
#SBATCH --output=cnn_output_%j.txt

#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G

module purge
module load gcc/10.4.0
module load cuda/11.5.0
# Avoid CUDA memory fragmentation (important for large models)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Show allocated GPU (sanity check: should show NO MIG table)
nvidia-smi

# Always launch with srun inside batch job
cd $PWD
./cnn_example
