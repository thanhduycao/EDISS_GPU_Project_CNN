#!/bin/bash

#SBATCH --job-name=cnn_benchmark
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G

# 1. Load the exact modules you used to compile
module purge
module load gcc/10.4.0
module load cuda/12.6.1

# 2. THE FIX: Explicitly export the library path so the GPU node sees it
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# 3. Navigate to the folder
cd $SLURM_SUBMIT_DIR

# 4. Debug: Check if the library is found (should verify the fix)
echo "Checking libraries:"
ldd ./xray_unrolled | grep curand

# 5. Run with --export=ALL to carry the environment variables to the GPU
echo "Starting run..."
srun --export=ALL ./xray_unrolled