#!/bin/sh

#SBATCH --job-name=exp_prolog1_job
#SBATCH --output=logs/exp_prolog1_output_%j.out
#SBATCH --time=5:00:00
#SBATCH --partition=gpushort
#SBATCH --account=users
#SBATCH --mem=64000
#SBATCH --gres=gpu:a100:1

# add this if need more memory --> #SBATCH --cpus-per-task=16

export SWI_HOME_DIR=$HOME/swipl
export PATH=$SWI_HOME_DIR/bin:$PATH
export HOME=/scratch/s5112583
export WANDB_API_KEY=d8d6947a1d3f86f16545a695662f854d86f1198b

nvidia-smi

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0


# Uncomemment the commented lines below if you want to create a virtual environment and/or install dependencies

#python -m venv projectEnvironment
source projectEnvironment/bin/activate
#pip install --upgrade pip
#pip install unsloth[cu124-torch250]==2025.3.8 --no-deps
#pip install -r requirements.txt

PYTHON_SCRIPT="./exp_prolog1.py"

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost


PORT=$((29500 + RANDOM % 1000))  # Random port between 29500 and 30499
torchrun --nproc-per-node=1 --master-port=$PORT "$PYTHON_SCRIPT"
