#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops-hgx-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-output/lm_rollouts_%j.out
#SBATCH --error=slurm-output/lm_rollouts_%j.err


source ~/.zshrc

conda activate vllm

cd ~/refbank-hackathon

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
EXPERIMENT_NAME="hawkins2020_characterizing_cued"
N_TRIALS=100
METHOD="direct"

python lm-performance/call_lm.py \
    --model $MODEL_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --n_trials $N_TRIALS \
    --method $METHOD