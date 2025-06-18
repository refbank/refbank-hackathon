#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
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

# MODEL_NAME="google/gemma-3-27b-it"
MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
# export VLLM_USE_V1=0
# MODEL_NAME="HuggingFaceM4/idefics2-8b"
EXPERIMENT_NAME="hawkins2020_characterizing_cued"
METHOD="direct"
HISTORY_TYPE="yoked"

python lm-performance/call_lm.py \
    --model $MODEL_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --method $METHOD \
    --history_type $HISTORY_TYPE