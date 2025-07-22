#!/bin/bash
#SBATCH --job-name=blip2-run
#SBATCH --output=blip2_%j.out
#SBATCH --error=blip2_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd ~/refbank-hackathon/lm-performance/

python call_lm3.py \
  --model Salesforce/blip2-opt-2.7b \
  --experiment_name hawkins2020_characterizing_cued \
  --n_trials 50 \
  --history_type yoked
