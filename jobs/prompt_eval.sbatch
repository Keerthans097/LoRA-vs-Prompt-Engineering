#!/bin/bash
#SBATCH -J prompt_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
# module load cuda   # if needed on your cluster
source ~/.bashrc || true
conda activate lvp || source env/bin/activate || true

CONFIG=${CONFIG:-configs/config.yaml}
KSHOT=${KSHOT:-0}
SEED=${SEED:-41}
OUT=${OUT:-results/prompt_${KSHOT}shot_seed${SEED}.json}

python src/prompt_eval.py --config $CONFIG --k_shot $KSHOT --seed $SEED --out $OUT
