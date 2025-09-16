#!/bin/bash
#SBATCH -J lora_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
# module load cuda   # if needed
source ~/.bashrc || true
conda activate lvp || source env/bin/activate || true

CONFIG=${CONFIG:-configs/config.yaml}
N=${N:-128}
RANK=${RANK:-8}
SEED=${SEED:-41}
OUTDIR=${OUTDIR:-results}

python src/lora_train_eval.py --config $CONFIG --n_train $N --r $RANK --seed $SEED --outdir $OUTDIR
