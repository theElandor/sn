#!/bin/bash
#SBATCH --job-name=IOS_normalizer_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --output=runs/logs/train_%j.out
#SBATCH --error=runs/logs/train_%j.err

source /home/mlugli/ScanNormalizer/venv/bin/activate
python scripts/train.py --data-root data/input \
    --sampling fps \
    --points 16384 \
    --epochs 3 \
    --batch-size 4 \
    --fold-dir data/splits/fold_2 \
    --output-dir exp/rotation
