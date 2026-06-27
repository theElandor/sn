#!/bin/bash
#SBATCH --job-name=IOS_normalizer_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

source venv/bin/activate
python train.py --data-root  input \
    --sampling fps \
    --points 4096 \
    --epochs 3 \
    --batch-size 4 \
    --fold-dir splits/fold_2 \
    --output-dir exp/rotation
