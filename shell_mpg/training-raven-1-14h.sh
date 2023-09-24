#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 13:58:00
#SBATCH -o /u/jungs/MLPConv/logs/cluster.%A.%a.%x.log
#SBATCH -a 0-0
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate mlp

cd /u/jungs/Targeted-Adversarial-Training/

if [ "$#" -eq 2 ]; then
    python target_train_new.py \
    	--mode AT \
    	--ccm \
    	--random_target \
    	--lambda-1 $1 \
    	--untargeted $2
fi

if [ "$#" -ge 3 ]; then
    python target_train_new.py \
    	--mode AT \
    	--ccm \
    	--random_target \
    	--lambda-1 $1 \
    	--untargeted $2 \
    	--adaptive_eps "${@:3}"
fi