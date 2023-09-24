#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 13:58:00
#SBATCH -o /u/jungs/Targeted-Adversarial-Training/logs/cluster.%A.%a.%x.log
#SBATCH -a 0-0
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate mlp

cd /u/jungs/Targeted-Adversarial-Training/

if [ "$#" -eq 0 ]; then
    python target_train_new.py \
    	--mode AT \
    	--ccm \
    	--random_target \
    	--num_workers_train 0 \
    	--num_workers_valid 0 \
    	--num_workers_test 0
fi

if [ "$#" -eq 3 ]; then
    python target_train_new.py \
    	--mode AT \
    	--ccm \
    	--random_target \
    	--num_workers_train 0 \
    	--num_workers_valid 0 \
    	--num_workers_test 0 \
    	--lambda-r $1 \
    	--lambda-c $2 \
    	--untargeted $3
fi

if [ "$#" -ge 4 ]; then
    python target_train_new.py \
    	--mode AT \
    	--ccm \
    	--random_target \
    	--num_workers_train 0 \
    	--num_workers_valid 0 \
    	--num_workers_test 0 \
    	--lambda-r $1 \
    	--lambda-c $2 \
    	--untargeted $3 \
    	--adaptive_eps "${@:4}"
fi