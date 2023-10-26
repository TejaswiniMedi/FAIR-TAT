#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 13:58:00
#SBATCH -o /u/jungs/Targeted-Adversarial-Training/logs/cluster.%A.%a.%x.log
#SBATCH -a 0-6
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate mlp

cd /u/jungs/Targeted-Adversarial-Training/

COUNTER=0
lrs="0.2 0.1 0.05 0.025 0.01 0.005 0.001"
for lr in $lrs
do
	if [[ $SLURM_ARRAY_TASK_ID -eq COUNTER ]]
	then
    	lr_max=$lr
	fi
	let COUNTER++
done

python target_train_new.py \
    --mode AT \
    --ccm \
    --random_target \
    --lambda-r 0.5 \
    --lambda-c 1.5 \
    --untargeted 0 \
    --adaptive_eps T-rob \
    --lr_max ${lr_max} \
    --num_workers_train 0 \
    --num_workers_valid 0 \
    --num_workers_test 0 \
    --prefix ${SLURM_JOB_ID}