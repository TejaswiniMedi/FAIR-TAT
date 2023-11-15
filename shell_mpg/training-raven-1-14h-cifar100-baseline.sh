#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 13:58:00
#SBATCH -o /u/jungs/Targeted-Adversarial-Training/logs/cluster.%A.%a.%x.log
#SBATCH -a 0-3
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate mlp

cd /u/jungs/Targeted-Adversarial-Training/

COUNTER=0
lrs_loop="0.1"
epochs_loop="350"
bs_loop="128"
adaptive_loop="G-rob T-rob G-cfps_T-cfps"
kl_loop="0.0 2.0"

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    python target_train_new.py \
        --data cifar100 \
        --mode AT \
        --ccm \
        --kldiv 0.2 \
        --random_target \
        --lambda-r 1.5 \
        --lambda-c 1.5 \
        --untargeted 1 \
        --lr_max 0.01 \
        --num_workers_train 0 \
        --num_workers_valid 0 \
        --num_workers_test 0 \
        --lr_schedule steplr \
        --batch-size 128 \
        --epochs 350 \
        --prefix ${SLURM_JOB_ID}
fi

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    python target_train_new.py \
        --data cifar100 \
        --mode AT \
        --ccm \
        --kldiv 0.0 \
        --random_target \
        --lambda-r 1.5 \
        --lambda-c 1.5 \
        --untargeted 1 \
        --lr_max 0.01 \
        --num_workers_train 0 \
        --num_workers_valid 0 \
        --num_workers_test 0 \
        --lr_schedule steplr \
        --batch-size 128 \
        --epochs 350 \
        --prefix ${SLURM_JOB_ID}
fi

if [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python target_train_new.py \
        --data cifar100 \
        --mode AT \
        --ccm \
        --kldiv 0.2 \
        --random_target \
        --lambda-r 1.5 \
        --lambda-c 1.5 \
        --untargeted 0 \
        --lr_max 0.01 \
        --num_workers_train 0 \
        --num_workers_valid 0 \
        --num_workers_test 0 \
        --lr_schedule steplr \
        --batch-size 128 \
        --epochs 350 \
        --prefix ${SLURM_JOB_ID}
fi

if [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python target_train_new.py \
        --data cifar100 \
        --mode AT \
        --ccm \
        --kldiv 0.0 \
        --random_target \
        --lambda-r 1.5 \
        --lambda-c 1.5 \
        --untargeted 0 \
        --lr_max 0.01 \
        --num_workers_train 0 \
        --num_workers_valid 0 \
        --num_workers_test 0 \
        --lr_schedule steplr \
        --batch-size 128 \
        --epochs 350 \
        --prefix ${SLURM_JOB_ID}
fi