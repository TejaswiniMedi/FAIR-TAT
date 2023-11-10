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
bs_loop="128 256"
adaptive_loop="T-rob G-cfps_T-cfps"
for lr in ${lrs_loop}
do
    for ep in ${epochs_loop}
    do
        for bs in ${bs_loop}
        do
            for ad in ${adaptive_loop}
            do
            	if [[ $SLURM_ARRAY_TASK_ID -eq COUNTER ]]
            	then
                	lr_max=$lr
                	epochs=$ep
                	batchsize=$bs
                	adaptive=$ad
            	fi
            	let COUNTER++
            done
        done
    done
done
adaptive=${adaptive//_/ }

python target_train_new.py \
    --mode AT \
    --ccm \
    --random_target \
    --lambda-r 0.5 \
    --lambda-c 1.5 \
    --untargeted 0 \
    --adaptive_eps $adaptive \
    --lr_max ${lr_max} \
    --num_workers_train 0 \
    --num_workers_valid 0 \
    --num_workers_test 0 \
    --lr_schedule step4 \
    --batch-size $batchsize \
    --epochs $epochs \
    --prefix ${SLURM_JOB_ID}