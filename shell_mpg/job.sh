#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 23:58:00
#SBATCH -o /u/jungs/Targeted-Adversarial-Training/logs/cluster.%A.%a.%x.log
#SBATCH -a 0-7
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate spline

cd /u/jungs/Targeted-Adversarial-Training/

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    python target_train_new.py --data cifar100 --mode AT     --ccm                 --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
	python target_train_new.py --data cifar100 --mode FAT    --ccm                 --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
	python target_train_new.py --data cifar100 --mode TRADES --ccm                 --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
	python target_train_new.py --data cifar100 --mode AT     --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
	python target_train_new.py --data cifar100 --mode FAT    --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
	python target_train_new.py --data cifar100 --mode TRADES --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]
then
	python target_train_new.py --data cifar100 --mode AT     --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0                      --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]
then
	python target_train_new.py --data cifar100 --mode AT     --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 1                      --num_workers_train 0 --num_workers_valid 0 --num_workers_test 0
fi
