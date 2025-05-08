# This is official code for [FAIR-TAT paper](https://ieeexplore.ieee.org/abstract/document/10943714)
This repository contains the code and experiments for the FAIR-TAT framework, which introduces class-wise fairness into adversarial training through tailored targeted attacks using CFPS-based class similarity. The implementation builds on adversarial training baselines such as AT, FAT, and TRADES, while incorporating targeted sampling and adaptive ε-scaling strategies.

To conveniently install dependencies automatically with [anaconda](https://www.anaconda.com/) you can use the following:

```
conda env create -f env.yml

conda activate spline
```

✅ Targeted Training (CFPS-guided targets, with adaptive ε)

```
python target_train_new.py --mode AT     --ccm --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob
python target_train_new.py --mode FAT    --ccm --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob
python target_train_new.py --mode TRADES --ccm --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob
```

Random Target Training (with adaptive ε)

```
python target_train_new.py --mode AT     --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob
python target_train_new.py --mode FAT    --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob
python target_train_new.py --mode TRADES --ccm --random_target --lambda-r 0.5 --lambda-c 0.5 --untargeted 0 --adaptive_eps G-rob
```


To Download Corruptions dataset
```
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-10-C.tar
tar -xvf CIFAR-100-C.tar
```


To cite our work

```
@INPROCEEDINGS{10943714,
  author={Medi, Tejaswini and Jung, Steffen and Keuper, Margret},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training}, 
  year={2025},
  volume={},
  number={},
  pages={7827-7836},
  keywords={Training;Computer vision;Analytical models;Accuracy;Computational modeling;Buildings;Artificial neural networks;Robustness;Data models;Resilience;adversarial fairness;adversarial robustness;adversarial training},
  doi={10.1109/WACV61041.2025.00760}}
```
contact in case of queries : tejaswini.medi@uni-mannheim.de
