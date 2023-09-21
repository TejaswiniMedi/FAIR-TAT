# Targeted-Adversarial-Training
Targeted Adversarial Training for Image Classification

To conveniently install dependencies automatically with [anaconda](https://www.anaconda.com/) you can use:

```
conda env create -f env.yml

conda activate spline
```
To run the code :
```
python train.py --mode 'AT' --fname 'AT_CFA' --ccm --lambda-1 0.5
```
