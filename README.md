# Targeted-Adversarial-Training
Targeted Adversarial Training for Image Classification

To conveniently install dependencies automatically with [anaconda](https://www.anaconda.com/) you can use the following:

```
conda env create -f env.yml

conda activate spline
```
To download CIFAR10 and create a test/validation split:

```
python generate_validation.py
```

To run the code :
```
python train.py --mode 'AT' --ccm --lambda-1 0.5
```
