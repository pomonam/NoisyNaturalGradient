# Noisy Natural Gradient (noisy K-FAC & noisy EK-FAC)
This repository contains a clean-up code for noisy K-FAC ("Noisy Natural Gradient as Variational Inference") and noisy EK-FAC ("Eigenvalue Corrected Noisy Natural Gradient").
 
Papers: 
- [Noisy Natural Gradient as Variational Inference](http://proceedings.mlr.press/v80/zhang18l/zhang18l.pdf)
- [Eigenvalue Corrected Noisy Natural Gradient](https://arxiv.org/pdf/1811.12565.pdf)

## Usage
The repository is composed of two parts: regression and classification. The choice of hyper-parameters is described in the paper.

#### Noisy K-FAC
- Classification
```
python train.py --config config/classification/kfac_vgg16_plain.json
```

- Regression (single run)
```
python train.py --config config/regression/kfac_concrete.json
```

- Regression (repeated runs)
```
python regression_baseline.py --config config/regression/kfac_concrete.json
```

#### Noisy EK-FAC
- Classification
```
python train.py --config config/classification/ekfac_vgg16_plain.json
```

- Regression (single run)
```
python train.py --config config/regression/ekfac_concrete.json
```

- Regression (repeated runs)
```
python regression_baseline.py --config config/regression/ekfac_concrete.json
```

## Requirements
The code was implemented & tested in Python 3.5. All required modules are listed in requirements.txt and can be installed with the following command:
```
pip install -r requirements.txt
```
In addition, please install [zhusuan](https://github.com/thu-ml/zhusuan), a Python probabilistic programming library for Bayesian deep learning.

## Citation
To cite this work, please use:
```
@article{zhang2017noisy,
  title={Noisy Natural Gradient as Variational Inference},
  author={Zhang, Guodong and Sun, Shengyang and Duvenaud, David and Grosse, Roger},
  journal={arXiv preprint arXiv:1712.02390},
  year={2017}
}
@article{bae2018eigenvalue,
  title={Eigenvalue Corrected Noisy Natural Gradient},
  author={Bae, Juhan and Zhang, Guodong and Grosse, Roger},
  journal={arXiv preprint arXiv:1811.12565},
  year={2018}
}
```

## TensorBoard Visualization
The implementation supports TensorBoard visualization.
```
tensorboard --logdir=experiments/cifar10/ekfac_vgg16_aug/summary
```

## Contributors
- [Juhan Bae](https://github.com/pomonam)
- [Guodong Zhang](https://github.com/gd-zhang)
- [Shengyang Sun](https://github.com/ssydasheng)
