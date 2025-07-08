# PCDT: Pessimistic Critic Decision Transformer for Offline Reinforcement Learning

PCDT is a approach to Offline Reinforcement Learning where uses pessimistic Q-function  to enhance Decision Transformer stitching ability. This repo references DT(https://github.com/kzl/decision-transformer) and QRT(https://github.com/charleshsc/QT).

## Environment

### 1.Create and activate conda enviroment

```
conda create -n pcdt python==3.9
conda activate pcdt
```

### 2.Install PyTorch

The PCDT can be implemented through Pytorch==1.12.0 with CUDA 11.3:

```
conda install pytorch==1.12.0 pytorch-cuda=11.3 -c pytorch -c nvidia
```

### 3.Install D4RL datasets

Our work use D4RL benchmark, please install it:

```
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
```

### 4.Additional installations

To be clear, our code is derived from the Decision Transformer, so please install the transformers package:

```
pip install transformers==4.11.0
```

Now, you can run this.

## Quick Start

If you want use DT_models, please pretrain DT_models:

```
python pretrain.py --seed 123 --another_hyperparameters ...
```

The models are organized according to the following structure:

```
└── PCDT
    ├── D4RL
    ├── decision_transformer
    ├── DT_bcmodels
    │   └── ${env_name1}
    │	│
    │	└── ${env_name2}
    │	│
    │	└── ...
    ├── experiment.py
    ├── logger.py
    ├── pretrain,py
    └── tabulate.py
```



When your environment is ready, you could run experiment. For example:

```
python experiment.py --seed 123 --another_hyperparameters ...
```

## Citation

If you find this work is relevant with your research or applications, please feel free to cite our work!
