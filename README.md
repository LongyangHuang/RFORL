# RFORL

A PyTorch implementation for our paper "Offline Reinforcement Learning without Regularization and Pessimism". Our code is built off of [TD3-BC](https://github.com/sfujim/TD3_BC).


## Prerequisites

- PyTorch 2.0.1 with Python 3.7 
- MuJoCo 2.00 with mujoco-py 2.0.2.13
- [d4rl](https://github.com/rail-berkeley/d4rl) 1.1 or higher (with v2 datasets)


## Usage

For training RFORL on `Envname` (e.g. halfcheetah-expert-v2), run:

```
python main.py --env_name=halfcheetah-expert-v2
```

The results are collected in `./results`
