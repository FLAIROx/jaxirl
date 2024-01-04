<h1 align="center">JaxIRL</h1>

[**Installation**](#install) | [**Setup**](#setup) | [**Algorithms**](#algorithms) | [**Citation**](#cite)

## Inverse Reinforcement Learning in JAX

Contains JAX implementation of algorithms for inverse reinforcement learning (IRL).
Inverse RL is an online approach to imitation learning where we try to extract a reward function that makes the expert optimal.
IRL doesn't suffer from compounding errors (like behavioural cloning) and doesn't need expert actions to train (only example trajectories of states). 
Depending on the hyperparameters, our implementation is 2 to 10x faster than standard IRL implementations in PyTorch (e.g. 100 minutes to train hopper).

## What is IRL?
IRL is commonly framed as a two-player zero-sum game between a policy player and a reward function player. Intuitively, the reward function player tries to pick out differences between the current learner policy and the expert demonstration, while the policy player attempts to maximise this reward function to move closer to expert behaviour. This setup is effectively a GAN in the trajectory space, where the reward player is the Discriminator and the policy player is a Generator.

## Running Experiments
The experts are already provided, but to re-run then, simply delete the corresponding expert file and they will be automatically retrained.
The default configs for the experts are in `jaxirl/configs/inner_training_configs.py`.
To change the default configs for the IRL training, change `jaxirl/configs/outer_training_configs.py`.

To train an IRL agent, run:
```bash
python jaxirl/irl/main.py --loss loss_type --env env_name
```

This package supports training via:
- Behavioral Cloning (loss_type = BC)
- IRL (loss_type = IRL)
- Standard RL (loss_type = NONE)

We support the following brax environments:
- halfcheetah
- hopper
- walker
- ant

and classic control environments:
- cartpole
- pendulum
- reacher
- gridworld

## Setup

The high-level structure of this repository is as follows:
```
├── jaxirl  # package folder
│   ├── configs # standard configs for inner and outer loop
│   ├── envs # extra envs
│   ├── irl # main scripts that implement Imitation Learning and IRL algorithms
│   ├── ├── bc.py # Code for standard Behavioural Cloning, called when loss_type = BC
│   ├── ├── irl.py # Code implementing basic IRL algorithm, called when loss_type = IRL
│   ├── ├── gail_discriminator.py # Used by irl.py to implement IRL algorithm
│   ├── ├── main.py # Main script to call to execute all algorithms
│   ├── ├── rl.py # Code use to train basic RL agent, called when loss_type = NONE
|   ├── training # generated expert demos
│   ├── ├── ppo_v2_cont_irl.py # PPO implementation for continuous action envs
│   ├── ├── ppo_v2_irl.py # PPO implementation for discrete action envs
│   ├── ├── supervised.py # Standard supervised training implementation
│   ├── ├── wrappers.py # Utility wrappers for training
│   ├── utils # utility functions
├── experts # expert policies
├── experts_test # expert policies for test version of environment
```
### Install 
```
conda create -n jaxirl python=3.10.8
conda activate jaxirl
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=jaxirl:$PYTHONPATH
```
> [!IMPORTANT]
> All scripts should be run from under ```jaxirl/```. 

## Algorithms

Our IRL implementation is the moment matching version. 
This includes implementation tricks to make learning more stable, including decay on the discriminator and learner learning rates and gradient penalties on the discriminator.

## Citation

If you find this code useful in your research, please cite:
```bibtex
@misc{sapora2023evil,
      title={EvIL: Evolution Strategies for Generalisable Imitation Learning}, 
      author={Silvia Sapora and Chris Lu and Gokul Swamy and Yee Whye Teh and Jakob Nicolaus Foerster},
      year={2023},
}
```

## See Also 🙌
Our work is reused code, tricks and implementation details from the following libraries, we encourage you to take a look!

- [FastIRL](https://github.com/gkswamy98/fast_irl): PyTorch implementation of moment matching IRL and FILTER algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
