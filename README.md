<h1 align="center">JaxIRL</h1>

<p align="center">
      <img src="https://img.shields.io/badge/python-3.8_%7C_3.9-blue" />
      <a href= "https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
      <a href= "https://github.com/FLAIROx/jaxirl/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
       
</p>

[**Installation**](#install) | [**Setup**](#setup) | [**Algorithms**](#algorithms) | [**Citation**](#citation)

## Inverse Reinforcement Learning in JAX

Contains JAX implementation of algorithms for **inverse reinforcement learning** (IRL).
Inverse RL is an online approach to imitation learning where we try to **extract a reward function** that makes the expert optimal.
IRL *doesn't suffer from compounding errors* (like behavioural cloning) and doesn't need expert actions to train (only example trajectories of states). 
Depending on the environment and hyperparameters, our implementation is about 🔥 100x 🔥 faster than standard IRL implementations in PyTorch (e.g. 3.5 minutes to train a single hopper agent ⚡).
By running multiple agents in parallel, you can be even faster! (e.g. 200 walker agents can be trained in ~400 minutes on 1 GPU! That's 2 minutes per agent ⚡⚡).

<div class="collage">
    <div class="column" align="center">
        <div class="row" align="center">
            <img src="https://github.com/FLAIROx/jaxirl/blob/main/plots/hopper.png" alt="Hopper" width="40%">
            <img src="https://github.com/FLAIROx/jaxirl/blob/main/plots/walker2d.png" alt="walker" width="40%">
        </div>
        <div class="row" align="center">
            <img src="https://github.com/FLAIROx/jaxirl/blob/main/plots/ant.png" alt="ant" width="40%">
            <img src="https://github.com/FLAIROx/jaxirl/blob/main/plots/halfcheetah.png" alt="halfcheetah" width="40%">
        </div>
    </div>
</div>

## A game-theoretic perspective on IRL
<img src="https://github.com/FLAIROx/jaxirl/blob/main/plots/irl.png" align="left" alt="IRL" width="10%">

IRL is commonly framed as a **two-player zero-sum game** between a policy player and a reward function player. Intuitively, the reward function player tries to pick out differences between the current learner policy and the expert, while the policy player attempts to maximise this reward function to move closer to expert behaviour. This setup is effectively a GAN in the trajectory space, where the reward player is the Discriminator and the policy player is a Generator.

<br/><br/><br/> 

## Why JAX?
JAX is a game-changer in the world of machine learning, empowering researchers and developers to train models with unprecedented efficiency and scalability. Here's how it sets a new standard for performance:

- GPU Acceleration: JAX harnesses the full power of GPUs by JIT compiling code in XLA. Executing environments directly on the GPU, we eliminate CPU-GPU bottlenecks due to data transfer. This results in remarkable speedups compared to traditional frameworks like PyTorch.
- Parallel Training at Scale: JAX effortlessly scales to multi-environment and multi-agent training scenarios, enabling **efficient parallelization** for massive performance gains.

All our code can be used with `jit`, `vmap`, `pmap` and `scan` inside other pipelines. 
This allows you to:
- 🎲 Efficiently run tons of seeds in parallel on one GPU
- 💻 Perform rapid hyperparameter tuning


## Running Experiments
The experts are already provided, but to re-run them, simply delete the corresponding expert file and they will be automatically retrained.
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

Our IRL implementation is the [moment matching](https://arxiv.org/abs/2103.03236) version. 
This includes implementation tricks to make learning more stable, including decay on the discriminator and learner learning rates and gradient penalties on the discriminator.

## Reproduce Results
Simply run
```
python3 jaxirl/irl/main.py --env env_name --loss IRL -sd 1
```
and the default parameters in ```outer_training_configs.py``` and the trained experts in ```experts/``` will be used.

## Citation

If you find this code useful in your research, please cite:
```bibtex
@misc{sapora2024evil,
      title={EvIL: Evolution Strategies for Generalisable Imitation Learning}, 
      author={Silvia Sapora and Gokul Swamy and Chris Lu and Yee Whye Teh and Jakob Nicolaus Foerster},
      year={2024},
}
```

## See Also 🙌
Our work reused code, tricks and implementation details from the following libraries, we encourage you to take a look!

- [FastIRL](https://github.com/gkswamy98/fast_irl): PyTorch implementation of moment matching IRL and FILTER algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
