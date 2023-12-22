# jaxirl

Contains Jax implementation of the IRL algorithms for inverse reinforcement learning.

## Running Experiments
To experts are already provided, but to re-run then, simply delete the corresponding expert file and they will be automatically retrained.
The default configs for the experts are in `jaxirl/configs/inner_training_configs.py`.
To change the default configs for the IRL training, change `jaxirl/configs/outer_training_configs.py`.

To train a learner, run:
```bash
python jaxirl/irl/main.py --loss loss_type --env env_name
```

This package supports training via:
- Behavioral Cloning (loss_type = BC)
- IRL (loss_type = IRL)
- EvIL (loss_type = XE)
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