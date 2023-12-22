from jaxirl.envs.gridworld import GridWorldNew, EnvParams, generate_one_wall
from gymnax.environments.misc.reacher import EnvParams as ReacherEnvParams
from gymnax.environments.classic_control.cartpole import EnvParams as CartPoleEnvParams
from gymnax.environments.classic_control.pendulum import EnvParams as PendulumEnvParams
import jax.numpy as jnp
from jaxirl.configs.inner_training_configs import (
    CARTPOLE_CONFIG,
    GRIDWORLD_CONFIG,
    BRAX_CONFIG,
    PENDULUM_CONFIG,
    REACHER_CONFIG,
)
import gymnax
from purejaxrl.wrappers import BraxGymnaxWrapper


def get_env(env_name):
    if is_brax_env(env_name):
        env, env_params = BraxGymnaxWrapper(env_name), None
    elif env_name == "gridworld":
        env, env_params = get_gridworld_env()
    else:
        env, env_params = gymnax.make(env_name)
    if env_name == "Reacher-misc":
        config = REACHER_CONFIG.copy()
    elif env_name == "CartPole-v1":
        config = CARTPOLE_CONFIG.copy()
    elif is_brax_env(env_name):
        config = BRAX_CONFIG.copy()
        config["ENV_NAME"] = env_name
    elif env_name == "Pendulum-v1":
        config = PENDULUM_CONFIG.copy()
    elif env_name == "gridworld":
        config = GRIDWORLD_CONFIG.copy()
    else:
        raise NotImplementedError("env not implemented")
    return env, env_params, config


def get_eval_config(config):
    config["num_eval_envs"] = 100
    if is_brax_env(config["env"]):
        config["num_eval_steps"] = 1000
    elif config["env"] == "gridworld":
        config["num_eval_steps"] = 30
    elif config["env"] == "Reacher-misc":
        config["num_eval_steps"] = 100
    elif config["env"] == "CartPole-v1":
        config["num_eval_steps"] = 500
    elif config["env"] == "Pendulum-v1":
        config["num_eval_steps"] = 200
    return config


def is_brax_env(env_name):
    return (
        env_name == "hopper"
        or env_name == "halfcheetah"
        or env_name == "humanoid"
        or env_name == "ant"
        or env_name == "walker2d"
    )


def get_gridworld_env():
    env = GridWorldNew()
    mazes = jnp.ones(shape=(5, 5), dtype=jnp.int32)
    envs_params = EnvParams(mazes)
    return env, envs_params


def get_test_params(es_config):
    if es_config["env"] == "gridworld":
        mazes = generate_one_wall(5, 5, True, True, jnp.array([2]), jnp.array([1]))
        return EnvParams(mazes)
    elif es_config["env"] == "Reacher-misc":
        return ReacherEnvParams(torque_scale=10.0, max_steps_in_episode=200)
    elif es_config["env"] == "CartPole-v1":
        return CartPoleEnvParams(gravity=20)
    elif es_config["env"] == "Pendulum-v1":
        return PendulumEnvParams(max_torque=10.0)
    else:
        raise NotImplementedError(
            f"Test version of env {es_config['env']} not implemented"
        )
