from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax.linen as flax_nn
from typing import Any
import wandb
import os
from datetime import datetime

from jaxirl.training.ppo_v2_irl import ActorCritic
from jaxirl.training.ppo_v2_cont_irl import ActorCritic as ActorCriticCont
from jaxirl.configs.outer_training_configs import (
    HALFCHEETAH_IRL_CONFIG,
    HOPPER_IRL_CONFIG,
    ANT_IRL_CONFIG,
    WALKER_IRL_CONFIG,
)
from jaxirl.utils.env_utils import get_eval_config


class TrainRNG(Enum):
    SAME = "SAME"
    SAME_EVERY_STEP = "SAME_EVERY_STEP"
    DIFFERENT = "DIFFERENT"
    DIFFERENT_IN_PAIRS = "DIFFERENT_IN_PAIRS"


class TrainRestart(Enum):
    NONE = "NONE"
    RESTART_BEST = "RESTART_BEST"
    SAMPLE_INIT = "SAMPLE_INIT"
    SAMPLE_RECENT_INIT = "SAMPLE_RECENT_INIT"


class RewardType(Enum):
    REWARD_STATE = "REWARD_STATE"
    REWARD_STATE_ACTION = "REWARD_STATE_ACTION"
    NONE = "NONE"


class RealReward(Enum):
    IRL_STATE = "IRL_STATE"
    IRL_STATE_ACTION = "IRL_STATE_ACTION"
    GROUND_TRUTH_REWARD = "GROUND_TRUTH_REWARD"


class LossType(Enum):
    XE = "XE"
    IRL = "IRL"
    NONE = "NONE"
    BC = "BC"


def get_plot_filename(es_config):
    if wandb.run is None:
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        return f"{os.getcwd()}/plots/{es_config['env']}_{date_time}.png"
    else:
        return f"{os.getcwd()}/plots/{es_config['env']}_{wandb.run.name}.png"


def _get_xentropy_match_score_expert(obsv, expert_actions, network_params, network):
    pi, _value = network.apply(network_params, obsv)
    # H(q, p)
    return optax.softmax_cross_entropy_with_integer_labels(
        pi.logits, expert_actions
    ).mean()


def get_xentropy_match_score_expert(
    params, expert_obsv, expert_actions, ppo_network, is_discrete
):
    if is_discrete:
        xe_function = _get_xentropy_match_score_expert
    else:
        xe_function = _get_nlog_l_expert
    partial_get_xentropy_match_score2 = partial(
        xe_function, network=ppo_network, network_params=params
    )
    xentropy = jax.vmap(partial_get_xentropy_match_score2, (0, 0), 0)(
        expert_obsv, expert_actions
    )
    return jnp.mean(xentropy)


def _get_nlog_l_expert(obsv, expert_actions, network_params, network):
    pi, _value = network.apply(network_params, obsv)
    # H(p, q)
    return jnp.mean(-pi.log_prob(expert_actions), axis=-1)


def get_action_size(env, env_params):
    if hasattr(env, "action_size"):
        return env.action_size
    elif hasattr(env.action_space(env_params), "n"):
        return env.action_space(env_params).n
    else:
        return env.action_space(env_params).shape[0]


def get_observation_size(env, env_params):
    if hasattr(env, "observation_size"):
        return (env.observation_size,)
    else:
        return env.observation_space(env_params).shape


def get_network(action_size, training_config):
    if training_config["DISCRETE"]:
        return ActorCritic(
            action_size,
            activation=training_config["ACTIVATION"],
        )
    else:
        return ActorCriticCont(
            action_size,
            activation=training_config["ACTIVATION"],
        )


def maybe_concat_action(include_action, action_size, obs, action):
    if include_action:
        if len(action.shape) == 0:
            action = jax.nn.one_hot(action, action_size)
        return jnp.concatenate([obs, action], axis=-1)
    else:
        return obs


class RewardNetwork(flax_nn.Module):
    hsize: Any
    activation_fn: Any
    sigmoid: bool = False

    @flax_nn.compact
    def __call__(self, x):
        for n in range(len(self.hsize)):
            x = flax_nn.Dense(features=self.hsize[n])(x)
            x = self.activation_fn(x)
        x = flax_nn.Dense(1, name="vals")(x)
        if self.sigmoid:
            return flax_nn.sigmoid(x)
        return x


def is_irl(es_config):
    return (
        RealReward[es_config["real_reward"]] == RealReward.IRL_STATE
        or RealReward[es_config["real_reward"]] == RealReward.IRL_STATE_ACTION
        or es_config["real_reward"] == RealReward.IRL_STATE
        or es_config["real_reward"] == RealReward.IRL_STATE_ACTION
    )


def is_reward(es_config):
    return (
        RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE
        or RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE_ACTION
        or es_config["reward_type"] == RewardType.REWARD_STATE
        or es_config["reward_type"] == RewardType.REWARD_STATE_ACTION
    )


def get_irl_config(es_config, original_training_config):
    if es_config is None:
        wandb.init(project="IRL")
        es_config = wandb.config

    original_training_config["NORMALIZE_REWARD"] = es_config["reward_normalize"]
    es_training_config = original_training_config.copy()
    if "inner_steps" in es_config:
        es_training_config["NUM_STEPS"] = es_config["inner_steps"]
    if "percentage_training" in es_config:
        es_training_config["NUM_UPDATES"] = int(
            original_training_config["NUM_UPDATES"] * es_config["percentage_training"]
        )
    elif "num_updates_inner_loop" in es_config:
        es_training_config["NUM_UPDATES"] = int(es_config["num_updates_inner_loop"])
        es_training_config["ORIG_NUM_UPDATES"] = int(
            original_training_config["ORIG_NUM_UPDATES"]
            * original_training_config["NUM_STEPS"]
            / (es_config["num_updates_inner_loop"] * es_training_config["NUM_STEPS"])
        )
    else:
        raise ValueError(
            "Either percentage_training or num_updates_inner_loop key must be present in the configuration"
        )
    # we set the total number of timesteps in IRL to be the same as standard RL
    if "irl_generations" not in es_config:
        if "percentage_training" in es_config:
            es_config["irl_generations"] = int(1 / es_config["percentage_training"])
        else:
            es_config["irl_generations"] = int(
                original_training_config["ORIG_NUM_UPDATES"]
                * original_training_config["NUM_STEPS"]
                / (
                    es_config["num_updates_inner_loop"]
                    * es_training_config["NUM_STEPS"]
                )
            )
    es_config["buffer_size"] = (
        es_config["num_eval_envs"]
        * es_config["inner_steps"]
        * es_config["irl_generations"]
    )
    print("Num IRL outer loop steps: ", es_config["irl_generations"])
    print("total timesteps RL", original_training_config["TOTAL_TIMESTEPS"])
    total_irl_timesteps = (
        es_config["irl_generations"]
        * es_training_config["NUM_UPDATES"]
        * es_training_config["NUM_STEPS"]
        * es_training_config["NUM_ENVS"]
    )
    print("Total timesteps IRL", total_irl_timesteps)
    print(
        "Total timesteps IRL inner loop",
        es_config["num_updates_inner_loop"]
        * es_config["inner_steps"]
        * es_training_config["NUM_ENVS"],
    )
    return es_config, es_training_config


def generate_config(args, seed):
    if args.loss == "IRL" and args.env == "hopper":
        config = HOPPER_IRL_CONFIG.copy()
    elif args.loss == "IRL" and args.env == "ant":
        config = ANT_IRL_CONFIG.copy()
    elif args.loss == "IRL" and args.env == "halfcheetah":
        config = HALFCHEETAH_IRL_CONFIG.copy()
    elif args.loss == "IRL" and args.env == "walker2d":
        config = WALKER_IRL_CONFIG.copy()
    else:
        config = {}
    if args.generations is not None:
        config["generations"] = args.generations
    config["seed"] = seed
    config["wandb_log"] = args.log
    config["plot"] = args.plot
    config["save_to_file"] = args.save
    config["env"] = args.env
    config["loss"] = args.loss
    config = get_eval_config(config)

    return config


class RewardWrapper:
    def __init__(
        self,
        env,
        env_params,
        reward_network,
        rew_network_params,
        include_action=False,
        training_config=None,
        invert_reward=False,
    ):
        self._env = env
        self.action_size = get_action_size(env, env_params)
        self.observation_size = get_observation_size(env, env_params)
        self.reward_network = reward_network
        self.rew_network_params = rew_network_params
        self.include_action = include_action
        self.gamma = training_config["GAMMA"]
        self.agent_net = get_network(self.action_size, training_config)
        self.invert_reward = int(invert_reward)

    def reset(self, key, params=None):
        obsv, env_state = self._env.reset(key, params)
        return obsv, env_state

    def step(self, key, state, action, params=None, prev_done=False, agent_params=None):
        obs, next_state, real_reward, done, info = self._env.step(
            key, state, action, params
        )
        reward = real_reward
        if self.reward_network is not None:
            reward_input = maybe_concat_action(
                self.include_action,
                self.action_size,
                self._get_obs(state, params),
                action,
            )
            reward = self.reward_network.apply(
                self.rew_network_params, reward_input
            ) * (-1 * self.invert_reward)
            new_reward = jnp.squeeze(reward)
            reward = new_reward
        info["real_reward"] = real_reward
        return obs, next_state, reward, done, info

    def _get_obs(self, state, params=None):
        if hasattr(state, "obs"):
            return state.obs
        elif hasattr(self._env, "get_obs"):
            try:
                return self._env.get_obs(state)
            except TypeError:
                return self._env.get_obs(state, params)

    def observation_space(self, params):
        return self._env.observation_space(params)

    def action_space(self, params):
        return self._env.action_space(params)
