from abc import ABC
from functools import partial

import jax
import jax.numpy as jnp
from jaxirl.utils.buffer import ObsvActionBuffer
from jaxirl.utils.env_utils import get_test_params
from jaxirl.utils.utils import (
    RewardWrapper,
    RewardType,
    get_action_size,
    get_network,
    get_observation_size,
    get_xentropy_match_score_expert,
    maybe_concat_action,
)
from jaxirl.irl.gail_discriminator import Discriminator

from jaxirl.training.ppo_v2_irl import make_train, eval
from jaxirl.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)


class IRL(ABC):
    def __init__(
        self,
        env,
        training_config,
        es_config,
        logging_run,
        env_params,
        expert_data,
        test_expert_data,
    ) -> None:
        super().__init__()
        self._env = env
        self.test_env_params = get_test_params(es_config)
        self._reward_network = None
        self.action_size = get_action_size(env, env_params)
        if training_config["DISCRETE"]:
            observation_shape = env.observation_space(env_params).shape[0]
            self.make_train = make_train
            self.eval = eval
        else:
            self.make_train = make_train_cont
            self.eval = eval_cont
            observation_shape = get_observation_size(env, env_params)[0]
        self.agent_net = get_network(self.action_size, training_config)
        self._include_action = False
        if RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE:
            self.in_features = observation_shape
        elif RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE_ACTION:
            self.in_features = observation_shape + self.action_size
            self._include_action = True
        elif RewardType[es_config["reward_type"]] == RewardType.NONE:
            pass
        else:
            raise NotImplementedError(
                f"reward type not implemented {es_config['reward_type']}"
            )

        self.env_params = env_params
        self._training_config = training_config
        self._es_config = es_config
        self._generations = es_config["irl_generations"]
        self._log_every = es_config["log_gen_every"]
        self._run = logging_run
        self.expert_obsv = expert_data[0]
        self.expert_actions = expert_data[1]
        self.test_expert_obsv = test_expert_data[0]
        self.test_expert_actions = test_expert_data[1]
        self.inner_lr_linear = es_config["inner_lr_linear"]
        self._training_config["ANNEAL_LR"] = self.inner_lr_linear
        self._training_config["LR"] = es_config["inner_lr"]
        self.updates_every = es_config["discr_updates_every"]
        if es_config["save_to_file"]:
            self._save_dir = es_config["save_dir"]
        self.num_gpus = len(jax.devices())
        expert_data = maybe_concat_action(
            self._include_action,
            self.action_size,
            self.expert_obsv,
            self.expert_actions,
        )
        self.expert_data_flat = expert_data.reshape([-1, expert_data.shape[-1]])

        self.buffer = ObsvActionBuffer(
            obsv_shape=observation_shape,
            action_shape=self.action_size,
            include_action=self._include_action,
            envs=self._es_config["num_eval_envs"],
            max_size=es_config["buffer_size"],
        )
        self.discriminator_config = {
            "reward_net_hsize": es_config["reward_net_hsize"],
            "learning_rate": es_config["irl_lrate_init"],
            "discr_updates": es_config[
                "discr_updates"
            ],  # how many discriminator update steps should be done using the same batch of data
            "n_features": self.in_features,
            "l1_loss": es_config["discr_l1_loss"],
            "transition_steps_decay": int(es_config["discr_trans_decay"]),
            "schedule_type": es_config["discr_schedule_type"],
            "discr_loss": es_config["discr_loss"],
            "discr_final_lr": es_config["discr_final_lr"],
            "buffer": self.buffer,
            "expert_data": self.expert_data_flat,
            "batch_size": es_config["discr_batch_size"],
        }
        self.discr = Discriminator(**self.discriminator_config)
        self._reward_network = self.discr

    def wandb_callback(self, gen, episode_returns):
        if gen % self._log_every:
            (
                episode_returns_mean,
                episode_returns_min,
                episode_returns_max,
            ) = episode_returns
            metrics = {
                "avg_return": jnp.mean(episode_returns_mean),
                "min_return": episode_returns_min,
                "max_return": episode_returns_max,
            }
            if self._run is not None:
                self._run.log(
                    step=int(gen),
                    data=metrics,
                )

    def xe_loss(self, runner_state):
        return get_xentropy_match_score_expert(
            params=runner_state[0].params,
            expert_obsv=self.expert_obsv,
            expert_actions=self.expert_actions,
            ppo_network=self.agent_net,
            is_discrete=self._training_config["DISCRETE"],
        )

    def xe_test_loss(self, runner_state):
        return get_xentropy_match_score_expert(
            params=runner_state[0].params,
            expert_obsv=self.test_expert_obsv,
            expert_actions=self.test_expert_actions,
            ppo_network=self.agent_net,
            is_discrete=self._training_config["DISCRETE"],
        )

    @partial(jax.jit, static_argnums=(0))
    def get_discriminator_loss(self, buffer_state, discr_train_state, key):
        new_discr_train_state, _discr_losses = self.discr.train_epoch(
            imit_data_buffer_state=buffer_state,
            train_state=discr_train_state,
            key=key,
        )
        return new_discr_train_state

    def train_agents(
        self,
        rew_network_params,
        rng,
        runner_state=None,
        test=False,
    ):
        current_config = self._training_config.copy()
        if test:
            cur_env_params = self.test_env_params
            current_config["NUM_UPDATES"] = current_config["ORIG_NUM_UPDATES"]
        else:
            cur_env_params = self.env_params
        wrapped_env = RewardWrapper(
            self._env,
            cur_env_params,
            self._reward_network,
            rew_network_params=rew_network_params.params,
            include_action=self._include_action,
            training_config=self._training_config,
            invert_reward=True,
        )

        train_fn = self.make_train(
            config=current_config,
            env=wrapped_env,
            env_params=cur_env_params,
            runner_state_start=runner_state,
            log_timestep_returns=False,
        )
        train_out = jax.jit(train_fn)(rng)
        return (
            train_out["runner_state"],
            train_out["metrics"],
            train_out["obs"],
            train_out["actions"],
        )

    def train_step(self, carry, unused):
        rng, runner_state, discr_train_state, buffer_state, gen = carry

        rng, rng_loss, rng_train, rng_sample = jax.random.split(rng, 4)

        if self._es_config["dual"]:
            runner_state = None
        runner_state, metrics, actor_obs, actor_actions = self.train_agents(
            rew_network_params=discr_train_state,
            rng=rng_train,
            runner_state=runner_state,
        )
        buffer_state = self.buffer.add(
            actor_obs, actor_actions, rng_sample, buffer_state
        )
        new_discr_train_state = self.get_discriminator_loss(
            buffer_state,
            discr_train_state,
            rng_loss,
        )
        # xe_loss = self.xe_loss(runner_state)
        if self._es_config["dual"]:
            runner_state = None
        next_discr_train_state = jax.tree_map(
            lambda x, y: jax.lax.select(gen % self.updates_every == 0, x, y),
            new_discr_train_state,
            discr_train_state,
        )
        # test_runner_state, test_metrics = self.train_agents(discr_train_state.params, rng_train, None, None, test=True)
        # test_xe_loss = self.xe_test_loss(test_runner_state)
        episode_returns = metrics["returned_episode_returns"].mean()
        jax.debug.print(
            "{g} - return {r}",
            g=gen,
            r=episode_returns,
        )
        if self._es_config["wandb_log"]:
            jax.debug.callback(
                self.wandb_callback,
                gen,
                (
                    jax.lax.pmean(episode_returns, axis_name="i"),
                    jax.lax.pmin(episode_returns, axis_name="i"),
                    jax.lax.pmax(episode_returns, axis_name="i"),
                ),
            )
        return (
            rng,
            runner_state,
            next_discr_train_state,
            buffer_state,
            gen + 1,
        ), episode_returns

    def train(self, rng):
        discr_rng, rng = jax.random.split(rng)
        discr_rng = jax.random.split(discr_rng, self._es_config["seeds"])
        discr_train_state, rng = jax.vmap(self.discr._init_state)(discr_rng)
        buffer_state = jax.vmap(self.buffer.init_state)(discr_rng)

        runner_state = None
        vmap_train_step = jax.jit(
            jax.vmap(
                self.train_step,
                in_axes=((0, 0, 0, 0, None), None),
                axis_name="i",
                out_axes=((0, 0, 0, 0, None), 0),
            )
        )
        carry_init, _ = vmap_train_step(
            (rng, runner_state, discr_train_state, buffer_state, 0), None
        )
        (
            _rng,
            runner_state,
            last_discr_state,
            buffer_state,
            last_gen,
        ), irl_train_metrics = jax.lax.scan(
            vmap_train_step, carry_init, [jnp.zeros(self._generations)]
        )

        if self._es_config["run_test"]:
            test_runner_state, test_metrics = jax.vmap(
                self.train_agents, in_axes=(0, 0, None, None, None), out_axes=(0, 0)
            )(last_discr_state, _rng, None, None, True)
            train_runner_state, train_metrics = jax.vmap(
                self.train_agents, in_axes=(0, 0, None, None, None), out_axes=(0, 0)
            )(last_discr_state, _rng, None, None, False)
            test_xe_loss = jax.vmap(self.xe_test_loss)(test_runner_state)
            train_xe_loss = jax.vmap(self.xe_loss)(train_runner_state)
            metrics = {}
            metrics["test_metrics"] = jnp.mean(test_metrics["returned_episode_returns"])
            metrics["test_xe_loss"] = jnp.mean(test_xe_loss)
            metrics["final_train_metrics"] = jnp.mean(
                train_metrics["returned_episode_returns"]
            )
            metrics["final_train_xe_loss"] = jnp.mean(train_xe_loss)
            print(metrics)
            if self._run is not None:
                self._run.log(
                    step=int(last_gen),
                    data=metrics,
                )
        return last_discr_state, irl_train_metrics
