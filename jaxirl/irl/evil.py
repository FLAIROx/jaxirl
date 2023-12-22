from abc import ABC
from functools import partial
import evosax
import wandb
import jax
import jax.numpy as jnp
from jaxirl.irl.irl import IRL
from jaxirl.utils.restart_util import PrevParamsHandler
from jaxirl.utils.utils import (
    RealReward,
    RewardWrapper,
    RewardNetwork,
    RewardType,
    get_network,
    get_observation_size,
    get_xentropy_match_score_expert,
    is_irl,
    is_reward,
)
from jaxirl.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
)
from jaxirl.training.ppo_v2_irl import make_train
import os.path as osp


class EvIL(ABC):
    def __init__(
        self,
        env,
        training_config,
        es_config,
        logging_run,
        env_params,
        expert_data,
    ) -> None:
        super().__init__()
        self._env = env
        self._reward_network = None
        if training_config["DISCRETE"]:
            action_num = env.action_space(env_params).n
            observation_shape = env.observation_space(env_params).shape[0]
            self.make_train = make_train
        else:
            action_num = env.action_space(env_params).shape[0]
            self.make_train = make_train_cont
            observation_shape = get_observation_size(env, env_params)[0]
        self.agent_net = get_network(action_num, training_config)
        reward_net = RewardNetwork(
            hsize=es_config["reward_net_hsize"],
            activation_fn=es_config["reward_net_activation_fn"],
            sigmoid=es_config["reward_net_sigmoid"],
        )
        self._include_action = False
        if RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE:
            self.in_features = observation_shape
            reward_params = reward_net.init(
                jax.random.PRNGKey(0), jnp.zeros(observation_shape)
            )
        elif RewardType[es_config["reward_type"]] == RewardType.REWARD_STATE_ACTION:
            self.in_features = observation_shape + action_num
            reward_params = reward_net.init(
                jax.random.PRNGKey(0), jnp.zeros(observation_shape + action_num)
            )
            self._include_action = True
        elif RewardType[es_config["reward_type"]] == RewardType.NONE:
            self.in_features = observation_shape
            reward_params = None
        else:
            raise NotImplementedError(
                f"reward type not implemented {es_config['reward_type']}"
            )

        if RealReward[es_config["real_reward"]] == RealReward.IRL_STATE:
            self.in_features = observation_shape
            self.irl_training_class = IRL(
                env=env,
                env_params=env_params,
                training_config=training_config,
                es_config=es_config,
                logging_run=None,
                expert_data=expert_data,
            )
        elif RealReward[es_config["real_reward"]] == RealReward.IRL_STATE_ACTION:
            self.in_features = observation_shape + action_num
            self._include_action = True
            self.irl_training_class = IRL(
                env=env,
                env_params=env_params,
                training_config=training_config,
                es_config=es_config,
                logging_run=None,
                expert_data=expert_data,
            )
        elif RealReward[es_config["real_reward"]] == RealReward.GROUND_TRUTH_REWARD:
            pass
        else:
            raise NotImplementedError(
                f"reward type not implemented {es_config['real_reward']}"
            )

        if is_reward(es_config):
            self._reward_network = reward_net
        elif RewardType[es_config["reward_type"]] == RewardType.NONE:
            self._reward_network = None
        else:
            raise NotImplementedError(
                f"reward type not implemented {es_config['reward_type']}"
            )

        self.env_params = env_params
        self._training_config = training_config
        self._es_config = es_config
        self._popsize = es_config["popsize"]
        self._generations = es_config["irl_generations"]
        self._log_every = es_config["log_gen_every"]
        self._run = logging_run
        self.expert_obsv = expert_data[0]
        self.expert_actions = expert_data[1]
        if es_config["save_to_file"]:
            self._save_dir = es_config["save_dir"]
        self.num_gpus = len(jax.devices())
        print("num gpus", self.num_gpus)

        # Initialize reshaper based on placeholder network shapes
        self._param_reshaper = evosax.ParameterReshaper(reward_params)
        self._es_num_params = self._param_reshaper.total_params

    def wandb_callback(self, state, fitness, train_metrics):
        if self._log_every % 0:
            last_return = train_metrics["last_return"].reshape(-1)
            returns = train_metrics["returned_episode_returns"].reshape(-1)

            metrics = {
                "fitness": wandb.Histogram(fitness),
                "max_fitness": jnp.max(fitness),
                "mean_fitness": jnp.mean(fitness),
                "max_last_return": jnp.max(last_return),
                "avg_last_return": jnp.mean(last_return),
                "hist_last_return": wandb.Histogram(last_return),
                "max_return": jnp.max(returns),
                "avg_return": jnp.mean(returns),
                "hist_return": wandb.Histogram(returns),
            }
            wandb.log(
                step=int(state.gen_counter.item()),
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

    def get_loss(self, loss_type, runner_state, metrics):
        return -self.xe_loss(runner_state)

    def save_to_file_callback(self, state):
        jnp.save(
            osp.join(self._save_dir, f"state_{int(state.gen_counter.item())}.npy"),
            state,
        )

    def train_agents_irl(self, rng, runner_state=None):
        last_discr_state = self.irl_training_class.train(rng)
        runner_state, metrics = jax.jit(self.irl_training_class.train_agents)(
            rew_network_params=last_discr_state.params,
            rng=rng,
            runner_state=None,
        )
        return runner_state, metrics

    def train_agents_reward(self, rew_network_params, rng, runner_state=None):
        new_reward_env = RewardWrapper(
            self._env,
            self.env_params,
            self._reward_network,
            rew_network_params,
            include_action=self._include_action,
            training_config=self._training_config,
        )
        train_fn = self.make_train(
            config=self._training_config,
            env=new_reward_env,
            env_params=self.env_params,
            runner_state_start=runner_state,
        )
        train_out = jax.jit(train_fn)(rng)
        return train_out["runner_state"], train_out["metrics"]

    def train_agents(self, rew_network_params, rng, runner_state=None):
        rng = jax.random.PRNGKey(self._es_config["seed"])
        if is_irl(self._es_config):
            return self.train_agents_irl(rng, runner_state)
        else:
            return self.train_agents_reward(rew_network_params, rng, runner_state)

    def get_training_fn(self, num_gpus):
        es_training_fn = jax.jit(
            jax.vmap(self.train_agents, in_axes=(0, 0, 0), out_axes=(0, 0))
        )
        if num_gpus > 1:
            es_training_fn = jax.pmap(
                es_training_fn,
                in_axes=(0, 0, 0),
                out_axes=(0, 0),
                devices=jax.devices(),
            )
        return es_training_fn

    def train(self, rng):
        strategy = evosax.OpenES(
            popsize=self._popsize,
            lrate_init=self._es_config["lrate_init"],
            num_dims=self._es_num_params,
            opt_name="adam",
            centered_rank=True,
            maximize=True,
        )
        es_params = strategy.default_params
        rng_es, rng = jax.random.split(rng, 2)
        state = strategy.initialize(rng_es, es_params)
        es_loss_fn = jax.jit(
            jax.vmap(
                partial(self.get_loss, self._es_config["loss"]),
                in_axes=(0, 0),
                out_axes=(0),
            )
        )
        if self.num_gpus > 1:
            es_loss_fn = jax.pmap(
                es_loss_fn, in_axes=(0, 0), out_axes=(0), devices=jax.devices()
            )
        es_training_fn = self.get_training_fn(self.num_gpus)
        prev_params_handler = PrevParamsHandler(
            self._es_config["train_restart"],
            popsize=self._popsize,
            max_size=self._es_config["max_buffer_size"],
            restart_percentage=self._es_config["restart_top_perc"],
        )
        prev_params_handler_state = prev_params_handler.init_state()

        def es_step(rng, state, prev_params_handler_state):
            rng, rng_iter, rng_train, rng_sample = jax.random.split(rng, 4)
            network_params, state = strategy.ask(rng_iter, state, es_params)
            network_params_tree = self._param_reshaper.reshape(network_params)
            runner_state = prev_params_handler.get_prev_params(
                prev_params_handler_state, rng_sample
            )
            rng_train = jax.random.split(rng_train, self._popsize // 2)
            rng_train = jnp.tile(rng_train, reps=(2, 1))
            if self.num_gpus > 1:
                rng_train = rng_train.reshape(self.num_gpus, -1, *rng_train.shape[1:])
                if runner_state is not None:
                    runner_state = jax.tree_map(
                        lambda x: x.reshape(self.num_gpus, -1, *x.shape[1:]),
                        runner_state,
                    )
            runner_state, metrics = es_training_fn(
                network_params_tree, rng_train, runner_state
            )
            pmap_fitness = es_loss_fn(runner_state, metrics)
            fitness = pmap_fitness.reshape(-1)
            if self.num_gpus > 1:
                runner_state = jax.tree_map(
                    lambda x: x.reshape(-1, *x.shape[2:]), runner_state
                )
            prev_params_handler_state = prev_params_handler.add_to_buffer(
                prev_params_handler_state, runner_state, fitness
            )
            state = strategy.tell(network_params, fitness, state, es_params)
            jax.debug.print(
                "{g} - fitness {f} - best {b}",
                g=state.gen_counter,
                f=jnp.mean(fitness),
                b=jnp.max(fitness),
            )

            if self._es_config["wandb_log"]:
                jax.debug.callback(self.wandb_callback, state, fitness, metrics)
                if self._es_config["save_to_file"]:
                    jax.debug.callback(self.save_to_file_callback, state)

            return rng, state, prev_params_handler_state

        for g in range(self._generations):
            rng, state, prev_params_handler_state = es_step(
                rng, state, prev_params_handler_state
            )

        return state.best_member
