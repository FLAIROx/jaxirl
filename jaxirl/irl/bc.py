from abc import ABC

import jax
import jax.numpy as jnp
import wandb
from jaxirl.utils.utils import (
    RewardWrapper,
    get_action_size,
    get_network,
    get_observation_size,
)

from jaxirl.training.ppo_v2_irl import make_train, eval
from jaxirl.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)
from jaxirl.training.supervised import SupervisedIL


class BC(ABC):
    def __init__(
        self,
        env,
        training_config,
        es_config,
        logging_run,
        env_params,
        expert_data=None,
    ) -> None:
        super().__init__()
        self._env = env
        self._reward_network = None
        self.action_size = get_action_size(env, env_params)
        self.observation_size = get_observation_size(env, env_params)
        if training_config["DISCRETE"]:
            self.make_train = make_train
            self.eval = eval
        else:
            self.make_train = make_train_cont
            self.eval = eval_cont
        self.agent_net = get_network(self.action_size, training_config)
        self._include_action = False

        self.env_params = env_params
        self._training_config = training_config
        self._es_config = es_config
        self._generations = es_config["generations"]
        self._log_every = es_config["log_gen_every"]
        self._run = logging_run
        self.agent_net = get_network(self.action_size, training_config)
        if es_config["save_to_file"]:
            self._save_dir = es_config["save_dir"]
        self.num_gpus = len(jax.devices())
        self.supervised_config = self.get_supervised_config()
        self.expert_obsv = expert_data[0]
        self.expert_actions = expert_data[1]

    def get_supervised_config(self):
        # this is for IL
        return {
            "DISCRETE": self._training_config["DISCRETE"],
            "NUM_ACTIONS": self.action_size,
            "NUM_UPDATES": 1,
            "NUM_EPOCHS_PER_UPDATE": 8,
            "LR": 5e-3,
            "OBS_SIZE": self.observation_size,
        }

    def wandb_callback(self, gen, loss, reward):
        if gen % self._log_every == 0:
            metrics = {
                "mean_fitness": jnp.mean(loss),
                "avg_return": jnp.mean(reward),
            }
            wandb.log(
                step=int(gen),
                data=metrics,
            )

    def train_step(self, carry, unused):
        (rng, runner_state, gen) = carry
        eval_rng, train_rng, rng = jax.random.split(rng, 3)
        il_agent_train = SupervisedIL(
            network_il=self.agent_net,
            expert_obsv=self.expert_obsv.reshape(-1, self.expert_obsv.shape[-1]),
            expert_actions=self.expert_actions,
            config=self.supervised_config,
        )
        il_train = jax.jit(il_agent_train.train)
        il_train_state, loss_history = il_train(train_rng, runner_state)
        il_train_params = il_train_state[0].params
        no_diff_env = RewardWrapper(
            self._env,
            self.env_params,
            None,
            None,
            include_action=self._include_action,
            training_config=self._training_config,
        )
        _, _, cumr = self.eval(
            num_envs=self._es_config["num_eval_envs"],
            num_steps=self._es_config["num_eval_steps"],
            env=no_diff_env,
            env_params=self.env_params,
            agent_params=il_train_params,
            rng=eval_rng,
            network=self.agent_net,
            return_reward=True,
        )
        jax.debug.print(
            "{g} - loss {l} - return {f}", g=gen, l=loss_history[0][-1], f=cumr
        )
        if self._es_config["wandb_log"]:
            jax.debug.callback(self.wandb_callback, gen, loss_history[0][-1], cumr)
        return (rng, il_train_state, gen + 1), None

    def train(self, rng):
        runner_state = None
        carry_init, _ = jax.jit(self.train_step)((rng, runner_state, 0), None)
        print(self._generations)
        (_rng, last_runner_state, last_gen), _ = jax.lax.scan(
            self.train_step, carry_init, [jnp.zeros(self._generations)]
        )
        return last_runner_state
