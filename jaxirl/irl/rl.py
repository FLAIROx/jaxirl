from abc import ABC

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import wandb
from jaxirl.utils.utils import (
    RewardWrapper,
    get_action_size,
    get_network,
)

from jaxirl.training.ppo_v2_irl import make_train, eval
from jaxirl.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)


class RL(ABC):
    def __init__(
        self,
        env,
        training_config,
        outer_config,
        logging_run,
        env_params,
    ) -> None:
        super().__init__()
        self._env = env
        self._reward_network = None
        self._shaping_network = None
        self.action_size = get_action_size(env, env_params)
        print("Inner loop update steps", training_config["NUM_UPDATES"])
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
        self._outer_config = outer_config
        self._generations = outer_config["generations"]
        self._log_every = outer_config["log_gen_every"]
        self._run = logging_run
        if outer_config["save_to_file"]:
            self._save_dir = outer_config["save_dir"]
        self.num_gpus = len(jax.devices())
        print("num gpus", self.num_gpus)

    def wandb_callback(self, gen, loss, train_metrics):
        if gen % self._log_every == 0:
            last_return = train_metrics["last_return"].reshape(-1)
            returns = train_metrics["returned_episode_returns"].reshape(-1)
            original_returns = train_metrics[
                "timestep_returned_episode_learned_returns"
            ].mean()
            plt.plot(original_returns)
            metrics = {
                "mean_fitness": loss,
                "avg_last_return": jnp.mean(last_return),
                "avg_return": jnp.mean(returns),
            }
            wandb.log(
                step=int(gen),
                data=metrics,
            )

    def train_agents(self, rng, runner_state=None):
        wrapped_env = RewardWrapper(
            self._env,
            self.env_params,
            None,
            None,
            None,
            None,
            include_action=self._include_action,
            training_config=self._training_config,
        )
        training_config = self._training_config.copy()
        training_config["LR"] = self._outer_config["lrate_init"]
        training_config["ANNEAL_LR"] = self._outer_config["inner_lr_linear"]
        train_fn = self.make_train(
            config=training_config,
            env=wrapped_env,
            env_params=self.env_params,
            runner_state_start=runner_state,
            log_timestep_returns=True,
        )
        train_out = jax.jit(train_fn)(rng)
        return train_out["runner_state"], train_out["metrics"]

    def train_step(self, carry, unused):
        rng, runner_state, gen = carry
        rng, rng_train = jax.random.split(rng, 2)

        runner_state, metrics = self.train_agents(
            rng=rng_train,
            runner_state=runner_state,
        )
        fitness = metrics["returned_episode_returns"].mean()
        jax.debug.print("{g} - return {f}", g=gen, f=fitness)
        # jax.debug.print("{g}", g=gen)
        if self._outer_config["wandb_log"]:
            jax.debug.callback(self.wandb_callback, gen, fitness, metrics)
        return (rng, runner_state, gen + 1), None

    def train(self, rng):
        print("TRAIN RL ONLY")
        runner_state = None
        carry_init, _ = jax.jit(self.train_step)((rng, runner_state, 0), None)
        (_rng, last_runner_state, last_gen), _ = jax.lax.scan(
            self.train_step, carry_init, [jnp.zeros(self._generations)]
        )
        return last_runner_state
