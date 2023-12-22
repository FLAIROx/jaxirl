import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, NamedTuple
import distrax
import optax
from flax.training.train_state import TrainState
import numpy as np
import gymnax
from flax.linen.initializers import constant, orthogonal
from jaxirl.training.wrappers import (
    LogWrapperRewardIRL,
    NormalizeVecRewardIRL,
    VecEnvRewardIRL,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def eval(
    num_envs,
    num_steps,
    env,
    env_params,
    agent_params,
    rng,
    network,
    return_reward=False,
):
    env = LogWrapperRewardIRL(env)
    env = VecEnvRewardIRL(env)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    obsv, env_state = env.reset(reset_rng, env_params)
    prev_done = jnp.array([True] * num_envs)

    # COLLECT TRAJECTORIES
    def _env_step(runner_state, unused):
        agent_params, env_state, last_obs, rng, prev_done = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(agent_params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_envs)
        obsv, env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params, prev_done, agent_params
        )
        prev_done = done
        transition = Transition(done, action, value, reward, log_prob, last_obs, info)
        runner_state = (agent_params, env_state, obsv, rng, prev_done)
        return runner_state, transition

    rng, _rng = jax.random.split(rng)
    runner_state = (agent_params, env_state, obsv, _rng, prev_done)
    runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, num_steps)
    if return_reward:
        return (
            traj_batch.obs,
            traj_batch.action,
            jnp.mean(jnp.sum(traj_batch.reward / jnp.sum(traj_batch.done), axis=0)),
        )
    else:
        return traj_batch.obs, traj_batch.action


def get_network(env, env_params, config):
    return ActorCritic(
        env.action_space(env_params).n,
        activation=config["ACTIVATION"],
    )


def make_train(config, env, env_params, runner_state_start, log_timestep_returns=False):
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # env = FlattenObservationWrapper(env)
    env = LogWrapperRewardIRL(env)
    env = VecEnvRewardIRL(env)
    env = NormalizeVecRewardIRL(env, config["GAMMA"], config["NORMALIZE_REWARD"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["ORIG_NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = get_network(env, env_params, config)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        # INIT ENV
        if runner_state_start is None:
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
            network_params = network.init(_rng, init_x)
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = env.reset(reset_rng, env_params)
            prev_done = jnp.array([True] * config["NUM_ENVS"])
            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obsv, _rng, prev_done)
        else:
            start_sum_of_retuns = runner_state_start[1].env_state.sum_of_returns
            start_timestep = runner_state_start[1].env_state.timestep
            logging_env_state = runner_state_start[1].env_state.replace(
                sum_of_returns=jnp.zeros_like(start_sum_of_retuns),
                timestep=jnp.zeros_like(start_timestep),
            )
            new_env_state = runner_state_start[1].replace(env_state=logging_env_state)
            runner_state = (
                runner_state_start[0],
                new_env_state,
                runner_state_start[2],
                runner_state_start[3],
                runner_state_start[4],
            )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, prev_done = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step,
                    env_state,
                    action,
                    env_params,
                    prev_done,
                    train_state.params,
                )
                prev_done = done
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng, prev_done)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, prev_done = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng, prev_done)
            return runner_state, metric

        if log_timestep_returns:
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["NUM_UPDATES"]
            )
        else:
            runner_state, _ = jax.lax.scan(
                _update_step, runner_state, None, config["NUM_UPDATES"]
            )
            metric = {}
        metric["returned_episode_returns"] = (
            runner_state[1].env_state.sum_of_returns
            / runner_state[1].env_state.timestep
        )
        if runner_state_start is not None:
            metric["diff_returned_episode_returns"] = (
                runner_state[1].env_state.sum_of_returns - start_sum_of_retuns
            ) / start_timestep
        else:
            metric["diff_returned_episode_returns"] = metric["returned_episode_returns"]
        metric["last_return"] = runner_state[1].env_state.returned_episode_returns
        metric["real_reward_var"] = runner_state[1].var
        return {"runner_state": runner_state, "metrics": metric}

    return train
