from flax import struct
from typing import Any
import jax.numpy as jnp
import jax
from jaxirl.utils.utils import maybe_concat_action


@struct.dataclass
class BufferState:
    buffer_obsv: Any
    buffer_actions: Any
    buffer_size: int = 0
    norm_mean: Any = None
    norm_var: Any = None
    count: float = 1e-4


def update_norm_stats(state, obs):
    obs = obs.reshape(-1, obs.shape[-1])
    batch_mean = jnp.mean(obs, axis=0)
    batch_var = jnp.var(obs, axis=0)
    batch_count = obs.shape[0]

    delta = batch_mean - state.norm_mean
    tot_count = state.count + batch_count

    new_mean = state.norm_mean + delta * batch_count / tot_count
    m_a = state.norm_var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return {"mean": new_mean, "var": new_var, "count": new_count}


def normalise_obs(state, obs):
    return (obs - state.norm_mean) / jnp.sqrt(state.norm_var + 1e-8)


class ObsvActionBuffer:
    def __init__(
        self,
        obsv_shape,
        action_shape,
        include_action,
        ep_length=1000,
        envs=100,
        max_size=10000,
    ):
        self.ep_length = ep_length
        self.num_envs = envs
        self.max_size = max_size
        self.include_action = include_action
        if len(jnp.array([obsv_shape]).shape) == 1:
            self.obsv_shape = (obsv_shape,)
        else:
            self.obsv_shape = obsv_shape
        self.action_shape = action_shape

    def init_state(self, rng):
        return BufferState(
            buffer_obsv=jnp.zeros((self.max_size, *self.obsv_shape), dtype=jnp.float32),
            buffer_actions=jnp.zeros(
                (self.max_size, self.action_shape), dtype=jnp.float32
            ),
            buffer_size=0,
            norm_mean=jnp.zeros(self.obsv_shape[-1]),
            norm_var=jnp.ones(self.obsv_shape[-1]),
            count=1e-4,
        )

    def add(self, obsv, actions, key, state):
        # obsv shape = (num_updates, ep_length, num_envs, obsv_shape)
        chosen_envs = jax.random.randint(
            key, shape=(self.num_envs,), minval=0, maxval=obsv.shape[2]
        )
        chosen_obsv = obsv[:, :, chosen_envs].reshape(-1, *self.obsv_shape)
        chosen_actions = actions[:, :, chosen_envs].reshape(-1, self.action_shape)
        new_obsv_buffer = jnp.concatenate((chosen_obsv, state.buffer_obsv), axis=0)[
            : self.max_size
        ]
        new_actions_buffer = jnp.concatenate(
            (chosen_actions, state.buffer_actions), axis=0
        )[: self.max_size]
        new_stats = update_norm_stats(state, chosen_obsv)
        state = state.replace(
            buffer_obsv=new_obsv_buffer,
            buffer_actions=new_actions_buffer,
            buffer_size=jnp.minimum(
                state.buffer_size + chosen_obsv.shape[0], self.max_size
            ),
            count=new_stats["count"],
            norm_mean=new_stats["mean"],
            norm_var=new_stats["var"],
        )
        return state

    def sample(self, num_samples, key, state):
        chosen_idx = jax.random.randint(
            key, shape=(num_samples,), minval=0, maxval=state.buffer_size
        )
        obsv, actions = state.buffer_obsv[chosen_idx], state.buffer_actions[chosen_idx]
        imitation_data = maybe_concat_action(
            self.include_action,
            self.action_shape,
            normalise_obs(state, obsv),
            actions,
        )
        imitation_data_flat = imitation_data.reshape([-1, imitation_data.shape[-1]])
        return imitation_data_flat
