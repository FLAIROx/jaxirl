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


class ObsvActionBuffer:
    def __init__(
        self,
        obsv_shape,
        action_shape,
        include_action,
        envs=100,
        max_size=10000,
    ):
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
            buffer_obsv=jnp.zeros((self.max_size, *self.obsv_shape), dtype=jnp.float16),
            buffer_actions=jnp.zeros(
                (self.max_size, self.action_shape), dtype=jnp.float16
            ),
            buffer_size=0,
        )

    def add(self, obsv, actions, key, state):
        # obsv shape = (num_updates, inner_steps, num_envs, obsv_shape)
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
        state = state.replace(
            buffer_obsv=new_obsv_buffer,
            buffer_actions=new_actions_buffer,
            buffer_size=jnp.minimum(
                state.buffer_size + chosen_obsv.shape[0], self.max_size
            ),
        )
        return state

    def sample(self, num_samples, key, state):
        chosen_idx = jax.random.randint(
            key, shape=(num_samples,), minval=0, maxval=state.buffer_size
        )
        obsv, actions = state.buffer_obsv[chosen_idx], state.buffer_actions[chosen_idx]
        imitation_data = maybe_concat_action(
            self.include_action, self.action_shape, obsv, actions
        )
        imitation_data_flat = imitation_data.reshape([-1, imitation_data.shape[-1]])
        return imitation_data_flat
