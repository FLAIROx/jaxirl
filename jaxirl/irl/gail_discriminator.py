from functools import partial
from typing import Tuple, Any, Callable
import jax
import jax.numpy as jnp
from flax import linen as flax_nn
import optax
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict


class Discriminator(flax_nn.Module):
    reward_net_hsize: jnp.ndarray
    learning_rate: float
    discr_updates: int
    n_features: int  # usually OBS_SIZE + ACTION SIZE
    transition_steps_decay: int
    discr_final_lr: float
    buffer: Any
    expert_data: jnp.ndarray
    batch_size: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = flax_nn.relu
    l1_loss: float = 0.0
    schedule_type: str = "linear"
    discr_loss: str = "bce"

    def setup(self):
        self.g_layers = [
            flax_nn.Dense(layer_size) for layer_size in self.reward_net_hsize
        ] + [flax_nn.Dense(1)]

    def __call__(self, x) -> jnp.ndarray:
        for i, layer in enumerate(self.g_layers):
            x = layer(x)
            if i != len(self.g_layers) - 1:
                x = self.activation_fn(x)
        return x

    def scheduler(self, step_number):
        lr = jax.lax.select(
            step_number == 0,
            self.learning_rate,
            self.learning_rate / jnp.maximum((step_number // self.discr_updates), 1),
        )
        return lr

    def linear_schedule(self, count):
        frac = 1.0 - (count // self.discr_updates) / self.transition_steps_decay
        lr = (self.learning_rate - self.discr_final_lr) * jnp.maximum(
            frac, 0
        ) + self.discr_final_lr
        return lr

    def _init_state(
        self,
        rng: jax.random.PRNGKeyArray,
    ) -> Tuple[TrainState, jax.random.PRNGKeyArray]:
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(self.n_features)
        if self.schedule_type == "linear":
            schedule = self.linear_schedule
        elif self.schedule_type == "constant":
            schedule = self.learning_rate
        elif self.schedule_type == "harmonic":
            schedule = self.scheduler
        else:
            raise ValueError(f"Schedule type {self.schedule_type} not recognized")

        train_state = TrainState.create(
            apply_fn=self.apply,
            params=self.init(_rng, init_x),
            tx=optax.adamw(learning_rate=schedule, weight_decay=self.l1_loss, eps=1e-5),
        )
        return train_state, rng

    def batch_loss(
        self,
        params: FrozenDict,
        expert_batch: jnp.ndarray,
        imitation_batch: jnp.ndarray,
        key: Any,
    ) -> jnp.ndarray:
        def loss_exp_bce(
            expert_transition: jnp.ndarray,
        ) -> jnp.ndarray:
            exp_d = self.apply(params, expert_transition)
            # exp_loss = optax.sigmoid_binary_cross_entropy(exp_d, 1.)
            return exp_d

        def loss_imit_bce(imitation_transition: jnp.ndarray) -> jnp.ndarray:
            imit_d = self.apply(params, imitation_transition)
            # imit_loss = optax.sigmoid_binary_cross_entropy(imit_d, 0.)
            return imit_d

        def loss_exp_mse(
            expert_transition: jnp.ndarray,
        ) -> jnp.ndarray:
            exp_d = self.apply(params, expert_transition)
            target = jnp.tile(jnp.array([1.0]), (exp_d.shape[0],))
            return optax.l2_loss(exp_d, target)

        def loss_imit_mse(imitation_transition: jnp.ndarray) -> jnp.ndarray:
            imit_d = self.apply(params, imitation_transition)
            target = jnp.tile(jnp.array([0.0]), (imit_d.shape[0],))
            return optax.l2_loss(imit_d, target)

        def reg_loss(
            params: FrozenDict,
        ) -> jnp.ndarray:
            flat_params, _ = jax.flatten_util.ravel_pytree(params)
            return self.l1_loss * jnp.abs(jnp.array(flat_params)).sum()

        @partial(jax.grad, argnums=1)
        def f_interpolate(
            params: FrozenDict,
            input: jnp.ndarray,
        ):
            return self.apply(params, input)[0]

        def interpolate(
            alpha: float,
            expert_batch: jnp.ndarray,
            imitation_batch: jnp.ndarray,
        ):
            return alpha * expert_batch + (1 - alpha) * imitation_batch

        if self.discr_loss == "mse":
            loss_expert = loss_exp_mse
            loss_imitation = loss_imit_mse
        elif self.discr_loss == "bce":
            loss_expert = loss_exp_bce
            loss_imitation = loss_imit_bce

        exp_loss = jnp.mean(jax.vmap(loss_expert)(expert_batch), axis=0)[0]
        imit_loss = jnp.mean(jax.vmap(loss_imitation)(imitation_batch), axis=0)[0]
        reg = reg_loss(params)
        alpha = jax.random.uniform(key, (imitation_batch.shape[0],))

        interpolated = jax.vmap(interpolate)(alpha, expert_batch, imitation_batch)
        gradients = jax.vmap(f_interpolate, (None, 0))(params, interpolated)
        gradients = gradients.reshape((expert_batch.shape[0], -1))
        grad_norm = jnp.linalg.norm(gradients, axis=1)
        grad_penalty = ((grad_norm - 0.4) ** 2).mean()

        # here we use 10 as a fixed parameter as a cost of the penalty.
        loss = exp_loss - imit_loss + 10 * grad_penalty

        return loss + reg

    def update_step(
        self,
        loss_grad_fn: Callable[
            [FrozenDict, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, FrozenDict]
        ],
        buffer_state: jnp.ndarray,
        carry: Tuple[TrainState, jax.random.PRNGKeyArray],
        unused: Any,
    ) -> Tuple[TrainState, jnp.ndarray]:
        train_state, key = carry
        key_exp_sel, key_imit_sel, train_key, key = jax.random.split(key, 4)
        expert_batch = jax.random.choice(
            key_exp_sel,
            self.expert_data,
            shape=(self.batch_size,),
            replace=False,
        )
        imitation_batch = self.buffer.sample(
            self.batch_size,
            key_imit_sel,
            buffer_state,
        )
        loss_val, grads = loss_grad_fn(
            train_state.params, expert_batch, imitation_batch, train_key
        )
        train_state = train_state.apply_gradients(grads=grads)
        return (train_state, key), loss_val

    def train_epoch(
        self,
        imit_data_buffer_state: Any,
        train_state: TrainState,
        key: Any,
    ) -> Tuple[TrainState, jnp.ndarray]:
        loss_grad_fn = jax.value_and_grad(self.batch_loss)
        _update_step = partial(self.update_step, loss_grad_fn, imit_data_buffer_state)

        train_state, losses = jax.lax.scan(
            _update_step, (train_state, key), xs=None, length=self.discr_updates
        )
        return train_state[0], losses

    def predict_reward(
        self,
        input,
        params: FrozenDict,
    ) -> jnp.ndarray:
        d = self.apply(params, input)
        return -d
