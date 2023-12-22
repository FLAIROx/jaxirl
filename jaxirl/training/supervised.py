import jax
import jax.numpy as jnp
import optax

import flax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Any
from flax.training.train_state import TrainState

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class TransitionIL(NamedTuple):
    action_expert: jnp.ndarray
    obs: jnp.ndarray


class TrainInfo(NamedTuple):
    train_state: Any
    step_num: Any
    update_num: Any
    rng: Any


def loss_il(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    expert_obsv: jnp.ndarray,  # obs
    action_expert: jnp.ndarray,
    config: dict,
) -> jnp.ndarray:
    pi, _v = apply_fn(params_model, expert_obsv)
    if config["DISCRETE"]:
        total_loss = (
            jax.vmap(optax.softmax_cross_entropy_with_integer_labels, (0, 0), 0)(
                pi.logits, action_expert
            )
        ).mean()
        accuracy = jnp.mean(
            jnp.argmax(flax.linen.softmax(pi.logits), -1) == action_expert
        )
    else:
        total_loss = jnp.mean(-pi.log_prob(action_expert), axis=-1)
        accuracy = 0.0
    return total_loss, accuracy


class SupervisedIL:
    def __init__(self, network_il, expert_obsv, expert_actions, config):
        self.network_il = network_il
        self.expert_obsv = expert_obsv.reshape(-1, expert_obsv.shape[-1])
        if config["DISCRETE"]:
            self.expert_actions = expert_actions.reshape(-1)
        else:
            self.expert_actions = expert_actions.reshape(-1, config["NUM_ACTIONS"])
        self.config = config

    def _init_state(self, rng, runner_state=None):
        if runner_state is None:
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(self.config["OBS_SIZE"])
            network_params = self.network_il.init(_rng, x=init_x)
            tx = optax.adam(self.config["LR"], eps=1e-5)
            train_state = TrainState.create(
                apply_fn=self.network_il.apply,
                params=network_params,
                tx=tx,
            )
            train_state = TrainInfo(
                train_state=train_state, step_num=0, update_num=0, rng=rng
            )
        else:
            train_state = runner_state
        return (train_state, rng)

    def train(self, rng, runner_state=None):
        # INIT NETWORKS
        train_state, rng = self._init_state(rng, runner_state)

        # how many times we update the params
        val, loss_history = jax.lax.scan(
            self._update_step, train_state, None, self.config["NUM_UPDATES"]
        )

        return val, loss_history

    def _update_step(self, val, unused):
        new_train_state, loss_history = self._update(val.train_state)
        val = TrainInfo(
            train_state=new_train_state,
            step_num=0,
            update_num=(val.update_num + 1),
            rng=val.rng,
        )
        return val, (loss_history[0].mean(), loss_history[1].mean())

    def _update(self, train_state):
        _update_epoch_batch = self._mk_update_epoch()
        train_state, loss_history = jax.lax.scan(
            _update_epoch_batch, train_state, None, self.config["NUM_EPOCHS_PER_UPDATE"]
        )

        return train_state, loss_history

    def _mk_update_epoch(
        self,
    ):
        def _update_epoch_partial(train_state, unused):
            grad_fn = jax.value_and_grad(loss_il, has_aux=True)

            total_loss, grads = grad_fn(
                train_state.params,
                self.network_il.apply,
                expert_obsv=self.expert_obsv,
                action_expert=self.expert_actions,
                config=self.config,
            )

            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        return _update_epoch_partial

    def eval(self, rng, starting_params=None):
        # INIT NETWORKS
        train_state, rng = self._init_state(rng, starting_params)

        pi, _v = self.network_il.apply(train_state.params, self.expert_obsv)
        if self.config["DISCRETE"]:
            total_loss = (
                jax.vmap(optax.softmax_cross_entropy_with_integer_labels, (0, 0), 0)(
                    pi.logits, self.expert_actions
                )
            ).mean()
            accuracy = jnp.mean(
                jnp.argmax(flax.linen.softmax(pi.logits), -1) == self.expert_actions
            )
        else:
            total_loss = jnp.mean(-pi.log_prob(self.expert_actions), axis=-1)
            accuracy = 0

        return total_loss, accuracy
