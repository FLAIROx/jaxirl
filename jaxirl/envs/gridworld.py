import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
import flax


def get_state_from_obs(num_columns, num_rows, num_rewards, obs):
    obs = obs.reshape(num_columns, num_rows, -1)
    # agent_pos
    state = jnp.asarray(jnp.nonzero(obs[:, :, 1], size=(1), fill_value=num_columns))
    for num_reward in range(2, num_rewards + 2):
        reward = jnp.asarray(
            jnp.nonzero(obs[:, :, num_reward], size=(1), fill_value=num_columns)
        )
        state = jnp.concatenate([state, reward], axis=0)
    return state.reshape(-1)


def get_new_found_array(matching_pos, old_found_array):
    index = jax.numpy.nonzero(matching_pos, size=1)[0][0]
    return old_found_array.at[index].set(True)


def rooms_maze(
    key,
    columns: int,
    rows: int,
    vertical_wall_y: Optional[int] = 1,
    horizontal_wall_x: Optional[int] = 1,
):
    grid = jnp.ones((rows, columns))
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)

    # max value is exclusive
    vertical_wall_door_up = jax.random.randint(
        next(rng1), shape=(), minval=0, maxval=horizontal_wall_x
    )
    vertical_wall_door_down = jax.random.randint(
        next(rng2), shape=(), minval=horizontal_wall_x + 1, maxval=rows
    )
    horizontal_wall_door_left = jax.random.randint(
        next(rng3), shape=(), minval=0, maxval=vertical_wall_y
    )
    horizontal_wall_door_right = jax.random.randint(
        next(rng4), shape=(), minval=vertical_wall_y + 1, maxval=columns
    )
    vertical_wall_door_up = 0
    vertical_wall_door_down = 5
    horizontal_wall_door_left = 2
    horizontal_wall_door_right = 4

    grid = grid.at[horizontal_wall_x, :].set(0.0)
    grid = grid.at[:, vertical_wall_y].set(0.0)
    grid = grid.at[
        horizontal_wall_x, [horizontal_wall_door_left, horizontal_wall_door_right]
    ].set(1.0)
    grid = grid.at[
        [vertical_wall_door_up, vertical_wall_door_down], vertical_wall_y
    ].set(1.0)
    return jnp.array(grid, dtype=jnp.bool_)


def wall_maze(
    columns: int,
    rows: int,
    vertical: bool = True,
    wall: Optional[int] = 1,
    door: Optional[int] = 1,
):
    grid = jnp.ones((rows, columns))

    vertical_grid = grid.at[wall, :].set(0.0)
    horizontal_grid = grid.at[:, wall].set(0.0)
    vertical_grid = vertical_grid.at[wall, door].set(1.0)
    horizontal_grid = horizontal_grid.at[door, wall].set(1.0)
    grid = jax.lax.select(vertical, vertical_grid, horizontal_grid)
    return jnp.array(grid, dtype=jnp.bool_)


def generate_walls(
    key,
    rows: int,
    columns: int,
    wall: bool = True,
    wall_x: int = None,
    wall_y: int = None,
):
    return jax.lax.select(
        wall,
        rooms_maze(key, columns, rows, wall_x, wall_y),
        jnp.ones((rows, columns), dtype=jnp.bool_),
    )


def generate_one_wall(
    rows: int,
    columns: int,
    wall: bool = True,
    vertical: bool = True,
    wall_pos: int = None,
    door_pos: int = None,
):
    return jax.lax.select(
        wall,
        wall_maze(columns, rows, vertical, wall_pos, door_pos),
        jnp.ones((rows, columns), dtype=jnp.bool_),
    )


@flax.struct.dataclass
class Observations:
    shape: chex.Array


@flax.struct.dataclass
class EnvState:
    pos: chex.Array
    goals: chex.Array
    found: chex.Array
    right_order: bool
    grid_env: chex.Array
    time: int


@flax.struct.dataclass
class EnvParams:
    maze: chex.Array


#   wall: int = 1
#   wall_x: int = 1
#   wall_y: int = 1

# jax.tree_util.register_pytree_node(EnvParams, EnvParams.tree_flatten, EnvParams.tree_unflatten)


class GridWorldNew(environment.Environment):
    def __init__(
        self,
        rows=5,
        columns=5,
        num_rewards=2,
        max_steps_in_episode=30,
        last_reward_stays=False,
    ):
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.grid_indexes = (
            jnp.indices([rows, columns]).transpose(1, 2, 0).reshape(-1, 2)
        )
        self.num_rewards = num_rewards
        self.max_steps_in_episode = max_steps_in_episode
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.last_reward_stays = last_reward_stays
        reward = jnp.arange(self.num_rewards)
        self.reward_order = jnp.argsort(reward)

    @property
    def default_params(self) -> EnvParams:
        # default env parameters
        return EnvParams(jnp.ones((self.columns, self.rows), dtype=jnp.bool_))

    # @partial(jax.jit, static_argnames=['params'])
    # @jax.jit
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment.

        Returns: env_state, obsv, reward, done, info
        """
        # Move the agent
        p = state.pos + self.directions[action]
        px = jnp.clip(p[0], 0, self.rows - 1)
        py = jnp.clip(p[1], 0, self.columns - 1)
        in_map = state.grid_env[px, py]
        new_pos = jax.lax.select(in_map, jnp.array([px, py]), state.pos)

        # e.g. [[False, False], [True, True], [True, False]]
        pos_match = jnp.equal(state.goals, jnp.tile(p, [self.num_rewards, 1]))
        # e.g. [False, True, False]
        found_goal_arr = jax.numpy.all(pos_match, axis=1)
        # e.g. True
        found_goal = jnp.any(found_goal_arr)
        # e.g. 1
        found_goal_idx = jnp.nonzero(found_goal_arr, size=1)[0][0]
        found_before = jax.lax.select(
            found_goal,
            jnp.bool_(state.found[found_goal_idx]),
            jnp.array(True, dtype=jnp.bool_),
        )
        total_goals_found = jnp.uint8(state.found).sum()
        goal_reached = jnp.logical_and(found_goal, jnp.logical_not(found_before))
        goal_correct_order = jnp.logical_and(
            goal_reached, self.reward_order[found_goal_idx] == total_goals_found
        )

        # Update state dict and evaluate termination conditions
        new_found = jax.lax.select(
            goal_correct_order,
            get_new_found_array(found_goal_arr, state.found),
            state.found,
        )
        new_found = jax.lax.select(
            jnp.logical_and(
                total_goals_found == self.num_rewards - 1, self.last_reward_stays
            ),
            state.found,
            new_found,
        )
        new_state = EnvState(
            new_pos,
            state.goals,
            new_found,
            state.right_order,
            state.grid_env,
            state.time + 1,
        )
        done = self.is_terminal(state, params)
        obs = lax.stop_gradient(self.get_obs(new_state))
        return (
            obs,
            lax.stop_gradient(new_state),
            goal_correct_order.astype(jnp.float32),
            done,
            {},
        )

    # @partial(jax.jit, static_argnames=['params'])
    # @jax.jit
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment.

        Returns: state, obs
        """
        k1, k2 = jax.random.split(key)
        # grid_env = generate_walls(k1, self.rows, self.columns, params.wall, params.wall_x, params.wall_y)
        grid_env = params.maze
        goal_pos, agent_pos = sample_init_state(
            k2, grid_env, self.grid_indexes, self.num_rewards, params
        )
        rewards_found = jnp.array([False] * self.num_rewards)
        state = EnvState(agent_pos, goal_pos, rewards_found, True, grid_env, 0)
        # return self.get_obs(state), state
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        # wall = 0
        # agent = 1
        # rewards >= 2
        one_hot_enc_obs = jnp.zeros((self.columns, self.rows, self.num_rewards + 2))
        one_hot_enc_obs = one_hot_enc_obs.at[:, :, 0].set((1 - state.grid_env))
        one_hot_enc_obs = one_hot_enc_obs.at[
            state.goals[:, 0], state.goals[:, 1], jnp.arange(2, self.num_rewards + 2)
        ].set(jnp.logical_not(state.found).astype(int))
        one_hot_enc_obs = one_hot_enc_obs.at[state.pos[0], state.pos[1], 1].set(1.0)
        return one_hot_enc_obs.flatten()

    def get_obs_ints(self, state: EnvState) -> chex.Array:
        return jnp.concatenate(
            (state.pos, state.goals.flatten(), state.found.astype(int)), axis=-1
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done = jnp.all(state.found)

        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps_in_episode - 1
        done = jnp.logical_or(done, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "GridWorld"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            0,
            1,
            (self.columns * self.rows * (self.num_rewards + 2),),
            dtype=jnp.float32,
        )

    def observation_space_reward(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            0, 1, (self.columns * self.rows * (self.num_rewards + 2)), dtype=jnp.float32
        )

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)


def get_random_position(grid_env, grid_indexes, num_pos, rng):
    rng_prob, rng_choice = jax.random.split(rng, 2)
    prob = jax.random.uniform(rng_prob, grid_env.shape)
    prob = jnp.where(grid_env, prob, 0)
    return jax.random.choice(
        rng_choice, grid_indexes, shape=[num_pos], p=prob.reshape(-1), replace=False
    )


def sample_init_state(
    key: chex.PRNGKey,
    grid_env: chex.Array,
    grid_indexes: chex.Array,
    num_rewards: int,
    params: EnvParams,
) -> Tuple[chex.Array, chex.Array]:
    """Sample a new initial state."""
    pos_indexes = get_random_position(
        grid_env, grid_indexes, num_rewards + 1, key
    )
    # pos_index = jnp.array([2,0])
    # goal_indexes = get_random_position(grid_env, grid_indexes, num_rewards, next(rng))
    # goal_indexes = jnp.array([[0,2]])
    goal_indexes = pos_indexes[1:, :]
    return goal_indexes, pos_indexes[0, :]
