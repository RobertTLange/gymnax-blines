from functools import partial
import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from typing import Any, Callable, Tuple
import gymnax


class BatchManager:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        action_size,
        state_space,
    ):
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda

        try:
            self.state_shape = [state_space.shape[0]]
        except Exception:
            self.state_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *self.action_size),
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
            "states": jax.ops.index_update(
                buffer["states"], buffer["_p"], state
            ),
            "actions": jax.ops.index_update(
                buffer["actions"], buffer["_p"], action.squeeze()
            ),
            "rewards": jax.ops.index_update(
                buffer["rewards"], buffer["_p"], reward.squeeze()
            ),
            "dones": jax.ops.index_update(
                buffer["dones"], buffer["_p"], done.squeeze()
            ),
            "log_pis_old": jax.ops.index_update(
                buffer["log_pis_old"], buffer["_p"], log_pi
            ),
            "values_old": jax.ops.index_update(
                buffer["values_old"], buffer["_p"], value
            ),
            "_p": (buffer["_p"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"],
            reward=buffer["rewards"],
            done=buffer["dones"],
        )
        batch = (
            buffer["states"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]


class RolloutManager(object):
    def __init__(self, env_name, model, gen_vmap_dict=None):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = gymnax.make(env_name)
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape
        self.apply_fn = model.apply
        self.gen_vmap_dict = gen_vmap_dict
        self.select_action = self.select_action_ppo

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def gen_evaluate(self, rng, param_dicts, num_eval_steps, num_eval_envs):
        eval_fct = jax.vmap(
            self.batch_evaluate, in_axes=(None, self.gen_vmap_dict, None, None)
        )
        return eval_fct(rng, param_dicts, num_eval_steps, num_eval_envs)

    @partial(jax.jit, static_argnums=0)
    def select_action_ppo(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        value, pi = policy(train_state.apply_fn, train_state.params, obs, rng)
        rng, key = jax.random.split(rng)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0], rng

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def batch_evaluate(self, rng_input, train_state, num_eval_steps, num_envs):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, _, _, key = self.select_action(train_state, obs, rng_net)
            # action = jnp.nan_to_num(action, nan=0.0)
            next_o, next_s, reward, done, _ = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action,
            )
            carry, y = [next_o, next_s, train_state, rng], [reward, done]
            return carry, y

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [obs, state, train_state, rng_episode],
            [jnp.zeros((num_eval_steps, num_envs, 2))],
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        rewards, dones = scan_out[0], scan_out[1]
        rewards = rewards.reshape(num_eval_steps, num_envs, 1)
        dones = dones.reshape(num_eval_steps, num_envs, 1)
        ep_mask = (jnp.cumsum(dones, axis=0) < 1).reshape(
            num_eval_steps, num_envs, 1
        )
        return jnp.mean(jnp.sum(rewards * ep_mask, axis=0))


@partial(jax.jit, static_argnums=0)
def policy(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    obs: jnp.ndarray,
    rng,
):
    value, pi = apply_fn(params, obs, rng)
    return value, pi
