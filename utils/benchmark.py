import jax
import time
import numpy as np
from gymnax.experimental import RolloutWrapper
from utils.vec_env.wrapper import make_parallel_env


def speed_gymnax_random(env_name, num_env_steps, num_envs, rng, env_kwargs):
    """Random episode rollout in gymnax."""
    # Define rollout manager for env
    if env_name == "Freeway-MinAtar":
        env_params = {"max_steps_in_episode": 1000}
    else:
        env_params = {}
    manager = RolloutWrapper(
        None, env_name=env_name, env_params=env_params, env_kwargs=env_kwargs
    )

    # Multiple rollouts for same network (different rng, e.g. eval)
    rng, rng_batch = jax.random.split(rng)
    rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
    if num_envs == 1:
        rollout_fn = manager.single_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(
            rng_batch_eval, None
        )
        steps_per_batch = obs.shape[0]
    else:
        rollout_fn = manager.batch_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(
            rng_batch_eval, None
        )
        steps_per_batch = obs.shape[0] * obs.shape[1]
    step_counter = 0

    start_t = time.time()
    # Loop over batch/single episode rollouts until steps are collected
    while step_counter < num_env_steps:
        rng, rng_batch = jax.random.split(rng)
        rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
        _ = rollout_fn(rng_batch_eval, None)
        step_counter += steps_per_batch
    return time.time() - start_t


def speed_gymnax_network(
    env_name, num_env_steps, num_envs, rng, model, model_params, env_kwargs
):
    """MLP episode rollout in gymnax."""
    # Define rollout manager for env
    if env_name == "Freeway-MinAtar":
        env_params = {"max_steps_in_episode": 1000}
    else:
        env_params = {}
    manager = RolloutWrapper(
        model.apply,
        env_name=env_name,
        env_params=env_params,
        env_kwargs=env_kwargs,
    )

    # Multiple rollouts for same network (different rng, e.g. eval)
    rng, rng_batch = jax.random.split(rng)
    rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
    if num_envs == 1:
        rollout_fn = manager.single_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(
            rng_batch_eval, model_params
        )
        steps_per_batch = obs.shape[0]
    else:
        rollout_fn = manager.batch_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(
            rng_batch_eval, model_params
        )
        steps_per_batch = obs.shape[0] * obs.shape[1]
    step_counter = 0

    start_t = time.time()
    # Loop over batch/single episode rollouts until steps are collected
    while step_counter < num_env_steps:
        rng, rng_batch = jax.random.split(rng)
        rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
        _ = rollout_fn(rng_batch_eval, model_params)
        step_counter += steps_per_batch
    return time.time() - start_t


def speed_numpy_random(env_name, num_env_steps, num_envs):
    """Random episode rollout in numpy."""
    envs = make_parallel_env(env_name, seed=0, n_rollout_threads=num_envs)
    envs.reset()
    num_batch_steps = int(np.ceil(num_env_steps / num_envs))
    start_t = time.time()
    for i in range(num_batch_steps):
        action = [envs.action_space.sample() for e in range(num_envs)]
        obs, reward, done, _ = envs.step(action)
        # NOTE: Automatic reset taken care of by wrapper!
    return time.time() - start_t


def speed_numpy_network(
    env_name, num_env_steps, num_envs, rng, model, model_params
):
    """MLP episode rollout in numpy."""
    envs = make_parallel_env(env_name, seed=0, n_rollout_threads=num_envs)
    obs = envs.reset()
    num_batch_steps = int(np.ceil(num_env_steps / num_envs))

    if num_envs == 1:
        apply_fn = jax.jit(model.apply)
    else:
        apply_fn = jax.jit(jax.vmap(model.apply, in_axes=(None, 0, 0)))

    # Run once to compile apply function!
    rng, rng_batch = jax.random.split(rng)
    rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
    if env_name in ["Pendulum-v1", "MountainCarContinuous-v0"]:
        action = apply_fn(model_params, obs, rng_batch_eval).reshape(
            num_envs, 1
        )
    else:
        action = apply_fn(model_params, obs, rng_batch_eval).reshape(num_envs)

    start_t = time.time()
    for i in range(num_batch_steps):
        rng, rng_batch = jax.random.split(rng)
        rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()

        if env_name in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            action = apply_fn(model_params, obs, rng_batch_eval).reshape(
                num_envs, 1
            )
        else:
            action = apply_fn(model_params, obs, rng_batch_eval).reshape(
                num_envs
            )
        obs, reward, done, _ = envs.step(action.tolist())
        # NOTE: Automatic reset taken care of by wrapper!
    return time.time() - start_t
