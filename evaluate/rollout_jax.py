import jax
import gymnax
import numpy as np


def rollout_jax(
    env_name,
    network,
    params,
    num_episodes=10,
    RANDOM=True,
    TORCH=False,
    JAX=False,
):
    """Rollout a set of episodes in JAX environment (random/network)"""
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make(env_name, use_minimal_action_set=False)
    ep_returns = []
    for _ in range(num_episodes):
        ep_ret = 0
        rng, rng_reset = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)
        while True:
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            if RANDOM:
                action = env.action_space(env_params).sample(rng_act)
            elif TORCH:
                action = int(network(obs).max(1)[1].view(1, 1).squeeze())
            elif JAX:
                action = network.apply(params, obs, rng_act)
            next_obs, next_env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )
            ep_ret += reward
            if done:
                ep_returns.append(ep_ret)
                break
            else:
                env_state = next_env_state
                obs = next_obs

    return np.mean(ep_returns), np.std(ep_returns)


if __name__ == "__main__":
    import gym
    import torch
    from dqn_torch import QNetwork

    data_and_weights = torch.load(
        "minatar_torch_ckpt/breakout_data_and_weights",
        map_location=lambda storage, loc: storage,
    )

    returns = data_and_weights["returns"]
    network_params = data_and_weights["policy_net_state_dict"]

    env = gym.make("MinAtar/Breakout-v0")
    in_channels = env.game.n_channels
    num_actions = env.game.num_actions()

    network = QNetwork(in_channels, num_actions)
    network.load_state_dict(network_params)
    mean, std = rollout_jax(
        "Breakout-MinAtar", network, None, 50, False, True, True
    )
    print(mean, std)
