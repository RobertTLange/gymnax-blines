import jax
import jax.numpy as jnp
import numpy as np
import gym


def rollout_gym(
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
    if env_name in [
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        "Seaquest-MinAtar",
        "SpaceInvaders-MinAtar",
    ]:
        env_base = env_name.split("-")[0]
        # v1 is version with minimal action set
        env = gym.make(f"MinAtar/{env_base}-v0")
    else:
        env = gym.make(env_name)

    ep_returns = []
    for _ in range(num_episodes):
        ep_ret = 0
        obs = env.reset()
        while True:
            rng, rng_act = jax.random.split(rng, 2)
            if RANDOM:
                action = env.action_space.sample()
            elif TORCH:
                action = int(network(obs).max(1)[1].view(1, 1).squeeze())
            elif JAX:
                action = network.apply(params, jnp.array(obs), rng_act)
            next_obs, reward, done, info = env.step(action)
            ep_ret += reward
            if done:
                ep_returns.append(ep_ret)
                break
            else:
                obs = next_obs

    return np.mean(ep_returns), np.std(ep_returns)


if __name__ == "__main__":
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
    mean, std = rollout_gym(
        "Breakout-MinAtar", network, None, 10, False, True, False
    )
    print(mean, std)
