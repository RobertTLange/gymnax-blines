import jax
import jax.numpy as jnp
import numpy as np
import gym


def rollout_gym(env_name, network, params, num_episodes=10, random=False):
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
        env = gym.make(f"MinAtar/{env_base}-v1")
    else:
        env = gym.make(env_name)

    ep_returns = []
    for ep in range(num_episodes):
        ep_ret = 0
        obs = env.reset()
        while True:
            rng, rng_act = jax.random.split(rng, 2)
            if random:
                action = env.action_space.sample()
            else:
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
    mean, std = rollout_gym("Breakout-MinAtar", None, None, 50, True)
    print(mean, std)
