import gym
import numpy as np

from utils.vec_env.parallel import DummyVecEnv, SubprocVecEnv

from bsuite.environments import (
    catch,
    deep_sea,
    discounting_chain,
    memory_chain,
    umbrella_chain,
    mnist,
    bandit,
)


def make_env(env_name: str):
    if env_name in [
        "Catch-bsuite",
        "DeepSea-bsuite",
        "DiscountingChain-bsuite",
        "MemoryChain-bsuite",
        "UmbrellaChain-bsuite",
        "MNISTBandit-bsuite",
        "SimpleBandit-bsuite",
    ]:
        env = make_bsuite_env(env_name)
    elif env_name in [
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        "Seaquest-MinAtar",
        "SpaceInvaders-MinAtar",
    ]:
        env = make_minatar_env(env_name)
    else:
        env = gym.make(env_name)
    return env


def make_bsuite_env(bsuite_env_name: str):
    """Boilerplate helper for bsuite env generation."""
    if bsuite_env_name == "Catch-bsuite":
        env = catch.Catch()
    elif bsuite_env_name == "DeepSea-bsuite":
        env = deep_sea.DeepSea(size=8, randomize_actions=False)
    elif bsuite_env_name == "DiscountingChain-bsuite":
        env = discounting_chain.DiscountingChain(mapping_seed=0)
    elif bsuite_env_name == "MemoryChain-bsuite":
        env = memory_chain.MemoryChain(memory_length=5, num_bits=1)
    elif bsuite_env_name == "UmbrellaChain-bsuite":
        env = umbrella_chain.UmbrellaChain(chain_length=10, n_distractor=0)
    elif bsuite_env_name == "MNISTBandit-bsuite":
        env = mnist.MNISTBandit()
    elif bsuite_env_name == "SimpleBandit-bsuite":
        env = bandit.SimpleBandit()
    return env


def make_minatar_env(minatar_env_name: str):
    """Boilerplate helper for bsuite env generation."""
    if minatar_env_name == "Asterix-MinAtar":
        env = gym.make("MinAtar/Asterix-v1")
    elif minatar_env_name == "Breakout-MinAtar":
        env = gym.make("MinAtar/Breakout-v1")
    elif minatar_env_name == "Freeway-MinAtar":
        env = gym.make("MinAtar/Freeway-v1")
    elif minatar_env_name == "Seaquest-MinAtar":
        env = gym.make("MinAtar/Seaquest-v1")
    elif minatar_env_name == "SpaceInvaders-MinAtar":
        env = gym.make("MinAtar/SpaceInvaders-v1")
    return env


def make_parallel_env(env_id, seed, n_rollout_threads):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
