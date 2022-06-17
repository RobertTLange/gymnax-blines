import os
import jax
import gymnax
from utils.models import get_model_ready
from utils.helpers import load_pkl_object, load_config
from gymnax.visualize import Visualizer


def load_neural_network(config, agent_path):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config)

    params = load_pkl_object(agent_path)["network"]
    return model, params


def rollout_episode(env, env_params, model, model_params):
    state_seq = []
    rng = jax.random.PRNGKey(0)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)

    if model is not None:
        if model.model_name == "LSTM":
            hidden = model.initialize_carry()

    t_counter = 0
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        # action = int(network(obs).max(1)[1].view(1, 1).squeeze())
        if model is not None:
            if model.model_name == "LSTM":
                hidden, action = model.apply(model_params, obs, hidden, rng_act)
            else:
                action = model.apply(model_params, obs, rng_act)
        else:
            action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        print(t_counter, reward, action, done)
        print(10 * "=")
        t_counter += 1
        if done:
            break
        else:
            env_state = next_env_state
            obs = next_obs
    return state_seq


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train",
        "--train_type",
        type=str,
        default="es",
        help="es/ppo trained agent.",
    )
    parser.add_argument(
        "-random",
        "--random",
        action="store_true",
        default=False,
        help="Random rollout.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name.",
    )
    args, _ = parser.parse_known_args()

    # "Breakout-MinAtar"  # "PointRobot-misc"  # "MetaMaze-misc"
    if not args.random:
        base = f"agents/{args.env_name}/{args.train_type}"
        configs = load_config(base + ".yaml")
        model, model_params = load_neural_network(
            configs.train_config, base + ".pkl"
        )

        # import gym
        # import torch
        # from evaluate.dqn_torch import QNetwork

        # data_and_weights = torch.load(
        #     "evaluate/minatar_torch_ckpt/breakout_data_and_weights",
        #     map_location=lambda storage, loc: storage,
        # )

        # returns = data_and_weights["returns"]
        # network_params = data_and_weights["policy_net_state_dict"]

        # env = gym.make("MinAtar/Breakout-v0")
        # in_channels = env.game.n_channels
        # num_actions = env.game.num_actions()

        # network = QNetwork(in_channels, num_actions)
        # network.load_state_dict(network_params)
    else:
        model, model_params = None, None
    env, env_params = gymnax.make(args.env_name)
    state_seq = rollout_episode(env, env_params, model, model_params)
    vis = Visualizer(env, env_params, state_seq)
    vis.animate(f"docs/test_{args.env_name}.gif")
