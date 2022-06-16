import os
import jax
from train.models import get_model_ready
from mle_toolbox.utils import load_pkl_object, load_job_config


def load_neural_network(config, experiment_dir):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config)

    params = load_pkl_object(
        os.path.join(experiment_dir, "models/final/final_seed_0.pkl")
    )
    return model, params["params"]


if __name__ == "__main__":
    import jax
    import gymnax

    RANDOM = False
    JAX = False
    TORCH = True
    env_name = "Breakout-MinAtar"  # "PointRobot-misc"  # "MetaMaze-misc"
    if JAX:
        experiment_dir = "experiments/open_es"
        configs = load_job_config(
            os.path.join(experiment_dir, "open_es.yaml"), experiment_dir
        )
        model, params = load_neural_network(configs[0], experiment_dir)

        if configs[0].network_name == "LSTM":
            hidden = model.initialize_carry()
    elif TORCH:
        import gym
        import torch
        from evaluate.dqn_torch import QNetwork

        data_and_weights = torch.load(
            "evaluate/minatar_torch_ckpt/breakout_data_and_weights",
            map_location=lambda storage, loc: storage,
        )

        returns = data_and_weights["returns"]
        network_params = data_and_weights["policy_net_state_dict"]

        env = gym.make("MinAtar/Breakout-v0")
        in_channels = env.game.n_channels
        num_actions = env.game.num_actions()

        network = QNetwork(in_channels, num_actions)
        network.load_state_dict(network_params)

    state_seq = []
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make(env_name)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    t_counter = 0
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if RANDOM:
            action = env.action_space(env_params).sample(rng_act)
        elif TORCH:
            action = int(network(obs).max(1)[1].view(1, 1).squeeze())
        elif JAX:
            if configs[0].network_name == "LSTM":
                hidden, action = model.apply(params, obs, hidden, rng_act)
            else:
                action = model.apply(params, obs, rng_act)

        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        print(t_counter, reward, action, done)
        print(10 * "=")
        t_counter += 1
        if done or t_counter > 100:
            break
        else:
            env_state = next_env_state
            obs = next_obs

    from gymnax.visualize import Visualizer

    vis = Visualizer(env, env_params, state_seq)
    vis.animate(f"docs/test_{env_name}.gif")
