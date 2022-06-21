import jax
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object


def main(config, mle_log, log_ext=""):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config)

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from utils.es import train_es as train_fn
    elif config.train_type == "PPO":
        from utils.ppo import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type. Has to be in ('ES', 'PPO').")

    # Log and store the results.
    log_steps, log_return, network_ckpt = train_fn(
        rng, config, model, params, mle_log
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "network": network_ckpt,
        "train_config": config,
    }

    save_pkl_object(
        data_to_store,
        f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}.pkl",
    )


if __name__ == "__main__":
    # Use MLE-Infrastructure if available (e.g. for parameter search)
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/cartpole/ppo.yaml")
    #     main(mle.train_config, mle_log=mle.log)
    # # Otherwise use simple logging and config loading
    # except Exception:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="configs/CartPole-v1/ppo.yaml",
        help="Path to configuration yaml.",
    )
    parser.add_argument(
        "-seed",
        "--seed_id",
        type=int,
        default=0,
        help="Random seed of experiment.",
    )
    parser.add_argument(
        "-lr",
        "--lrate",
        type=float,
        default=5e-04,
        help="Random seed of experiment.",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_id, args.lrate)
    main(
        config.train_config,
        mle_log=None,
        log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
    )
