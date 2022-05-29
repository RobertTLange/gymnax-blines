import jax
from train.models import get_model_ready


def main(config, log):
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config)

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from train.es import train_es as train_fn
    elif config.train_type == "PPO":
        from train.ppo import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type ('ES', 'PPO').")

    # Log and store the results.
    train_fn(rng, config, model, params)


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/cartpole_ppo.yaml")
    main(mle.train_config, mle.log)
