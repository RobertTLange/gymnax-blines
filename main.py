from train.es import train_es
from train.ppo import train_ppo
from evosax import NetworkMapper


def main(config, log):
    # Setup the model architecture

    # Run the training loop

    # Log and store the results.

if __name__ == "__main__":
    from mle_toolbox.utils import get_jax_os_ready
    from mle_toolbox import MLExperiment

    get_jax_os_ready(
        num_devices=1,
        device_type="gpu",
    )

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/cartpole_ppo.yaml")
    main(mle.train_config, mle.log)
