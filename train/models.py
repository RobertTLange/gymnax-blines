import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
from evosax import NetworkMapper
import gymnax


def get_model_ready(rng, config):
    """Instantiate a model according to obs shape of environment."""
    # Get number of desired output units
    env, env_params = gymnax.make(config.env_name)

    # Instantiate model class (flax-based)
    if config.train_type == "ES":
        model = NetworkMapper[config.network_name](
            **config.network_config, num_output_units=env.num_actions
        )
    elif config.train_type == "PPO":
        if config.network_name == "Categorial-MLP":
            model = CategoricalSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )
        elif config.network_name == "Gaussian-MLP":
            model = GaussianSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )

    # Initialize the network based on the observation shape
    obs_shape = env.observation_space(env_params).shape
    params = model.init(rng, jnp.zeros(obs_shape), rng=rng)
    return model, params


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class CategoricalSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    model_name: str = "separate-mlp"

    @nn.compact
    def __call__(self, x, rng):
        x_v = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_critic + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        logits = nn.Dense(
            self.num_output_units,
            bias_init=default_mlp_init(),
        )(x_a)
        # pi = distrax.Categorical(logits=logits)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi


class GaussianSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-mlp"

    @nn.compact
    def __call__(self, x, rng):
        x_v = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_critic + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_actor + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_actor + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        mu = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_mu",
            bias_init=default_mlp_init(),
        )(x_a)
        log_scale = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_scale",
            bias_init=default_mlp_init(),
        )(x_a)
        scale = jax.nn.softplus(log_scale) + self.min_std
        pi = tfp.distributions.MultivariateNormalDiag(mu, scale)
        return v, pi
