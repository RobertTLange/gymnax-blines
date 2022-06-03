import jax
import jax.numpy as jnp
from evosax import (
    ProblemMapper,
    Strategies,
    ESLog,
    FitnessShaper,
    ParameterReshaper,
)


def train_es(rng, config, log, model, params):
    """Evolve a policy for a gymnax environment"""
    # Setup evaluation wrappers from evosax - uses gymnax backend!
    train_evaluator = ProblemMapper["Gym"](
        **config.problem_train_config.toDict(),
        env_name=config.env_name,
        test=False,
    )
    test_evaluator = ProblemMapper["Gym"](
        **config.problem_test_config.toDict(),
        env_name=config.env_name,
        n_devices=1,
        test=True,
    )

    # Set up model reshaping tools (from flat vector to network param dict)
    train_param_reshaper = ParameterReshaper(params)
    test_param_reshaper = ParameterReshaper(params, n_devices=1)

    # Augment the evaluation wrappers for batch (pop/mc) evaluation
    if config.network_name != "LSTM":
        train_evaluator.set_apply_fn(
            train_param_reshaper.vmap_dict, model.apply
        )
        test_evaluator.set_apply_fn(test_param_reshaper.vmap_dict, model.apply)
    else:
        train_evaluator.set_apply_fn(
            train_param_reshaper.vmap_dict, model.apply, model.initialize_carry
        )
        test_evaluator.set_apply_fn(
            test_param_reshaper.vmap_dict, model.apply, model.initialize_carry
        )

    # Set up the fitness shaping utility (max/centered ranks, etc.)
    fit_shaper = FitnessShaper(**config.fitness_config.toDict())

    # Setup the evolution strategy
    strategy = Strategies[config.es_name](
        num_dims=train_param_reshaper.total_params,
        **config.es_config.toDict(),
    )

    es_params = strategy.default_params
    for k, v in config.es_params.toDict().items():
        es_params[k] = v
    es_state = strategy.initialize(rng, es_params)
    best_perf_yet, best_model_ckpt = -jnp.inf, None

    es_logging = ESLog(
        train_param_reshaper.total_params,
        config.num_generations,
        top_k=5,
        maximize=config.fitness_config.maximize,
    )
    es_log = es_logging.initialize()

    for gen in range(config.num_generations):
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        # Ask for new parameter proposals and reshape into paramter trees.
        x, es_state = strategy.ask(rng_ask, es_state, es_params)
        reshaped_params = train_param_reshaper.reshape(x)

        # Rollout population performance, reshape fitness & update strategy.
        fitness = train_evaluator.rollout(rng_eval, reshaped_params).mean(
            axis=1
        )

        # Separate loss/acc when evolving classifier
        fit_re = fit_shaper.apply(x, fitness)
        es_state = strategy.tell(x, fit_re, es_state, es_params)
        # Update the logging instance.
        es_log = es_logging.update(es_log, x, fitness)

        # Sporadically evaluate the mean & best performance on test evaluator.
        if (gen + 1) % config.evaluate_every_gen == 0:
            rng, rng_test = jax.random.split(rng)
            # Stack best params seen & mean strategy params for eval
            best_params = es_log["top_params"][0]
            mean_params = es_state["mean"]
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)
            test_fitness = test_evaluator.rollout(
                rng_test, reshaped_test_params
            )

            test_fitness_to_log = test_fitness.mean(axis=1)

            if test_fitness_to_log[1] > best_perf_yet:
                best_perf_yet = test_fitness_to_log[1]
                best_model_ckpt = train_param_reshaper.reshape_single(
                    es_state["mean"]
                )

            log.update(
                {"num_steps": gen + 1},
                {"return": test_fitness_to_log[1]},
                model={
                    "params": train_param_reshaper.reshape_single(
                        es_state["mean"]
                    )
                },
                save=True,
            )
    return (
        {"params": best_model_ckpt},
        best_perf_yet,
        {"params": train_param_reshaper.reshape_single(es_state["mean"])},
        test_fitness_to_log[1],
    )
