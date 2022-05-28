import gymnax
import jax
import jax.numpy as jnp
import tqdm
from evosax import Strategies, ESLog, FitnessShaper, ParameterReshaper


def train_es(rng, config, model, params):
    train_param_reshaper = ParameterReshaper(params)
    test_param_reshaper = ParameterReshaper(params, n_devices=1)

    train_evaluator.set_apply_fn(train_param_reshaper.vmap_dict, model.apply)
    test_evaluator.set_apply_fn(test_param_reshaper.vmap_dict, model.apply)
    fit_shaper = FitnessShaper(**config.fitness_config.toDict())

    # Setup the evolution strategy -> only with non-masked params to optimize
    strategy = Strategies[config.es_name](
        num_dims=train_param_reshaper.params_to_opt,
        **config.es_config.toDict(),
    )

    es_params = strategy.default_params
    for k, v in config.es_params.toDict().items():
        es_params[k] = v
    es_state = strategy.initialize(rng, es_params)
    best_perf_yet, best_model_ckpt = -jnp.inf, None

    t = tqdm.tqdm(
        range(1, config.num_generations + 1), desc="ES Run", leave=True
    )
    es_logging = ESLog(
        train_param_reshaper.params_to_opt,
        config.num_generations,
        top_k=5,
        maximize=config.fitness_config.maximize,
    )
    es_log = es_logging.initialize()

    for gen in t:
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        # Ask for new parameter proposals and reshape into paramter trees.
        x, es_state = strategy.ask(rng_ask, es_state, es_params)
        reshaped_params = train_param_reshaper.to_mask(x)

        # Rollout population performance, reshape fitness & update strategy.
        fitness = train_evaluator.rollout(rng_eval, reshaped_params)
        # Separate loss/acc when evolving classifier
        if type(fitness) == tuple:
            fitness_to_opt, fitness_to_log = fitness[0], fitness[1]
        else:
            fitness_to_opt, fitness_to_log = fitness, fitness
        fit_re = fit_shaper.apply(x, fitness_to_opt.mean(axis=1))
        es_state = strategy.tell(x, fit_re, es_state, es_params)
        # Update the logging instance.
        es_log = es_logging.update(es_log, x, fitness_to_log.mean(axis=1))

        # Sporadically evaluate the mean & best performance on test evaluator.
        if (gen + 1) % config.evaluate_every_gen == 0:
            rng, rng_test = jax.random.split(rng)
            # Stack best params seen & mean strategy params for eval
            best_params = es_log["top_params"][0]
            mean_params = es_state["mean"]
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.to_mask(x_test)
            test_fitness = test_evaluator.rollout(
                rng_test, reshaped_test_params
            )

            # Separate loss/acc when evolving classifier
            if type(test_fitness) == tuple:
                test_fitness_to_log = test_fitness[1].mean(axis=1)
            else:
                test_fitness_to_log = test_fitness.mean(axis=1)

            if test_fitness_to_log[1] > best_perf_yet:
                best_perf_yet = test_fitness_to_log[1]
                best_model_ckpt = train_param_reshaper.flat_to_mask(
                    es_state["mean"]
                )

        t.set_description("R " + str(best_perf_yet))
        t.refresh()
    return (
        {"params": best_model_ckpt},
        best_perf_yet,
        {"params": train_param_reshaper.flat_to_mask(es_state["mean"])},
        test_fitness_to_log[1],
    )
