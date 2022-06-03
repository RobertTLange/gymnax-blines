import jax
import jax.numpy as jnp
import gymnax


def main(env_name: str):
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make(env_name)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        print(reward, action)
        if done:
            break
        else:
            env_state = next_env_state
            obs = next_obs

    @jax.jit
    def scan_episode(rng):
        # Initialize bandit and fwp weight state
        rng, rng_reset = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)

        def step(state_input, tmp):
            rng, obs, env_state = state_input
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            action = env.action_space(env_params).sample(rng_act)
            next_obs, next_env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )
            carry, out = [
                rng,
                next_obs,
                next_env_state,
            ], reward
            return carry, out

        _, scan_out = jax.lax.scan(
            step,
            [rng, obs, env_state],
            [jnp.zeros((env_params.max_steps_in_episode,))],
        )
        cum_return = scan_out.sum()
        return cum_return

    cum_return = scan_episode(rng)
    print(cum_return)


if __name__ == "__main__":
    main("DeepSea-bsuite")
