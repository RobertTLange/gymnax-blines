if __name__ == "__main__":
    import jax
    import gymnax

    state_seq = []
    env_name = "Breakout-MinAtar"
    rng = jax.random.PRNGKey(0)
    env, env_params = gymnax.make(env_name)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    t_counter = 0
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        print(t_counter, reward, action, done)
        print(10 * "=")
        t_counter += 1
        if done or t_counter > 30:
            break
        else:
            env_state = next_env_state
            obs = next_obs

    from gymnax.visualize import Visualizer

    vis = Visualizer(env, env_params, state_seq)
    vis.animate(f"docs/test_{env_name}.gif")
