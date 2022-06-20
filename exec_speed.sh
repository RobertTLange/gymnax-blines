#!/bin/bash
# Loop over environments and estimate runtime speed
for env_name in CartPole-v1 Pendulum-v1 Acrobot-v1 MountainCar-v0 MountainCarContinuous-v0
do
    python speed.py -env $env_name -np -n_envs 10
    python speed.py -env $env_name -np -n_envs 10 -net
    python speed.py -env $env_name -n_envs 10
    python speed.py -env $env_name -n_envs 10 -net
    python speed.py -env $env_name -np -n_envs 40
    python speed.py -env $env_name -np -n_envs 40 -net
    python speed.py -env $env_name -n_envs 40
    python speed.py -env $env_name -n_envs 40 -net
    python speed.py -env $env_name -n_envs 2000 -gpu
    python speed.py -env $env_name -n_envs 2000 -net -gpu
done

# TODO: Add bsuite numpy evaluation
# TODO: Add minatar numpy evaluation
# TODO: Add V100S results and TPU evaluation