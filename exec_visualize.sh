#!/bin/bash
# Loop over environments and visualize trained PPO agent
for env_name in CartPole-v1 Pendulum-v1 Acrobot-v1 MountainCar-v0 MountainCarContinuous-v0
do
    python visualize.py -env $env_name -train ppo
done