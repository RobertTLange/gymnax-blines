#!/bin/bash
# Loop over environments and train agents with PPO/ES
for env_name in CartPole-v1 Pendulum-v1 Acrobot-v1 MountainCar-v0 MountainCarContinuous-v0
do
    python train.py -config configs/$env_name/ppo.yaml
    python train.py -config configs/$env_name/es.yaml
done