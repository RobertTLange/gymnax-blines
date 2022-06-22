#!/bin/bash
export gym_envs="
CartPole-v1
Pendulum-v1
Acrobot-v1
MountainCar-v0
MountainCarContinuous-v0
"

export minatar_envs="
Asterix-MinAtar
Breakout-MinAtar
Freeway-MinAtar
SpaceInvaders-MinAtar
"
#Seaquest-MinAtar

export bsuite_envs="
Catch-bsuite
DeepSea-bsuite
MemoryChain-bsuite
UmbrellaChain-bsuite
DiscountingChain-bsuite
MNISTBandit-bsuite
SimpleBandit-bsuite
"

export misc_envs="
FourRooms-misc
MetaMaze-misc
PointRobot-misc
BernoulliBandit-misc
GaussianBandit-misc
"

if [[ "$1" == "speed" ]]
then
    # Loop over environments and estimate runtime speed
    for env_name in $minatar_envs $gym_envs
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
        python speed.py -env $env_name -n_envs 2000 -gpu -a100
        python speed.py -env $env_name -n_envs 2000 -net -gpu -a100
    done

    for env_name in $bsuite_envs $misc_envs
    do
        python speed.py -env $env_name -n_envs 10
        python speed.py -env $env_name -n_envs 10 -net
        python speed.py -env $env_name -n_envs 40
        python speed.py -env $env_name -n_envs 40 -net
        python speed.py -env $env_name -n_envs 2000 -gpu
        python speed.py -env $env_name -n_envs 2000 -net -gpu
        python speed.py -env $env_name -n_envs 2000 -gpu -a100
        python speed.py -env $env_name -n_envs 2000 -net -gpu -a100
    done

    # TODO: Add bsuite numpy evaluation
    # TODO: Add V100S results and TPU evaluation
elif [[ "$1" == "train" ]]
then
    # Loop over environments and train agents with PPO/ES
    for env_name in $gym_envs $minatar_envs $bsuite_envs
    do
        python train.py -config agents/$env_name/ppo.yaml --lrate 1e-04
        python train.py -config agents/$env_name/ppo.yaml --lrate 1e-03
        python train.py -config agents/$env_name/ppo.yaml
        python train.py -config agents/$env_name/es.yaml
    done

    for env_name in $misc_envs
    do
        python train.py -config configs/$env_name/es.yaml
    done

elif [[ "$1" == "visualize" ]]
then
    # Loop over environments and visualize trained PPO agent
    for env_name in $gym_envs $minatar_envs
    do
        python visualize.py -env $env_name -train ppo
    done
else
    echo "Provide valid argument to bash script"
fi