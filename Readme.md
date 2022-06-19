# Training & Speed Evaluation Tools for [`gymnax`](https://github.com/RobertTLange/gymnax)

<a href="https://github.com/RobertTLange/gymnax-blines/blob/main/docs/logo.png?raw=true"><img src="https://github.com/RobertTLange/gymnax-blines/blob/main/docs/logo.png?raw=true" width="215" align="right" /></a>


In this repository we provide training pipelines for [`gymnax`](https://github.com/RobertTLange/gymnax) agents in JAX. Furthermore, you can use the utilities to speed benchmark step transitions.

## Accelerated Training of Agents with [`gymnax`](https://github.com/RobertTLange/gymnax)

We provide training routines and the corresponding checkpoints for both Evolution Strategies (mostly OpenES using [`evosax`](https://github.com/RobertTLange/evosax) and PPO. You can train the agents as follows: 

```
python train.py -config configs/<env_name>/ppo.yaml
python train.py -config configs/<env_name>/es.yaml
```

## Visualize [`gymnax`](https://github.com/RobertTLange/gymnax) Environment Rollouts

You can also generate GIF visualizations of the agent's behaviour as follows:

```
python visualize.py -env <env_name>
```

Note that we do not support visualizations for most behavior suite environments, since they do not lend themselves to visual display (e.g. markov reward process, etc.).

|  |  |  |  | 
| --- | --- | --- | --- |
| ![](docs/Pendulum-v1.gif) | ![](docs/Acrobot-v1.gif)  | ![](docs/CartPole-v1.gif) | ![](docs/MountainCarContinuous-v0.gif) |
| ![](docs/Catch-bsuite.gif) | ![](docs/FourRooms-misc.gif)  | ![](docs/MetaMaze-misc.gif) | ![](docs/PointRobot-misc.gif) |

## Speed Up Evaluations for [`gymnax`](https://github.com/RobertTLange/gymnax)

Finally, we provide simple tools for benchmarking the speed of step transitions on different hardware (CPU, GPU, TPU). For a specific environment the speed estimates can be obtained as follows:

```
python speed.py -env <env_name> --use_gpu --use_network --num_envs 10
```

For each environment we estimate the seconds required to execute 1 Mio steps on various hardware and for both random (R)/neural network (N) policies. We report the mean over 10 independent runs:

| Environment Name | `np` <br /> CPU <br /> 10 Envs | `jax` <br /> CPU <br /> 10 Envs | `np` <br /> CPU <br /> 40 Envs | `jax` <br /> CPU <br /> 40 Envs | `jax` <br /> 2080Ti <br /> 2k Envs | `jax` <br /> V100s <br /> 10k Envs | `jax` <br /> v3-8 <br /> 20k Envs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`Acrobot-v1`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/acrobot.py) | R: 72 <br /> N: 118 | R: 1.16 <br /> N: 2.90 | R: 66 <br /> N: 77.2 | R: 1.03 <br /> N: 2.3 | R: 0.06 <br /> N: 0.08
| [`Pendulum-v1`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/pendulum.py) | R: 139 <br /> N: 118 | R: 0.41 <br /> N: 1.41 | R: 118 <br /> N: 70 | R: 0.32 <br /> N: 1.07 | R: 0.07 <br /> N: 0.09
| [`CartPole-v1`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/cartpole.py) | R: 46 <br /> N: 94 | R: 0.43 <br /> N: 1.42 | R: 45 <br /> N: 51 | R: 0.49 <br /> N: 0.98 | R: 0.06 <br /> N: 0.08
| [`MountainCar-v0`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/mountain_car.py) | R: 48 <br /> N: 105 | R: 0.53 <br /> N: 1.49 | R: 54 <br /> N: 57 | R: 0.39 <br /> N: 1.03 | R: 0.06 <br /> N: 0.09
| [`MountainCarContinuous-v0`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/continuous_mountain_car.py) | R: 117 <br /> N: 99 | R: 0.34 <br /> N: 1.11 | R: 98 <br /> N: 53 | R: 0.28 <br /> N: 0.85 | R: 0.09 <br /> N: 0.12
| --- | --- | --- | --- | ---| --- | --- | --- |
| [`Catch-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/catch.py) |
| [`DeepSea-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/deep_sea.py) |
| [`MemoryChain-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/memory_chain.py) |
| [`UmbrellaChain-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/umbrella_chain.py) |
| [`DiscountingChain-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/discounting_chain.py) |
| [`MNISTBandit-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/mnist.py) |
| [`SimpleBandit-bsuite`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/bandit.py) |
| --- | --- | --- | --- | ---| --- | --- | --- |
| [`Asterix-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/asterix.py) |
| [`Breakout-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/breakout.py) |
| [`Freeway-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/freeway.py) |
| [`Seaquest-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/seaquest.py) |
| [`SpaceInvaders-MinAtar`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/space_invaders.py) |
| --- | --- | --- | --- | ---| --- | --- | --- |
| [`FourRooms-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/rooms.py) |
| [`MetaMaze-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/meta_maze.py) |
| [`PointRobot-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/point_robot.py) |
| [`BernoulliBandit-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/bernoulli_bandit.py) |
| [`GaussianBandit-misc`](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/gaussian_bandit.py) |

TPU comparisons will follow once I (or you?) find the time.

## Running Grid Search with the `MLE-Infrastructure`

```
pip install mle-toolbox
mle run ...
```

## MinAtar Comparison with Torch DQN

In order to make sure that we fully replicate the world dynamics of the NumPy based MinAtar API, we provide different checks:

- Transition test assertions from random and trained behaviour.
- Performance test from DQN agents trained on the NumPy environment in the JAX environment.
- Performance test from DQN agents trained on the JAX environment in the NumPy environment.

We first replicate train the DQN agents using the code provided in the MinAtar repository without any parameter tuning. E.g.

```
cd examples
python dqn.py -g breakout -v -s
```