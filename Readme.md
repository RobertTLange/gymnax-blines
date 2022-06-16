# A Set of Training and Speed Evaluation Tools for `gymnax`

In this repository we provide training pipelines for `gymnax` agents in JAX. Furthermore, you can use the utilities to speed benchmark step transitions.

## Accelerated Training of Agents with `gymnax`

```
python main.py -config configs/<env_name>/ppo.yaml
python main.py -config configs/<env_name>/open_es.yaml
```

## Visualize `gymnax` Environment Rollouts

```
python visualize.py -env <env_name>
```

## Speed Up Evaluations for `gymnax`



```
python speed.py -env <env_name>
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