# Deep learning implementations

Readable implementations, from neural networks to robot policies. The explanations, paper notes, knowledge graph, and interactive visualizations live on the [Field Notes site](https://ishan8840.github.io/deep-learning-notes/).

| Chapter | Implementations |
| --- | --- |
| [Foundations](01-foundations/README.md) | Neural network, CNN, RNN, Transformer, ViT |
| [Generative models](02-generative-models/README.md) | VAE, DDPM |
| [Reinforcement learning](03-reinforcement-learning/README.md) | Tabular methods, DQN, REINFORCE, actor-critic, PPO |
| [Robot policies](04-robot-policies/README.md) | ACT, diffusion policy, flow-matching policy |

## Run

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. Run commands from the repository root.

```bash
uv sync
uv run python 01-foundations/01-neural-network/numpy_spiral.py --epochs 300
uv run python 01-foundations/02-cnn/cnn.py --smoke-test --epochs 1
uv run python 01-foundations/04-transformer/transformer.py --steps 20 --context 32 --layers 2
```

Use `uv sync --extra rl` for reinforcement-learning environments. Full robot-policy training needs additional robotics packages and the public `lerobot/pusht` dataset. Short synthetic runs verify code paths; checked-in rollouts are individual examples. Datasets and model weights are excluded from Git.
