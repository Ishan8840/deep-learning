# From neural networks to robot policies

A guided set of notes and implementations tracing how neural networks learn representations, generate data, and choose actions. Start with the [neural network](01-foundations/01-neural-network/README.md), then the [CNN](01-foundations/02-cnn/README.md), [recurrent network](01-foundations/03-recurrent-networks/README.md), [Transformer](01-foundations/04-transformer/README.md), and [vision transformer](01-foundations/05-vision-transformer/README.md). The last chapter applies these ideas to action generation in the PushT simulator.

## The path

| Chapter | What it answers |
| --- | --- |
| [01 · Foundations](01-foundations/README.md) | How do gradients, memory, and attention let a model learn? |
| [02 · Generative models](02-generative-models/README.md) | How do latent variables and denoising turn noise into structured data? |
| [03 · Reinforcement learning](03-reinforcement-learning/README.md) | How do value estimates and policies improve through interaction? |
| [04 · Robot policies](04-robot-policies/README.md) | How can models produce action sequences from images and state? |

The robot chapter contains [ACT](04-robot-policies/01-action-chunking/README.md), a [diffusion policy](04-robot-policies/02-diffusion-policy/README.md), and a [flow-matching policy](04-robot-policies/03-flow-matching-policy/README.md). The [diffusion](04-robot-policies/02-diffusion-policy/success_rollout.mp4) and [flow-matching](04-robot-policies/03-flow-matching-policy/success.mp4) clips show individual PushT rollouts, not measured success rates.

## Run an example

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are needed. From the repository root:

```bash
uv sync
uv run python 01-foundations/01-neural-network/numpy_spiral.py --epochs 300
uv run python 01-foundations/02-cnn/cnn.py --smoke-test --epochs 1
uv run python 01-foundations/04-transformer/transformer.py --steps 20 --context 32 --layers 2
uv run python 04-robot-policies/01-action-chunking/train.py --synthetic --epochs 1
```

The last two commands are pipeline checks on synthetic data. For the reinforcement-learning examples, install the optional environments with `uv sync --extra rl` and follow the commands in [chapter 03](03-reinforcement-learning/README.md). Full PushT training needs additional robotics packages and the public `lerobot/pusht` dataset; see [chapter 04](04-robot-policies/README.md).

## Scope

These are learning implementations and notes, not exact paper reproductions or benchmark claims. Notes link the original papers where applicable. Model weights and datasets are excluded from Git; the checked-in images and videos are examples to inspect.
