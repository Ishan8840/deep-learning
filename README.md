# Rebuilding ML models, from first principles to robot actions

I use this repository to understand models by implementing their core ideas. The path starts with losses, attention, VAEs, and diffusion, then applies generative modeling to action sequences for robot control.

The robot action-policy experiments use **PushT**, a simulated 2D pushing task. The reinforcement-learning examples use gridworlds, CartPole, and Pendulum. These are learning implementations, not exact reproductions or claims of state-of-the-art performance. The clips show individual rollouts; they are not aggregate success rates.

## Start here

| Idea | Implementation | Notes | What to look at |
| --- | --- | --- | --- |
| Basic training loop | [MLP](mlp/pt_nn.py) | [MLP](mlp/README.md) | FashionMNIST classifier, validation, and learning-rate schedule |
| Manual backpropagation | [NumPy spiral network](mlp/numpy_spiral.py) | [MLP](mlp/README.md) | Dense layers, ReLU, and softmax cross-entropy without autograd |
| Training objectives | [MSE, cross-entropy, KL](loss) | [Loss functions](loss/README.md) | Loss formulas and small PyTorch checks |
| Feature scaling | [Normalizer](norm/norm.py) | [Normalization](norm/README.md) | Mean, standard deviation, and inverse transform |
| Sequential models | [RNN](rnn/model.py) | [RNN and LSTM](rnn/README.md) | Hidden-state recurrence; LSTM is explained in notes but not implemented |
| Attention and vision transformers | [Transformer](transformer/model.py) | [Transformer](transformer/README.md) | Patch tokens, multi-head attention, and a small CIFAR-10 training loop |
| Latent generative models | [VAE](vae/autoencoder.py) | [VAE](vae/README.md) | [Latent interpolation](vae/latent_interpolation.png) and [latent space](vae/latent_space.png) |
| Denoising diffusion | [DDPM](ddpm/ddpm.py) | [DDPM](ddpm/README.md) | Forward noise process, denoising objective, and sampling |
| Action chunking | [ACT](act/act.py) | [ACT](act/README.md) | Predict a chunk of future actions from observations |
| Diffusion for actions | [DiT-based policy](dit/model.py) | [DiT](dit/README.md), [Diffusion Policy](dit/Diffusion%20Policy.md) | [PushT rollout](dit/success_rollout.mp4) |
| Flow matching for actions | [Flow model](flow%20matching/model.py) | [Flow Matching](flow%20matching/README.md) | [Successful rollout](flow%20matching/success.mp4) and [failure rollout](flow%20matching/failure.mp4) |
| Episode windows | [Trajectory utilities](trajectory/traj.py) | [Trajectory windows](trajectory/README.md) | Fixed-length training windows that stay within episode boundaries |
| Tabular reinforcement learning | [Value iteration](rl_basics/value_iteration.py), [prediction](rl_basics/random_walk.py), [Monte Carlo control](rl_basics/monte_carlo_control.py), [Q-learning and SARSA](rl_basics/frozen_lake.py) | [RL basics](rl_basics/README.md) | Planning, sampled returns, and bootstrapped control |
| CartPole control | [Double DQN](cartpole/dqn.py), [REINFORCE](cartpole/reinforce.py) | [CartPole](cartpole/README.md) | Value-based and policy-gradient learning on the same task |
| Online actor-critic | [Pendulum TD actor-critic](actor_critic/pendulum_td.py) | [Actor-critic](actor_critic/README.md) | One-step TD error drives the policy and value updates |
| Continuous-action PPO | [Pendulum PPO](ppo/model.py) | [PPO](ppo/README.md), [GAE](ppo/GAE.md) | Gaussian policy, squashed actions, advantage estimation, PPO clipping |

The notes explain the source papers and concepts. The code shows which parts I implemented. In particular, the DiT note covers the original image model, while `dit/` adapts transformer-based diffusion to robot action sequences. The ACT note describes the paper's architecture; `act/` is a compact implementation and its synthetic training example is a pipeline check, not a PushT result.

The NumPy MLP, tabular RL, CartPole, and Pendulum examples were adapted from my `neural-network-scratch`, `rl-basics`, `cartpole-rl`, and `pendulum-V1` repos. I kept one version of each algorithm and fixed issues found while moving the code. The original BipedalWalker PPO experiment is available in this repository's Git history; the current PPO script uses the cleaner single-episode Pendulum setup.

## The thread through the projects

1. **Represent the data.** A VAE learns a compact latent space. A transformer makes tokens interact through attention.
2. **Generate by refinement.** DDPM learns to reverse a noising process. Flow matching learns a vector field that moves noise toward data.
3. **Change the output from images to actions.** ACT predicts an action chunk directly. The DiT-based and flow-matching policies generate action chunks conditioned on camera images and robot state.
4. **Test in a closed loop.** The policy replans as PushT changes. The success and failure clips make clear why training loss alone is not a complete evaluation.

## Run a small example

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are needed for the base environment:

```bash
uv sync
uv run python transformer/train.py --smoke-test --epochs 1
uv run python act/train.py --synthetic --epochs 1
```

For the reinforcement-learning examples, install the optional classic-control environment dependencies:

```bash
uv sync --extra rl
uv run --extra rl python rl_basics/frozen_lake.py --episodes 10000
uv run --extra rl python cartpole/dqn.py --episodes 650
uv run --extra rl python cartpole/reinforce.py --episodes 600
uv run --extra rl python actor_critic/pendulum_td.py --episodes 1000
uv run --extra rl python ppo/model.py --episodes 1000
```

The smoke tests use generated data. For the MNIST DDPM example, download the dataset on first run:

```bash
uv run python ddpm/train.py --download --epochs 1
```

The PushT training and evaluation scripts require additional packages beyond the base `pyproject.toml`, including LeRobot, PyAV, Gymnasium, `gym_pusht`, Diffusers, OpenCV, Hugging Face Hub, and PyArrow. They also download the public `lerobot/pusht` dataset or use a local copy. Check the script options before running a full experiment:

```bash
uv run python dit/train.py --help
uv run python 'flow matching/train.py' --help
uv run python dit/eval.py --help
uv run python 'flow matching/inference.py' --help
```

Training checkpoints and datasets are intentionally excluded from Git. The videos above are examples, so use the evaluation scripts to measure a model across multiple episodes.

## Sources

Where relevant, the notes beside each implementation link to the original papers. The code and explanations here are my learning artifacts; credit for the underlying methods belongs to the paper authors.
