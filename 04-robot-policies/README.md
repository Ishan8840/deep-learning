# 04 · Robot policies

Here the model output becomes a **chunk of future actions** conditioned on images and robot state. The first two topics include PushT implementations. The next three are paper notes, followed by the flow-matching policy implementation.

| Topic | What to look for |
| --- | --- |
| [01 · ACT](01-act/README.md) | Direct action chunks with a conditional VAE |
| [02 · Diffusion policy](02-diffusion-policy/README.md) | Denoising action chunks with a Transformer backbone |
| [03 · ALOHA Unleashed](03-aloha-unleashed/README.md) | Scaling diffusion-based robot policies |
| [04 · VITA Flow Matching](04-vita-flow-matching/README.md) | Flow-matching paper notes |
| [05 · A2A Flow Matching](05-a2a-flow-matching/README.md) | Flow-matching paper notes |
| [06 · Flow-matching policy](06-flow-matching-policy/README.md) | PushT implementation and rollout videos |

The checked-in [diffusion rollout](02-diffusion-policy/success_rollout.mp4), [flow-matching rollout](06-flow-matching-policy/success.mp4), and [failure rollout](06-flow-matching-policy/failure.mp4) show individual PushT examples, not aggregate evaluation. Full training needs additional robotics packages and the public `lerobot/pusht` dataset.
