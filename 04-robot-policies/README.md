# 04 · Robot policies

The previous chapters provide the building blocks: attention for observations, latent or generative models for multiple possible futures, and closed-loop control. Here the model output becomes a **chunk of future robot actions** conditioned on images and state.

1. [Action chunking with transformers](01-action-chunking/README.md) predicts an action chunk directly.
2. [Diffusion policy](02-diffusion-policy/README.md) denoises a candidate action chunk, with a transformer backbone.
3. [Flow-matching policy](03-flow-matching-policy/README.md) integrates a learned vector field from noise to an action chunk.

The checked-in [diffusion rollout](02-diffusion-policy/success_rollout.mp4), [flow-matching rollout](03-flow-matching-policy/success.mp4), and [failure rollout](03-flow-matching-policy/failure.mp4) are individual PushT examples. They are useful for understanding behavior, not aggregate evaluation. The training scripts require extra packages beyond the base project environment, including LeRobot, PyAV, Gymnasium, `gym_pusht`, Diffusers, OpenCV, Hugging Face Hub, and PyArrow.
