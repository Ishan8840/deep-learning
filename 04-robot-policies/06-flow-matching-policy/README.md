# Flow-matching policy

[`model.py`](model.py) predicts PushT action chunks by integrating a learned vector field conditioned on images and state. [`train.py`](train.py) trains the policy; [`inference.py`](inference.py) runs closed-loop rollouts. The [successful](success.mp4) and [failed](failure.mp4) clips are individual examples, not success-rate measurements.

Read the [flow-matching notes](../../02-generative-models/05-flow-matching/README.md) for the underlying objective. The [VITA](../04-vita-flow-matching/README.md) and [A2A](../05-a2a-flow-matching/README.md) folders contain paper notes on other flow-based robot policies.
