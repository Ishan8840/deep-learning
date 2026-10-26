# PushT DiT training

Training setup for the conditional model in `model.py`, using
[`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht).
No dataset or model weights are included. The commands below are for your VM;
training and dataset downloads have not been run during setup.

## VM setup

Use an Ubuntu NVIDIA GPU VM with Python **3.11**, a working NVIDIA driver,
and enough disk space for the dataset cache and checkpoints. Start with one GPU.
Copy this directory to the VM, then run these commands from that directory:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Python 3.11 must already be installed; use a VM image that provides it if the
distribution does not package it. The CUDA wheel above requires a driver that
supports CUDA 12.6. Use a matching official PyTorch wheel index for a different
driver, keeping the torch/torchvision versions compatible with `requirements.txt`.
A separate CUDA toolkit installation is unnecessary for these wheels.

LeRobot is pinned to 0.4.3 and the dataset revision to `v3.0` to match the loader
API and dataset format. The script explicitly uses PyAV for video decoding.
See the [pinned dependencies](https://github.com/huggingface/lerobot/blob/v0.4.3/pyproject.toml)
and [loader implementation](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/datasets/lerobot_dataset.py).

## Launch on the VM

Choose a persistent disk for the Hugging Face cache and training output:

```bash
export HF_HOME=/path/to/persistent-disk/huggingface
python train.py --output-dir /path/to/persistent-disk/runs/pusht
```

**Launching `train.py` downloads uncached metadata, parquet files, and videos,
then starts training.** Installation alone does not download the dataset.
The dataset is public. `--dataset-root /path/to/local/pusht` optionally selects
an explicit dataset directory; otherwise LeRobot uses its Hugging Face cache.
Use the same cache/root when resuming.

For a GPU with native bfloat16 support, add `--amp bf16` to reduce memory use.
Otherwise use `--amp fp16` with gradient scaling, or the default `--amp none`.
If GPU memory is tight, reduce `--batch-size 32`. If video loading has problems,
try `--workers 0`. For a short VM check before the full run:

```bash
python train.py --steps 10 --warmup-steps 0 --batch-size 2 --workers 0 \
  --log-every 1 --eval-every 10 --val-batches 2 \
  --output-dir outputs/pusht-check
```

That command really downloads data and trains ten steps; it is provided for you
to run later. It has not been executed here.

## What gets trained

- One image and one state at time `t` condition your existing single DiT block.
- The [dataset metadata](https://huggingface.co/datasets/lerobot/pusht/blob/main/meta/info.json)
  specifies 96×96 RGB images, 2D states, 2D actions, and 10 Hz sampling.
- The target chunk is `[action[t], ..., action[t+15]]` by default. LeRobot's
  timestamp offsets respect episode boundaries; padded actions are excluded
  from the loss using `action_is_pad`.
- A seeded episode split reserves approximately 10% of episodes for validation.
  Neighboring frames from one episode cannot land in both splits.
- State and action per-coordinate min/max values are computed from training
  episodes only and map values to `[-1, 1]`. Validation values are not clipped.
- Decoder images arrive as CHW floats in `[0, 1]` and are mapped to `[-1, 1]`.
- The loss is masked noise-prediction MSE using `model.py`'s 1,000-step linear
  diffusion schedule. The loop uses AdamW, warmup/cosine LR decay, and gradient
  clipping. It uses its own loss loop to support masking and optional AMP;
  the simple `model.py:train_step` helper remains available.
- Defaults: 100,000 optimization steps, batch size 64, hidden dimension 256,
  four attention heads, horizon 16, learning rate `1e-4`, and seed 42.

Validation uses fixed noise and timesteps, up to `--val-batches` batches per
evaluation. It measures held-out denoising loss, not simulated task success.
This is a baseline setup for your model; no training quality claims have been
validated. Environment rollouts and reverse-diffusion action sampling are not
part of this training script.

## Outputs and resume

The output directory contains:

- `config.json`: resolved CLI configuration.
- `metrics.jsonl`: training loss, validation loss, LR, and elapsed time.
- `last.pt`: latest saved training state; saved at `--save-every`, whenever
  validation improves, and at the final step.
- `best.pt`: checkpoint with the lowest measured validation noise MSE.

Checkpoints include model constructor arguments and weights, optimizer,
scheduler, AMP scaler, step, normalization statistics, diffusion betas,
episode split, preprocessing metadata, and random states. Saves replace files
atomically. For the default training configuration, resume with:

```bash
python train.py --output-dir /path/to/persistent-disk/runs/pusht \
  --resume /path/to/persistent-disk/runs/pusht/last.pt
```

Repeat your original model/training flags if you changed them, including
`--steps`, `--batch-size`, `--amp`, and `--seed`. The script rejects incompatible
resume settings. Resume restores optimization state but restarts the data
iterator with a new shuffle; it does not reproduce an interrupted epoch's
exact remaining batch order. Only load checkpoints you trust.

For later inference, reconstruct `DiT(**checkpoint["model_kwargs"])`, load
`checkpoint["model"]`, and apply the saved preprocessing. The predicted output
is noise, so a reverse-diffusion sampler is needed to generate actions.
Convert normalized actions back to dataset coordinates with
`(action + 1) * 0.5 * (max - min).clamp_min(1e-6) + min`, using the saved
`checkpoint["normalization"]["action"]` tensors.
