# Sliding windows from episodes

[`traj.py`](traj.py) turns variable-length episodes into fixed-length windows. For an episode with $T$ timesteps and window size $w$, it creates $T-w+1$ windows:

$$[x_0,\ldots,x_{w-1}], [x_1,\ldots,x_w], \ldots, [x_{T-w},\ldots,x_{T-1}].$$

Windows are created **within each episode** before concatenation, so a training example never crosses an episode boundary. `create_batches` then slices the resulting windows into batches and keeps a smaller final batch when needed.

The script uses random tensors as a shape demonstration. It assumes every episode has at least $w$ steps; shorter episodes cause `torch.stack` to fail. It does not shuffle or split episodes for training and evaluation.
