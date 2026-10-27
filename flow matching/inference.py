import torch


@torch.no_grad()
def sample_actions(model, qpos, images, num_steps=20):
    batch_size = qpos.shape[0]

    x = torch.randn(
        batch_size,
        model.chunk_size,
        model.action_dim,
        device=qpos.device,
        dtype=qpos.dtype,
    )

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full(
            (batch_size,),
            i / num_steps,
            device=x.device,
            dtype=x.dtype,
        )

        v = model(qpos, images, x, t)

        x = x + dt * v

    return x