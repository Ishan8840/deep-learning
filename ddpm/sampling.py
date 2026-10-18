import torch
import ddpm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddpm.betas = ddpm.betas.to(device)
ddpm.alphas = ddpm.alphas.to(device)
ddpm.alpha_bars = ddpm.alpha_bars.to(device)

model = ddpm.UNet().to(device)

checkpoint = torch.load(
    "outputs/training/checkpoint_020.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()


@torch.no_grad()
def p_sample_step(model, x, t):
    t_b = torch.full(
        (x.shape[0],),
        t,
        device=x.device,
        dtype=torch.long
    )

    eps = model(x, t_b)

    mean = (
            x - ddpm.betas[t] * eps / torch.sqrt(1 - ddpm.alpha_bars[t])
        ) / torch.sqrt(ddpm.alphas[t])

    if t == 0:
        return mean

    return mean + torch.sqrt(ddpm.betas[t]) * torch.randn_like(x)
