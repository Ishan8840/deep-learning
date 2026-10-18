import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
import ddpm


def load_schedule(betas, device):
    """Restore the schedule used to train the selected checkpoint."""
    betas = betas.detach().cpu()
    if betas.ndim != 1 or len(betas) < 2 or not torch.all((betas > 0) & (betas < 1)):
        raise ValueError("Invalid checkpoint beta schedule")
    alphas = 1 - betas
    ddpm.betas = betas.to(device)
    ddpm.alphas = alphas.to(device)
    ddpm.alpha_bars = alphas.cumprod(0).to(device)


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


def image_pair(images, label):
    """Render a display-only copy of the two samples, in their original order."""
    pixels = ((images.detach().cpu() + 1) / 2).clamp(0, 1)
    pixels = (pixels * 255).round().to(torch.uint8)
    canvas = Image.new("RGB", (456, 264), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((8, 4), label, fill="black", font=font)
    for i in range(2):
        tile = Image.fromarray(pixels[i, 0].numpy()).resize(
            (224, 224), Image.Resampling.NEAREST
        )
        canvas.paste(tile, (i * 232, 40))
        draw.text((i * 232 + 8, 23), f"Sample {i + 1}", fill="black", font=font)
    return canvas


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=root / "outputs/training/checkpoint_020.pt")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()
    output = args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    load_schedule(checkpoint["betas"], device)
    num_steps = len(ddpm.betas)
    model = ddpm.UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 1, 28, 28, device=device)
    selected = sorted({min(num_steps - 1, round(t * num_steps / 1000))
                       for t in [999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 0]},
                      reverse=True)
    frame_interval = max(1, num_steps // 40)
    snapshots = {}
    frame_dir = output / "ddpm_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = [image_pair(x, "Initial Gaussian noise")]
    frames[0].save(frame_dir / "step_0000.png")

    for step, t in enumerate(range(num_steps - 1, -1, -1), start=1):
        x = p_sample_step(model, x, t)
        if t in selected:
            snapshots[t] = x.detach().cpu().clone()
        if step % frame_interval == 0 or t == 0:
            frame = image_pair(x, f"After t={t} | {step}/{num_steps} steps")
            frames.append(frame)
            frame.save(frame_dir / f"step_{step:04d}.png")
        if step % 100 == 0:
            print(f"Sampling: {step}/{num_steps} steps", flush=True)

    frames[0].save(
        output / "ddpm_sampling.gif", save_all=True,
        append_images=frames[1:], duration=[150] * (len(frames) - 1) + [1500],
        loop=0,
    )
    frames[-1].save(output / "ddpm_final.png")

    # Two rows (one per sample), twelve columns (one per selected timestep).
    grid = Image.new("RGB", (len(selected) * 120, 280), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=14)
    for column, t in enumerate(selected):
        pixels = ((snapshots[t] + 1) / 2).clamp(0, 1)
        pixels = (pixels * 255).round().to(torch.uint8)
        draw.text((column * 120 + 4, 4), f"After t={t}", fill="black", font=font)
        for row in range(2):
            tile = Image.fromarray(pixels[row, 0].numpy()).resize(
                (112, 112), Image.Resampling.NEAREST
            )
            grid.paste(tile, (column * 120 + 4, 28 + row * 124))
    grid.save(output / "ddpm_progression.png")
    print(f"Saved GIF, progression grid, final images, and {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
