import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
import ddpm


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
    if not torch.equal(checkpoint["betas"], ddpm.betas.cpu()):
        raise ValueError("Checkpoint schedule does not match ddpm.py")
    if len(ddpm.betas) != 1000:
        raise ValueError("This visualization requires a 1000-step schedule")
    ddpm.betas = ddpm.betas.to(device)
    ddpm.alphas = ddpm.alphas.to(device)
    ddpm.alpha_bars = ddpm.alpha_bars.to(device)
    model = ddpm.UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 1, 28, 28, device=device)
    selected = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 0]
    snapshots = {}
    frame_dir = output / "ddpm_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = [image_pair(x, "Initial Gaussian noise")]
    frames[0].save(frame_dir / "step_0000.png")

    for step, t in enumerate(range(999, -1, -1), start=1):
        x = p_sample_step(model, x, t)
        if t in selected:
            snapshots[t] = x.detach().cpu().clone()
        if step % 25 == 0:
            frame = image_pair(x, f"After t={t} | {step}/1000 steps")
            frames.append(frame)
            frame.save(frame_dir / f"step_{step:04d}.png")
        if step % 100 == 0:
            print(f"Sampling: {step}/1000 steps", flush=True)

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
    print(f"Saved GIF, progression grid, final images, and 41 frames to {output}")


if __name__ == "__main__":
    main()
