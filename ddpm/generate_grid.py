"""Generate 64 unfiltered samples using the existing reverse diffusion step."""

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision.utils import save_image

import ddpm
from sampling import p_sample_step


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "outputs/wide_lr0001/training/checkpoint_020.pt")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/wide_lr0001")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not torch.equal(checkpoint["betas"], ddpm.betas.cpu()):
        raise ValueError("Checkpoint schedule does not match ddpm.py")
    ddpm.betas = ddpm.betas.to(device)
    ddpm.alphas = ddpm.alphas.to(device)
    ddpm.alpha_bars = ddpm.alpha_bars.to(device)
    model = ddpm.UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(64, 1, 28, 28, device=device)
    for step, t in enumerate(range(len(ddpm.betas) - 1, -1, -1), start=1):
        x = p_sample_step(model, x, t)
        if step % 100 == 0:
            print(f"Sampling 64 images: {step}/{len(ddpm.betas)} steps", flush=True)
    assert x.shape == (64, 1, 28, 28) and torch.isfinite(x).all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x.cpu(), args.output_dir / "ddpm_samples_64.pt")
    output = args.output_dir / "ddpm_samples_64.png"
    save_image(((x.cpu() + 1) / 2).clamp(0, 1), output, nrow=8, padding=2, pad_value=0.3)
    with Image.open(output) as grid:
        grid.resize((grid.width * 4, grid.height * 4), Image.Resampling.NEAREST).save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
