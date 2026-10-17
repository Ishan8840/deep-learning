"""Plot the actual schedules defined in ddpm.py."""

from pathlib import Path

import matplotlib
from ddpm import betas, alphas, alpha_bars

matplotlib.use("Agg")
import matplotlib.pyplot as plt


timesteps = range(len(betas))

fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
for values, label, color in [
    (betas, r"$\beta_t$ — betas", "#2563eb"),
    (alphas, r"$\alpha_t$ — alphas", "#ea580c"),
    (alpha_bars, r"$\bar{\alpha}_t$ — alpha_bars", "#059669"),
]:
    ax.plot(timesteps, values, label=label, color=color, linewidth=2.5)

ax.set(title="DDPM noise schedules", xlabel="Schedule index (zero-based)", ylabel="Value",
       xlim=(0, len(betas) - 1), ylim=(-0.025, 1.025))
ax.grid(alpha=0.2)
ax.legend(loc="center right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
output = Path(__file__).resolve().parent / "outputs" / "ddpm_schedules.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=180)
plt.close(fig)
print(output)
