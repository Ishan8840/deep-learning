# LoRA (Low-Rank Adaptation)

As pretrained models grow, updating all their weights for each new task becomes expensive. **LoRA freezes the pretrained weights and trains a small, low-rank update instead.**

## Full fine-tuning vs. LoRA

In full fine-tuning, training updates the original weight matrix $W_0$. With LoRA, $W_0$ stays fixed and the effective weight is:

$$
W_{\text{effective}} = W_0 + \Delta W, \qquad \Delta W = BA
$$

Here, $A$ and $B$ are smaller trainable matrices. If $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, then for rank $r$:

$$
A \in \mathbb{R}^{r \times d_{\text{in}}}, \qquad
B \in \mathbb{R}^{d_{\text{out}} \times r}, \qquad
r \ll \min(d_{\text{in}}, d_{\text{out}})
$$

The update $BA$ has the same shape as $W_0$, but training requires only $r(d_{\text{in}} + d_{\text{out}})$ parameters instead of $d_{\text{in}}d_{\text{out}}$.

## Parameter example

For a $4096 \times 4096$ weight matrix and rank $r = 8$:

| Approach | Trainable parameters for this matrix |
| --- | ---: |
| Full fine-tuning | $4096^2 = 16{,}777{,}216$ |
| LoRA | $4096(8) + 8(4096) = 65{,}536$ |

LoRA trains **256 times fewer parameters** for this matrix. The idea is that adapting to a task may need changes along only a small number of directions.

## What the layer computes

A standard linear layer computes $h = W_0x$. With LoRA, it computes:

$$
h = W_0x + BAx
$$

The input follows two paths: $W_0x$ gives the original output, and $BAx$ gives the learned correction. Their outputs are added together.

For example, a pretrained robot model might already know many manipulation skills. To adapt it to fold a particular kind of shirt, LoRA learns an adjustment while keeping the pretrained weights frozen.

## Training and inference

- **During training:** Freeze $W_0$ and update only $A$ and $B$.
- **After training:** Keep the LoRA matrices separate, or merge them into $W_{\text{new}} = W_0 + BA$ so the layer computes $h = W_{\text{new}}x$.

**Mental model:** Freeze the large model, learn a small low-rank correction, and optionally merge that correction into the weights afterward.


### Sources
[Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
