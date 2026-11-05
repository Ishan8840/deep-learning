# Flow Matching

Flow matching trains a model to regress onto a target vector field that generates a desired probability path. It can offer faster training, faster generation, and better performance than [DDPM](../ddpm/README.md)-style models.

## From noise to data

Start with an easily sampled distribution:

$$
x_0\sim p_0=\mathcal N(0,I)
$$

The target is the true data distribution $q$:

$$
x_1\sim q(x_1)
$$

The goal is to transform the initial distribution into a final distribution that approximates the data:

$$
p_0\longrightarrow p_1,\qquad p_1\approx q
$$

For PushT, $x_1$ is one real action trajectory, and $q$ is the distribution of successful action trajectories.

## Training objective

Minimize the squared difference between the model's vector field $v_t(x;\theta)$ and the target vector field $u_t(x)$:

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t,\;x\sim p_t}
\left[\|v_t(x;\theta)-u_t(x)\|^2\right]
$$

The objective is intractable to use directly without knowing an appropriate probability path $p_t$ and target vector field $u_t$.

## Conditional probability paths

For each data point $x_1$, define an easy path from Gaussian noise to a small distribution around $x_1$. Mixing these conditional paths gives the overall probability path:

$$
p_t(x)=\int p_t(x\mid x_1)\,q(x_1)\,dx_1
$$

At $t=1$, this mixture closely approximates the data distribution:

$$
p_1(x)=\int p_1(x\mid x_1)\,q(x_1)\,dx_1\approx q(x)
$$

## Combining conditional vector fields

The overall velocity is a weighted average of conditional velocities. Each one is weighted by $p_t(x_1\mid x)$, rather than equally:

$$
u_t(x)=\int u_t(x\mid x_1)\,p_t(x_1\mid x)\,dx_1
$$

Equivalently:

$$
u_t(x)=\int u_t(x\mid x_1)\,
\frac{p_t(x\mid x_1)\,q(x_1)}{p_t(x)}\,dx_1
$$

For example, if two targets have weights $p_t(A\mid x)=0.9$ and $p_t(B\mid x)=0.1$, then:

$$
u_t(x)=0.9u_A+0.1u_B
$$

The velocity associated with target $A$ contributes more strongly.

## Conditional flow matching

Conditional Flow Matching (CFM) uses a simpler objective with the same optima as the original flow matching objective:

$$
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{t,\;x_1\sim q,\;x\sim p_t(\cdot\mid x_1)}
\left[\|v_t(x;\theta)-u_t(x\mid x_1)\|^2\right]
$$

## Gaussian conditional paths

Choose each conditional distribution to be Gaussian:

$$
p_t(x\mid x_1)
=\mathcal N\left(x;\mu_t(x_1),\sigma_t(x_1)^2I\right)
$$

Its center $\mu_t(x_1)$ moves and its spread $\sigma_t(x_1)$ changes with time. The path starts at standard Gaussian noise and ends at a narrow Gaussian around the data point:

$$
\mathcal N(0,I)
\quad\xrightarrow{t:0\rightarrow1}\quad
\mathcal N(x_1,\sigma_{\min}^2I)
$$

### Transporting noise

For a fixed target $x_1$, write $\mu_t=\mu_t(x_1)$ and $\sigma_t=\sigma_t(x_1)$. Scale and shift a noise sample $x_0\sim\mathcal N(0,I)$:

$$
\psi_t(x_0)=\sigma_t x_0+\mu_t
$$

- Multiplying by $\sigma_t$ sets the standard deviation.
- Adding $\mu_t$ moves the mean.

The transformed sample follows the desired conditional distribution:

$$
\psi_t(x_0)\sim\mathcal N(\mu_t,\sigma_t^2I)
$$

### Deriving the vector field

Differentiating the transport map with respect to time gives its velocity:

$$
u_t(\psi_t(x_0)\mid x_1)=\sigma'_t x_0+\mu'_t
$$

To express this velocity at the current position $x$, invert the transport map:

$$
x=\sigma_t x_0+\mu_t
\quad\Longrightarrow\quad
x_0=\frac{x-\mu_t}{\sigma_t}
$$

Substituting gives the conditional vector field:

$$
u_t(x\mid x_1)
=\frac{\sigma'_t}{\sigma_t}(x-\mu_t)+\mu'_t
$$

## Linear conditional paths

Choose the Gaussian's mean and standard deviation to change linearly with time:

$$
\mu_t=tx_1,\qquad \sigma_t=1-(1-\sigma_{\min})t
$$

| Time | Mean | Standard deviation | Conditional distribution |
| --- | --- | --- | --- |
| $t=0$ | $0$ | $1$ | $\mathcal N(0,I)$ |
| $t=1$ | $x_1$ | $\sigma_{\min}$ | $\mathcal N(x_1,\sigma_{\min}^2I)$ |

The mean moves from $0$ to $x_1$ while the standard deviation shrinks from $1$ to $\sigma_{\min}$. This produces straight-line conditional trajectories from noise toward the data.

## Sources

[Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
