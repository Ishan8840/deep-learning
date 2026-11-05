# Generalized advantage estimation

**GAE** estimates advantages by taking an exponentially weighted sum of TD errors. It provides a tunable tradeoff between bias from an imperfect critic and variance from sampled future rewards.

## Why advantage estimates matter

Policy-gradient methods face two challenges: learning from limited samples and maintaining stable improvement as the policy changes the data it collects.

A central difficulty is **credit assignment**: determining which actions contributed to good or bad outcomes, especially when rewards arrive later.

The advantage measures how much better an action is than the policy’s usual behavior in that state:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

- $Q^\pi(s,a)$: Expected return after taking action $a$ in state $s$, then following $\pi$.
- $V^\pi(s)$: Expected return from state $s$ when following $\pi$.

Here, the value functions use the chosen discount $\gamma$; the discount is omitted from their superscripts for readability.

A policy-gradient update is built from terms of the form:

$$
\hat A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

Positive advantages encourage the sampled action; negative advantages discourage it. The quality of $\hat A_t$ therefore directly affects the update.

## Two starting points

### Monte Carlo: use the observed return

Wait for the actual discounted return:

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
$$

Subtract the learned state-value baseline:

$$
\hat A_t^{\mathrm{MC}}=G_t-V_\phi(s_t)
$$

This avoids bootstrapping from estimated future values, but can have **high variance**: similar rollouts may produce very different returns.

With on-policy data and complete returns, subtracting a state-only baseline held fixed during the actor update does not introduce policy-gradient bias. The advantage estimate itself is unbiased for $A^\pi$ only when the baseline equals $V^\pi$.

### One-step TD: bootstrap from the critic

Use the immediate reward and the estimated value of the next state:

$$
\delta_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

This **temporal-difference (TD) error** is a one-step advantage estimate. It asks whether the reward plus estimated future return was better or worse than the critic expected.

Using $\hat A_t=\delta_t$ typically reduces variance, but introduces more reliance on the critic. If $V_\phi$ is inaccurate, bootstrapping can bias the policy-gradient update.

## GAE: combine future TD errors

GAE incorporates information from later TD errors, with exponentially decreasing weights:

$$
\begin{aligned}
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
&=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l} \\
&=\delta_t
+\gamma\lambda\,\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots
\end{aligned}
$$

This creates a continuum between one-step TD and an estimate based on the full observed return.

For example, with $\gamma=0.99$ and $\lambda=0.95$:

$$
\gamma\lambda=0.9405
$$

$$
\hat A_t
=\delta_t+0.9405\,\delta_{t+1}
+0.9405^2\delta_{t+2}+\cdots
$$

These are example settings, not a universally optimal choice.

## What does lambda control?

**$\lambda$ controls how much future TD information contributes to the estimate.**

| Setting | Advantage estimate | Typical tradeoff |
| --- | --- | --- |
| $\lambda=0$ | $\hat A_t=\delta_t$ | Lower variance; greater reliance on critic accuracy |
| $0<\lambda<1$ | Weighted sum of future TD errors | Intermediate bias–variance tradeoff |
| $\lambda=1$ | $\hat A_t=G_t-V_\phi(s_t)$ for a complete episode | No bootstrapping bias; typically higher variance |

At $\lambda=1$, the intermediate value terms cancel in the discounted TD sum. For a complete episode ending at terminal state $s_T$, with $V_\phi(s_T)=0$:

$$
\sum_{l=0}^{T-t-1}\gamma^l\delta_{t+l}
=\sum_{l=0}^{T-t-1}\gamma^l r_{t+l}-V_\phi(s_t)
$$

A rollout cut off before termination still needs a bootstrapped final value, so $\lambda=1$ does not remove all dependence on critic accuracy in that case.

With an exact value function, one-step TD is already unbiased for the discounted advantage. Smaller $\lambda$ can introduce additional bias when the critic is inaccurate.

## Gamma and lambda play different roles

- **$\gamma$ discounts future rewards** and determines which return the critic estimates.
- **$\lambda$ discounts future TD errors** within the advantage estimator, controlling reliance on bootstrapping.

Their product $\gamma\lambda$ determines how quickly future TD errors lose weight, but the parameters are not interchangeable: $\gamma$ also appears inside each TD error.

The GAE paper starts with an **undiscounted objective** and treats $\gamma<1$ as a way to reduce variance at the cost of bias relative to that objective. If the task’s objective is already discounted, discounting is part of the objective rather than automatically an estimation bias. See [Sections 2–3](https://arxiv.org/html/1506.02438v6#S2).

## How the actor and critic use GAE

The **critic** $V_\phi(s)$ learns to predict returns. The **actor** $\pi_\theta(a\mid s)$ learns which actions to take.

1. Collect trajectories using the current policy.
2. Use rewards and critic predictions to calculate $\delta_t$.
3. Combine the TD errors into $\hat A_t^{\mathrm{GAE}}$.
4. Use the advantages as fixed weights in the actor update.
5. Fit the critic to return targets.

Schematically, the actor’s estimated gradient direction is:

$$
\hat g
=\frac{1}{N}\sum_{i=1}^{N}\sum_t
\hat A_t^{(i)}
\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
$$

Here, $N$ is the number of sampled trajectories. The exact actor objective depends on the optimization algorithm; GAE supplies the advantage estimates.

## Experimental takeaway

The paper combines GAE with trust-region optimization and demonstrates learning on difficult continuous-control tasks, including simulated locomotion with neural-network policies and value functions.

The practical lesson is that intermediate settings can balance critic error against noisy long-horizon returns more effectively than either extreme alone.

GAE also provides the advantage estimates commonly used in **PPO’s clipped objective**. It is an estimator used by the optimizer, not a policy-update algorithm itself.

## Sources
[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)
