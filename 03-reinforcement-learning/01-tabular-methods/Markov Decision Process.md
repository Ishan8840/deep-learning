# Markov Decision Process

A Markov decision process (MDP) models an agent repeatedly making decisions in an environment.

## Agent–environment interaction

$$
s_t\rightarrow a_t\rightarrow(r_t,s_{t+1})
$$

At each timestep $t$:

1. The agent is in state $s_t$.
2. It chooses an action $a_t$.
3. The environment returns a reward $r_t$ and transitions to state $s_{t+1}$.
4. The process repeats.

## Formal definition

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma)
$$

| Component | Meaning |
| --- | --- |
| $\mathcal S$ | Set of possible states |
| $\mathcal A$ | Set of possible actions |
| $P(s'\mid s,a)$ | Probability of reaching state $s'$ after taking action $a$ in state $s$ |
| $R$ | Reward function |
| $\gamma$ | Discount factor |

## Markov property

Given the current state and action, predicting the next state does not require the earlier history. The current state contains the information needed for that prediction.

## Partial observability

An MDP assumes the agent observes the complete environment state $s_t$. In robotics, the agent often receives an observation $o_t$, such as an RGB image, instead.

| Quantity | Example |
| --- | --- |
| State $s_t$ | True robot pose, object pose, velocities, and contact forces |
| Observation $o_t$ | Camera image |

A single image may not reveal the full state: objects can be occluded, velocity may not be visible from one frame, and contact forces may be hidden.

This is a **Partially Observable Markov Decision Process (POMDP)**.

- **MDP:** The agent observes the state $s_t$.
- **POMDP:** The agent observes $o_t$, generated from the hidden state $s_t$.

Vision-based robot policies generally operate under partial observability.
