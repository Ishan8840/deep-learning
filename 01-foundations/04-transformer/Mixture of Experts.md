# Mixture of Experts

In a dense model, every example activates the same set of parameters. As both model size and the number of training examples grow, training becomes more expensive. A **mixture-of-experts (MoE) layer** increases model capacity without activating every parameter for every input: a trainable **gating network** (or *router*) selects only a few **experts** to process each input.

In the [Sparsely-Gated MoE paper](https://arxiv.org/pdf/1701.06538), the experts are feed-forward networks (MLPs) with the same architecture but separate learned weights. The gate and experts are trained jointly by backpropagation.

## How an MoE layer works

Let $E_1, E_2, \ldots, E_n$ be the experts. For an input $x$, the gate produces a sparse vector $G(x)$, where $G(x)_i$ is the weight assigned to expert $i$. The layer output is

$$
y = \sum_{i=1}^{n} G(x)_i E_i(x).
$$

For example, if five experts have gate weights

$$
G(x) = [0,\ 0.7,\ 0,\ 0.3,\ 0],
$$

then only experts 2 and 4 run, and their outputs are combined as

$$
y = 0.7E_2(x) + 0.3E_4(x).
$$

Experts with zero gate weight do not need to be evaluated for that input.

## Gating and routing

**Softmax gating** multiplies the input by a trainable weight matrix and applies softmax:

$$
G_{\text{softmax}}(x) = \operatorname{softmax}(xW_g).
$$

This gives every expert a nonzero weight, so it does not provide sparse computation.

**Noisy top-$k$ gating** adds Gaussian noise with a learned scale to the scores, keeps the $k$ highest scores, sets all others to $-\infty$, and then applies softmax. The excluded experts therefore receive weight zero. The noise helps prevent the gate from settling on a small group of experts too early.

## Balancing expert use

Without balancing, the gate may send most inputs to a few experts. Those experts then receive more training, which can make the imbalance worse. For a batch $B$, the paper defines each expert's **importance** as its total gate weight:

$$
\operatorname{Importance}(E_i) = \sum_{x \in B} G(x)_i.
$$

An auxiliary **importance loss** encourages these totals to be similar across experts. The paper also uses a separate **load loss** to balance how many inputs are routed to each expert: equal total gate weights alone do not guarantee equal numbers of inputs.

## The shrinking batch problem

If a batch contains $b$ inputs and each input selects $k$ of $n$ experts, then each expert receives about $kb/n$ inputs **when routing is balanced**. As $n$ grows, these small expert batches make GPU computation less efficient.

The paper addresses this by increasing the effective batch size. Ordinary layers and gates use **data parallelism**, while experts are distributed across devices. At the MoE layer, inputs from all devices are grouped by their selected expert, so each expert processes a larger combined batch.

## Source
[Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017)](https://arxiv.org/pdf/1701.06538)
