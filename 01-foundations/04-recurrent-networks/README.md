# Recurrent neural networks and LSTMs

[`model.py`](model.py) implements a vanilla RNN and trains it on synthetic sequences. The LSTM discussion below is conceptual; there is no LSTM implementation in this folder.

An RNN carries a hidden state between timesteps so that the current output can depend on earlier inputs:

$$h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)$$
$$y_t = h_t W_{hy} + b_y$$
Vanilla RNNs have a problem on not being good at handling long term dependencies. LSTMs are designed to handle the long term dependency problem.

**forget gate layer**
- look at $h_{t-1}$ and $x_t$ and outputs a number between 0 and 1 for each number in the cell state $C_{t−1}$.
- A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”
- This is a sigmoid layer
- $f_t = \sigma (W_f\cdot[h_{t-1}, x_t] + b_f)$

**input gate layer**
- sigmoid layer decides that values we will update
-  $i_t = \sigma (W_i\cdot[h_{t-1}, x_t] + b_i)$

**tanh layer**
- creates a vector of new candidate values, $C_t$
- $\widetilde C_t = \tanh (W_C\cdot[h_{t-1}, x_t] + b_C)$

**output**
- sigmoid layer to decide what parts of the cell state we are going to output
- then we put the cell state through a tanh and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to

### Continuation

RNNs struggle to preserve information from distant timesteps because information and gradients must pass through many recurrent steps. LSTMs alleviate this with a gated cell state, but still process information sequentially. [Attention](../06-transformer/README.md) allows direct connections between distant positions, making long-range dependencies easier to model.


### Sources
[colah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
