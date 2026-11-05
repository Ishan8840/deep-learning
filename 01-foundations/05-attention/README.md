# Attention

Attention allows the model to focus on the relevant parts of the input sequence. Self attention connects everything directly. In the sentence "The cat sat on the mat", "sat" can directly attend to "mat"

 [RNN + LSTM](../04-recurrent-networks/README.md) processes the sentence sequentially, so if "sat" and "mat" were 50 words apart, it would have to travel through 50 hiddent states. This matters because gradients have to also travel along this path during training making it harder to make the connection between the words. You also are not able to fully parallelize RNNs since they use recursion, whereas using attention you can calcuate Q, K, V for all token simultaneously.

[CNN](../03-cnn/README.md) models can only see a small local region at a time, so they need multiple layers to connect far apart tokens

#### Scaled Dot Product Attention
- input: queries and keys of dim $d_k$, values of dim $d_v$
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
- Q is the query / what am I looking for
- K is the key / what do I contain
- dot product K $\cdot$ V shows how well does what I am looking for match what you offer
- V is the value / what information to give if I am relevant
- larger $d_k$ -> larger dot products -> softmax saturation -> tiny gradients -> divide by $\sqrt{d_k}$ to stabalize the scale

#### Multi-Head Attention
- perform attention in parallel
- multiple heads let the model look at the same sequence in several different ways at the same time, rather than forcing one attention pattern to capture everything.
$$
\text{MultiHead}(Q, K, v) = \text{Concat}(\text{head}_1, ...,\text{head}_h)W^O
$$
- $d_k = d_v = d_{model} / h$
- due to reduced dim of each head, the computational cost is similar to single head attention with full dimensionality

### Sources
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
[Visualizing Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
