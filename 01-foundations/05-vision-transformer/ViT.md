# ViT

ViTs showed that a [Transformer](../04-transformer/README.md) trained on sequences of image patches can perform strongly on image classification, especially when pretrained at scale. This offers an alternative to [CNNs](../02-cnn/README.md); the data and compute needed depend on the training setup.

We reshape an image into a sequence of flattened 2D patches. The Transformer keeps a constant hidden size $D$ through its layers, so each flattened patch is mapped to $D$ dimensions with a trainable projection.

Learnable positional embeddings identify each patch's location.

Prepend a learnable classification token to the patch sequence. Its final representation summarizes the image for classification.

A classification head maps that token to class logits.


#### Sources
[ViT](https://arxiv.org/pdf/2010.11929)
