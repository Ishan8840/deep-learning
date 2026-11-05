# Vision transformer

The [Transformer](../06-transformer/README.md) uses attention over token sequences. A vision transformer (ViT) turns an image into a sequence of **patch tokens**, adds positional embeddings and a learnable classification token, and applies Transformer blocks to predict an image class.

[`model.py`](model.py) implements that architecture for fixed-size images. [`train.py`](train.py) trains it on CIFAR-10 or generated smoke-test data:

```bash
uv run python 01-foundations/07-vision-transformer/train.py --smoke-test --epochs 1
```

The smoke test checks tensor shapes and optimization, not classification quality. The code uses a convolution whose kernel and stride equal the patch size to project non-overlapping patches into token embeddings.

ViT showed that attention-based image models can compete strongly with [CNNs](../03-cnn/README.md) when trained with enough data and compute. See [*An Image Is Worth 16×16 Words*](https://arxiv.org/abs/2010.11929) for the original paper.

The ViT study note follows below. Continue with [DINO](../08-dino/README.md), [DINOv2](../09-dinov2/README.md), and [CLIP](../10-clip/README.md) for visual and multimodal representations. The code in this folder implements only the ViT classifier.

## Study notes

ViTs showed that a [Transformer](../06-transformer/README.md) trained on sequences of image patches can perform strongly on image classification, especially when pretrained at scale. This offers an alternative to [CNNs](../03-cnn/README.md); the data and compute needed depend on the training setup.

We reshape an image into a sequence of flattened 2D patches. The Transformer keeps a constant hidden size $D$ through its layers, so each flattened patch is mapped to $D$ dimensions with a trainable projection.

Learnable positional embeddings identify each patch's location.

Prepend a learnable classification token to the patch sequence. Its final representation summarizes the image for classification.

A classification head maps that token to class logits.


#### Sources
[ViT](https://arxiv.org/pdf/2010.11929)
