# Vision transformer

The [Transformer](../04-transformer/README.md) uses attention over token sequences. A vision transformer (ViT) turns an image into a sequence of **patch tokens**, adds positional embeddings and a learnable classification token, and applies Transformer blocks to predict an image class.

[`model.py`](model.py) implements that architecture for fixed-size images. [`train.py`](train.py) trains it on CIFAR-10 or generated smoke-test data:

```bash
uv run python 01-foundations/05-vision-transformer/train.py --smoke-test --epochs 1
```

The smoke test checks tensor shapes and optimization, not classification quality. The code uses a convolution whose kernel and stride equal the patch size to project non-overlapping patches into token embeddings.

ViT showed that attention-based image models can compete strongly with [CNNs](../02-cnn/README.md) when trained with enough data and compute. See [*An Image Is Worth 16×16 Words*](https://arxiv.org/abs/2010.11929) for the original paper.

Related notes: [ViT](ViT.md), [DINO](DINO.md), [DINOv2](DINOv2.md), and [CLIP](CLIP.md). These cover visual and multimodal representations; the code in this folder implements only the ViT classifier.
