# CLIP

**Contrastive Language–Image Pre-training (CLIP)** learns image representations by matching images to their captions. It was trained from scratch on roughly **400 million image–text pairs** collected from the internet. In the original paper, its zero-shot ImageNet accuracy matched that of a supervised [ResNet-50](../02-cnn/CNN.md) without training on ImageNet's 1.28 million labeled training images.

## Core idea

A conventional image classifier maps an image to one of a fixed set of labels. CLIP instead learns a **shared embedding space** in which an image and text describing it are close together:

$$
I \xrightarrow{f_\theta} z_I, \qquad T \xrightarrow{g_\phi} z_T, \qquad z_I,z_T\in\mathbb{R}^d.
$$

For example, a dog image should be more similar to *“a photo of a dog”* than to *“a photo of an airplane.”* This connection between visual and language concepts is called **vision–language alignment**.

## Architecture

CLIP has two independently run encoders:

- **Image encoder:** A ResNet or [Vision Transformer](ViT.md) converts an image into a feature vector, which is projected into the shared embedding space. With a ViT, the path is image → patches → Transformer → projection → image embedding.
- **Text encoder:** A Transformer converts caption tokens into a feature vector, which is projected into the same space.

The projected vectors are normalized before comparison. CLIP has **no cross-attention** between image and text: the encoders process their inputs separately, and their outputs meet through similarity scores.

## Contrastive training

For a batch of $N$ matched pairs $(I_i,T_i)$, CLIP computes the similarity of **every image with every caption**. The diagonal entries are matching, or *positive*, pairs; the other entries are in-batch *negative* pairs.

|  | $T_1$ | $T_2$ | $T_3$ |
| --- | :---: | :---: | :---: |
| $I_1$ | ✓ | × | × |
| $I_2$ | × | ✓ | × |
| $I_3$ | × | × | ✓ |

Similarity is based on the cosine of the angle between normalized embeddings:

$$
\operatorname{sim}(z_I,z_T)=\frac{z_I^\top z_T}{\lVert z_I\rVert\lVert z_T\rVert}.
$$

Aligned vectors have similarity near $1$; orthogonal vectors have similarity near $0$. CLIP scales the similarity scores with a learned temperature, then applies a softmax to each row and column. Cross-entropy loss trains it in **both directions**:

1. Given an image, identify its matching caption.
2. Given a caption, identify its matching image.

Together, these objectives increase the scores of matching pairs relative to the other pairs in the batch.

## Zero-shot classification

After training, CLIP can classify an image using text descriptions of candidate classes, without fitting a new classifier for those classes:

1. Turn each class into a prompt, such as *“a photo of a dog,” “a photo of a cat,”* and *“a photo of a car.”*
2. Encode the image and all three prompts.
3. Predict the class whose prompt has the highest image–text similarity.

For a golden retriever image, the dog prompt should score highest. The candidate prompts define the classes at inference time, so the result depends in part on the wording of those prompts.

## Limitations

- **Limited joint reasoning:** The separate encoders learn broad semantic alignment, but do not jointly inspect image and text in the way models with cross-modal interaction can.
- **Training-data dependence:** Missing concepts and biases in web image–text pairs can carry into CLIP's representations and predictions.
- **Loss of fine detail:** Distinguishing matching captions does not require retaining every visual detail. Precise geometry and spatial relationships may matter more in tasks such as robotics.

## Sources
[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
