# DINO

**DINO** (*self-distillation with no labels*) is a self-supervised method for learning visual representations from unlabeled images. The paper's central finding is that self-supervised [Vision Transformers](ViT.md) develop semantic structure that is less apparent in supervised ViTs and [CNN](../02-cnn/CNN.md)s: their features work well for k-nearest-neighbor (k-NN) classification, and some attention maps reveal foreground objects and boundaries.

## Why self-supervision?

A supervised image classifier may receive one class label for an image that also contains objects, textures, boundaries, and spatial relationships. DINO instead creates a learning signal from **different augmented views of the same image**. It asks the model to produce consistent outputs across changes in crop, scale, and appearance, without human class labels or segmentation masks.

## Student and teacher

DINO uses two networks with the same architecture: a **student** $g_{\theta_s}$ and a **teacher** $g_{\theta_t}$. Given an unlabeled image $x$, it creates different views $x_1$ and $x_2$:

$$
x \longrightarrow (x_1,x_2), \qquad
x_1 \xrightarrow{\text{student}} p_s(x_1), \qquad
x_2 \xrightarrow{\text{teacher}} p_t(x_2).
$$

The student learns to match the teacher's output distribution for another view of the same image. In simplified form, the cross-entropy objective is

$$
\mathcal{L}=-\sum_k p_t^{(k)}(x_2)\log p_s^{(k)}(x_1).
$$

These output dimensions are **not predefined class labels** such as dog or car. The useful image representation is learned through this training task.

### Teacher update: exponential moving average

Only the student receives gradients. The teacher output is treated as a fixed target during a student update; afterward, the teacher parameters follow the student by an **exponential moving average (EMA)**:

$$
\theta_t \leftarrow m\theta_t+(1-m)\theta_s.
$$

With $m$ close to $1$, the teacher changes more slowly than the student. It acts as a smoothed history of student parameters and provides a relatively stable target. **EMA makes the target stable, not automatically meaningful.** Both networks start without semantic knowledge; useful structure emerges gradually across images and augmentations.

## Why doesn't it collapse?

Matching two views admits a trivial solution: map **every image to the same output**. DINO uses *centering* and *sharpening* of the teacher output to discourage this representation collapse.

### Centering

DINO tracks a running center $c$ of the teacher's output logits and subtracts it before the teacher softmax. If one output dimension dominates across many images, its average grows and centering suppresses it:

$$
z_t \longrightarrow z_t-c.
$$

This discourages all images from activating the same dominant output dimensions.

### Sharpening

The teacher uses a low-temperature softmax:

$$
p_t^{(k)}(x)=
\frac{\exp\!\left((z_t^{(k)}(x)-c_k)/\tau_t\right)}
{\sum_j\exp\!\left((z_t^{(j)}(x)-c_j)/\tau_t\right)}.
$$

A smaller teacher temperature $\tau_t$ makes its output distribution more peaked and discourages uninformative, nearly uniform targets. In short, **centering counters a globally dominant output**, while **sharpening counters a uniform output**.

## Multi-crop training

DINO creates views at multiple scales. The **teacher receives global crops**; the **student receives global and local crops**. For example, a student view containing a dog's head may be trained to match the teacher's output for a larger view containing much of the dog.

This encourages local visual information to remain consistent with a broader view of the same image. It is an intuitive explanation for the learned structure, not a claim that DINO is explicitly taught object parts or identities. The paper also finds that **smaller ViT patches** improve the learned visual features.

## What emerges?

- **Semantically useful embeddings:** Related images tend to be near one another in the learned feature space. A new image can be classified by finding nearby labeled training images and using their labels. With a small ViT, DINO reached **78.3% ImageNet top-1 accuracy using k-NN**, without training a classifier on top of its features.
- **Object-like attention:** Some attention heads in the final ViT block highlight foreground objects and approximate their boundaries. These are *segmentation-like* patterns in attention maps; DINO was not trained with segmentation masks.

The paper reports that these properties are less apparent in supervised ViTs and convolutional networks.

## DINO vs. [CLIP](CLIP.md)

| | DINO | CLIP |
| --- | --- | --- |
| Training signal | Match different views of the same image | Match an image with its text caption |
| Training data | Unlabeled images | Image–text pairs |
| Main alignment | Image view ↔ image view | Image ↔ language |

**Mental model:** Unlabeled image → multiple crops → student matches teacher → EMA updates teacher → centering and sharpening help prevent collapse → useful visual representations.

## Sources
[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294)
