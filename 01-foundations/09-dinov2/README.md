# DINOv2

**DINOv2** is a family of pretrained [Vision Transformers](../07-vision-transformer/README.md) that turns an image into **general-purpose visual features**. It learns from images without class labels or captions. The goal is a visual backbone whose features transfer across image distributions and tasks, often with the backbone **frozen** and a small task-specific head trained on top.

DINOv2 combines [DINO](../08-dino/README.md)-style image-level self-distillation, iBOT-style masked patch learning, carefully curated data, and large-scale training.

## What the encoder produces

A ViT divides an image into patches and returns both an image-level representation and spatial patch tokens:

$$
\text{image}\xrightarrow{\text{DINOv2}}
\left(z_{\mathrm{image}},\,z_1,z_2,\ldots,z_N\right).
$$

| Feature | What it captures | Example uses |
| --- | --- | --- |
| **Image-level feature** | Overall image content | Classification, retrieval, object recognition |
| **Patch features** | Information tied to image regions | Segmentation, depth estimation, other dense tasks |

The patch features are useful because a task can retain *where* information appears, rather than relying only on one vector for the entire image.

## How DINOv2 learns

### 1. Match different views of an image

As in DINO, a student and a slowly updated teacher process different augmented views of the same image. The student learns to match the teacher's **image-level output**:

$$
x\rightarrow(x_1,x_2)
\rightarrow
\text{consistent image-level outputs}.
$$

This encourages features that remain useful across changes such as crop, scale, and color. The teacher's exponential moving average (EMA) update supplies a stable target during self-supervised training.

### 2. Predict masked patch outputs

DINOv2 also uses an **iBOT-style patch objective**. Some patches are masked in the student view; the teacher sees the corresponding unmasked image. At the masked positions, the student predicts the teacher's patch-level outputs.

$$
\text{masked student patches}
\longrightarrow
\text{teacher outputs for those patches}.
$$

Together, the two objectives train **global image features** and **local spatial features**. The model is predicting learned teacher outputs, not reconstructing raw pixels or using segmentation labels.

## What makes DINOv2 stronger?

| Ingredient | Role |
| --- | --- |
| **Curated data** | An automated pipeline filters and balances diverse images. The resulting **LVD-142M** dataset contains about **142 million images**. |
| **Larger ViTs** | The largest model, **ViT-g/14**, has about **1.1 billion parameters**; “/14” denotes $14\times14$ pixel patches. |
| **Training at scale** | The paper combines and adapts self-supervised methods to make large-model training more stable and efficient. |
| **Distillation** | A large trained model transfers its representations to smaller ViTs that are cheaper to run. |

**Two kinds of teacher:** The **EMA teacher** provides targets while the large model is learning from images. During **model distillation**, the already trained large model teaches smaller models. These are different stages with different purposes.

Data curation matters as much as scale: the paper reports stronger transfer on most tested benchmarks from curated LVD-142M than from uncurated images.

## Why the features matter

DINOv2 is evaluated on both **image-level tasks**, such as classification and retrieval, and **pixel-level tasks**, such as segmentation and depth estimation. These tasks can use a frozen visual backbone with a trained probe or decoder; “no fine-tuning” refers to the **backbone**, not to every downstream component.

### Possible use in robotics

A robotics system needs more than an image-level label like “cup.” It may need features for object regions, boundaries, parts, and scene layout. DINOv2 patch tokens can serve as pretrained visual inputs to a policy or world model:

$$
\text{RGB image}
\rightarrow
\text{DINOv2 patch tokens}
\rightarrow
\text{policy or world model}.
$$

This is a possible application of the learned spatial features, not a robotics benchmark reported by the paper.

## DINOv2 compared with DINO and CLIP

| | DINO | DINOv2 | [CLIP](../10-clip/README.md) |
| --- | --- | --- | --- |
| **Training input** | Images | Images | Image–text pairs |
| **Main learning signal** | Match views of the same image | Match views **and** masked patch outputs | Match images with captions |
| **Emphasis** | Self-supervised semantic structure | Transferable global and spatial features at scale | Language-aligned visual features |

DINOv2 builds on DINO's teacher–student approach and adds patch-level learning, curated data at scale, larger models, and distillation into smaller ones. CLIP uses text supervision, making its features directly aligned with language. Both CLIP and DINOv2 learn semantic information, but their training signals emphasize different properties.

 **142M curated images → image-level + patch-level self-supervision → large ViT → smaller distilled models → reusable visual features.**

## Sources
[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
