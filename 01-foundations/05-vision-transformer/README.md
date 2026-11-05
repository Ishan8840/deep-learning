# Vision transformer

[`model.py`](model.py) implements that architecture for fixed-size images. [`train.py`](train.py) trains it on CIFAR-10 or generated smoke-test data:

| File |
| --- |
| [model.py](model.py) |
| [test_model.py](test_model.py) |
| [train.py](train.py) |

Run from the repository root:

```bash
uv run python 01-foundations/05-vision-transformer/train.py --smoke-test --epochs 1
```

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/01-foundations/07-vision-transformer).
