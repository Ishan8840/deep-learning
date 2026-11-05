# Feature normalization

[`norm.py`](norm.py) demonstrates per-feature standardization:

$$z = \frac{x-\mu}{\max(\sigma, 10^{-2}) + \epsilon}.$$

The inverse operation multiplies by the same denominator and adds the mean. Clamping small standard deviations keeps nearly constant features from exploding after normalization. The included toy data has one almost constant column so that effect is visible.

In a training pipeline, fit the mean and standard deviation on **training data only**, then reuse them for validation, test, and inference data. This file is a toy demonstration of the arithmetic; it does not save fitted statistics or handle missing values.
