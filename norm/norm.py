import torch


class Normalizer:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = torch.clamp(std, min=1e-2)
        self.eps = eps

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x):
        return x * (self.std + self.eps) + self.mean


x = torch.tensor([
    [10.000000, 100.],
    [10.000001, 200.],
    [10.000002, 300.],
    [10.000003, 400.],
    [10.000004, 500.],
])

mean = x.mean(dim=0)
std = x.std(dim=0)

normalizer = Normalizer(mean, std)

x_norm = normalizer.normalize(x)
x_recovered = normalizer.unnormalize(x_norm)

print("mean:", mean)
print("std:", std)
print("normalized:", x_norm)
print("recovered:", x_recovered)

x_new = torch.tensor([
    [10.01, 350.]
])

print(normalizer.normalize(x_new))