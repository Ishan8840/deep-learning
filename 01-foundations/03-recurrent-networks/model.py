import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()

        self.W_xh = nn.Parameter(
            torch.randn(input_size, hidden_size) * 0.01
        )

        self.W_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.01
        )

        self.b = nn.Parameter(
            torch.zeros(hidden_size)
        )

        self.W_hq = nn.Parameter(
            torch.randn(hidden_size, out_size) * 0.01
        )

        self.b_q = nn.Parameter(
            torch.zeros(out_size)
        )

    def forward(self, seq):

        batch_size = seq.shape[1]

        h = torch.zeros(
            batch_size,
            self.W_hh.shape[0],
            device=seq.device
        )

        outputs = []

        for t in range(seq.shape[0]):
            x_t = seq[t]

            h = torch.tanh(
                x_t @ self.W_xh
                + h @ self.W_hh
                + self.b
            )

            y_t = h @ self.W_hq + self.b_q

            outputs.append(y_t)

        return torch.stack(outputs), h


# Data
seq = torch.randn(10, 8, 2)
targets = torch.randn(10, 8, 2)

# Model
model = RNN(
    input_size=2,
    hidden_size=4,
    out_size=2
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# Training
epochs = 1000

for epoch in range(epochs):

    optimizer.zero_grad()

    predictions, h = model(seq)

    loss = loss_fn(predictions, targets)

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.item())