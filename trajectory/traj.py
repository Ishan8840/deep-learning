import torch


def create_windows(episodes, w_size):
    ep_windows = []

    for ep in episodes:
        windows = []

        for i in range(ep.size(0) - w_size + 1):
            windows.append(ep[i:w_size + i])

        ep_windows.append(torch.stack(windows))

    return torch.cat(ep_windows, dim=0)


def create_batches(windows, batch_size):
    batches = []

    for i in range(0, windows.size(0), batch_size):
        batches.append(windows[i:batch_size + i])

    return batches



episodes = [
    torch.randn(100, 14),
    torch.randn(80, 14),
    torch.randn(120, 14),
]

window_size = 10

w = create_windows(episodes, window_size)

print(w.size())

