import torch


def create_windows(episodes, w_size):
    ep_windows = []

    for ep in episodes:
        windows = []

        for i in range(ep.size(0) - w_size + 1):
            windows.append(ep[i:w_size + i])

        ep_windows.append(torch.stack(windows))

    return torch.cat(ep_windows, dim=0)



episodes = [
    torch.randn(100, 14),
    torch.randn(80, 14),
    torch.randn(120, 14),
]

window_size = 10

w = create_windows(episodes, window_size)

print(w.size())

