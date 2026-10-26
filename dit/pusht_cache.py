"""Optional in-memory PushT loader; preserves LeRobot's timestamp/padding rules."""

import av
import numpy as np
import torch
from torch.utils.data import Dataset


class CachedPushTDataset(Dataset):
    def __init__(self, source, horizon, video_cache):
        def column(key):
            return torch.stack(list(source.hf_dataset[key]))

        self.actions = column("action").float()
        self.states = column("observation.state").float()
        episodes = column("episode_index").flatten().long()
        timestamps = column("timestamp").flatten().double().numpy()
        self.ends = torch.empty(len(episodes), dtype=torch.long)
        self.offsets = torch.arange(horizon)
        images = None
        for episode in episodes.unique().tolist():
            rows = (episodes == episode).nonzero().flatten()
            self.ends[rows] = rows[-1]
            path = source.root / source.meta.get_video_file_path(episode, "observation.image")
            if path not in video_cache:
                frames, times = [], []
                with av.open(str(path)) as container:
                    for frame in container.decode(video=0):
                        frames.append(frame.to_ndarray(format="rgb24"))
                        times.append(float(frame.pts * frame.time_base))
                video_cache[path] = np.asarray(times), np.stack(frames)
                print(f"Cached {len(frames)} RGB frames from {path.name}", flush=True)
            times, frames = video_cache[path]
            offset = source.meta.episodes[episode]["videos/observation.image/from_timestamp"]
            targets = timestamps[rows.numpy()] + offset
            right = np.searchsorted(times, targets).clip(0, len(times) - 1)
            left = (right - 1).clip(0, len(times) - 1)
            selected = np.where(abs(times[left] - targets) < abs(times[right] - targets), left, right)
            if np.max(abs(times[selected] - targets)) > 0.005:
                raise ValueError(f"Video timestamps do not align with episode {episode}")
            selected_frames = torch.from_numpy(frames[selected]).permute(0, 3, 1, 2)
            if images is None:
                images = torch.empty((len(episodes), *selected_frames.shape[1:]), dtype=torch.uint8)
            images[rows] = selected_frames
        self.images = images

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        indices = index + self.offsets
        valid_indices = indices.clamp(max=self.ends[index].item())
        return {
            "action": self.actions[valid_indices],
            "action_is_pad": indices > self.ends[index],
            "observation.state": self.states[index],
            "observation.image": self.images[index].float().div(255),
        }
