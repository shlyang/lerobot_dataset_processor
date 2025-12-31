from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DatasetProcessor:
    """Handles LeRobot dataset loading and manipulation."""

    def __init__(self):
        self.dataset: Optional[LeRobotDataset] = None
        self.raw_hf_dataset = None

    def load_dataset(self, repo_id: str, root: Optional[Path] = None) -> LeRobotDataset:
        """Loads a LeRobot dataset."""
        self.dataset = LeRobotDataset(repo_id, root=root)
        # Keep a reference to raw dataset (without torch transform) for images
        self.raw_hf_dataset = self.dataset.hf_dataset.with_format(None)
        return self.dataset

    @property
    def metadata(self):
        if self.dataset is None: return None
        return self.dataset.meta

    def get_episode_range(self, episode_idx: int) -> tuple[int, int]:
        """Returns (start_index, end_index) for an episode."""
        if self.dataset is None: return 0, 0
        from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_idx]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_idx]
        return int(from_idx), int(to_idx)

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Fetches data for a specific global frame index."""
        # Try to get frame with decoded video/images
        # LeRobot dataset with video format needs special handling
        frame_data = self.dataset[frame_idx]
        
        # Check if images are in video format (dict with 'path' key)
        # If so, they should already be decoded by LeRobotDataset
        return frame_data

    def get_episode_data(self, episode_idx: int, keys: List[str]) -> Dict[str, np.ndarray]:
        """Fetches all frames for an episode for specific keys (e.g., state, action)."""
        start, end = self.get_episode_range(episode_idx)
        
        # 优化：使用 select 批量获取，避免逐帧循环
        selected_data = self.dataset.hf_dataset.select(range(start, end))
        
        result = {}
        for key in keys:
            if key in selected_data.features:
                # 批量获取整列数据
                col_data = selected_data[key]
                if isinstance(col_data, list):
                    # 转换列表为 numpy 数组
                    if len(col_data) > 0:
                        if torch.is_tensor(col_data[0]):
                            result[key] = torch.stack(col_data).numpy()
                        else:
                            result[key] = np.array(col_data)
                elif torch.is_tensor(col_data):
                    result[key] = col_data.numpy()
                else:
                    result[key] = np.array(col_data)
        return result