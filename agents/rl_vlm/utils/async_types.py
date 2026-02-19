"""Data classes for async VLM/CLIP inference requests."""
from dataclasses import dataclass
import numpy as np


@dataclass
class InferenceRequest:
    prompt: str
    image: np.ndarray
    env_index: int
    id: str


@dataclass
class InferenceRequestOpenClip:
    action_desc: str      # Target action description (e.g., "accelerating gently and turning left")
    command_idx: int      # Command index (0-5)
    speed_idx: int        # Speed level index (0-3)
    image: np.ndarray     # Image data (H, W, 3)
    env_index: int        # Environment ID
    id: str               # Sample/frame ID