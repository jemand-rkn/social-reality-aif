from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import imageio.v3 as iio
import imageio
import numpy as np


class VideoFrameManager:
    """Build MP4 videos from image frames on disk."""

    def __init__(self, fps: int = 12):
        if fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.fps = int(fps)

    @staticmethod
    def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported frame shape: {arr.shape}")

        if arr.dtype == np.uint8:
            return arr

        arr = arr.astype(np.float32)
        if np.isfinite(arr).all() and arr.max(initial=0.0) <= 1.0 and arr.min(initial=0.0) >= 0.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    @staticmethod
    def _pad_to_shape(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_hw
        h, w, _ = frame.shape
        if h == target_h and w == target_w:
            return frame

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return np.pad(
            frame,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def _resolve_target_size(self, frame_paths: Iterable[Path]) -> Tuple[int, int]:
        max_h = 0
        max_w = 0
        for path in frame_paths:
            frame = self._to_rgb_uint8(iio.imread(path))
            h, w, _ = frame.shape
            max_h = max(max_h, int(h))
            max_w = max(max_w, int(w))

        if max_h <= 0 or max_w <= 0:
            raise ValueError("No valid frame size found")
        return max_h, max_w

    def write_mp4(
        self,
        frame_paths: list[Path],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        if not frame_paths:
            raise ValueError("frame_paths must not be empty")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_hw = self._resolve_target_size(frame_paths)
        total_frames = len(frame_paths)

        if progress_callback is not None:
            progress_callback(0, total_frames)

        with imageio.get_writer(str(output_path), fps=self.fps) as writer:
            for idx, frame_path in enumerate(frame_paths, start=1):
                frame = self._to_rgb_uint8(iio.imread(frame_path))
                frame = self._pad_to_shape(frame, target_hw)
                writer.append_data(frame)
                if progress_callback is not None:
                    progress_callback(idx, total_frames)

        return output_path
