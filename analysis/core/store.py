from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .types import AnalysisPaths


class ArtifactStore:
    def __init__(self, paths: AnalysisPaths):
        self.paths = paths

    def ensure_layout(self) -> None:
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.paths.shared_dir.mkdir(parents=True, exist_ok=True)
        self.paths.step_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.global_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)
        self.paths.videos_dir.mkdir(parents=True, exist_ok=True)

    def extracted_step_path(self, step: int) -> Path:
        return self.paths.extracted_dir / f"step_{step}.pt"

    def has_extracted_step(self, step: int) -> bool:
        return self.extracted_step_path(step).exists()

    def save_extracted_step(self, step: int, payload: Dict[str, Any]) -> Path:
        path = self.extracted_step_path(step)
        torch.save(payload, path)
        return path

    def load_extracted_step(self, step: int) -> Dict[str, Any]:
        return torch.load(self.extracted_step_path(step), map_location="cpu", weights_only=False)

    def shared_step_path(self, shared_id: str, step: int) -> Path:
        return self.paths.shared_dir / shared_id / f"step_{step}.npz"

    def has_shared_step(self, shared_id: str, step: int) -> bool:
        return self.shared_step_path(shared_id, step).exists()

    def save_shared_step(self, shared_id: str, step: int, payload: Dict[str, Any]) -> Path:
        out_dir = self.paths.shared_dir / shared_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step}.npz"
        np.savez(path, **payload)
        return path

    def load_shared_step(self, shared_id: str, step: int) -> Dict[str, Any]:
        payload = np.load(self.shared_step_path(shared_id, step), allow_pickle=True)
        return {k: self._decode_npz_value(payload[k]) for k in payload.files}

    def resolve_output_dir(self, override_dir: Optional[Path]) -> Path:
        return override_dir if override_dir is not None else self.paths.extracted_dir

    def step_metric_path(self, metric_id: str, step: int) -> Path:
        return self.paths.step_metrics_dir / metric_id / f"step_{step}.npz"

    def has_step_metric(self, metric_id: str, step: int) -> bool:
        return self.step_metric_path(metric_id, step).exists()

    def save_step_metric(self, metric_id: str, step: int, payload: Dict[str, Any]) -> Path:
        out_dir = self.paths.step_metrics_dir / metric_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step}.npz"
        encoded = {k: self._encode_npz_value(v) for k, v in payload.items()}
        np.savez(path, **encoded)
        return path

    def load_step_metric(self, metric_id: str, step: int) -> Dict[str, Any]:
        payload = np.load(self.step_metric_path(metric_id, step), allow_pickle=True)
        return {k: self._decode_npz_value(payload[k]) for k in payload.files}

    def _encode_npz_value(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if np.isscalar(value):
            return np.asarray(value)
        return np.asarray(value, dtype=object)

    def _decode_npz_value(self, value: Any) -> Any:
        if not isinstance(value, np.ndarray):
            return value
        if value.dtype == object and value.shape == ():
            return value.item()
        if value.shape == ():
            return value.item()
        return value

    def global_metric_path(self, metric_id: str) -> Path:
        return self.paths.global_metrics_dir / f"{metric_id}.npz"

    def has_global_metric(self, metric_id: str) -> bool:
        return self.global_metric_path(metric_id).exists()

    def save_global_metric(self, metric_id: str, payload: Dict[str, Any]) -> Path:
        self.paths.global_metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.global_metric_path(metric_id)
        encoded = {k: self._encode_npz_value(v) for k, v in payload.items()}
        np.savez(path, **encoded)
        return path

    def load_global_metric(self, metric_id: str) -> Dict[str, Any]:
        payload = np.load(self.global_metric_path(metric_id), allow_pickle=True)
        return {k: self._decode_npz_value(payload[k]) for k in payload.files}
