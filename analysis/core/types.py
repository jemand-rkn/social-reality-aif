from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AnalysisPaths:
    input_dir: Path
    artifacts_dir: Path
    extracted_dir: Path
    shared_dir: Path
    step_metrics_dir: Path
    global_metrics_dir: Path
    figures_dir: Path
    videos_dir: Path

    @classmethod
    def from_input_dir(cls, input_dir: Path) -> "AnalysisPaths":
        artifacts_dir = input_dir / "analysis_artifacts"
        return cls(
            input_dir=input_dir,
            artifacts_dir=artifacts_dir,
            extracted_dir=artifacts_dir / "extracted",
            shared_dir=artifacts_dir / "shared",
            step_metrics_dir=artifacts_dir / "step_metrics",
            global_metrics_dir=artifacts_dir / "global_metrics",
            figures_dir=input_dir / "analysis_figures",
            videos_dir=input_dir / "analysis_videos",
        )


@dataclass(frozen=True)
class ExtractRunConfig:
    input_dir: Path
    output_dir: Optional[Path] = None
    device: Optional[str] = None
    steps: Optional[list[int]] = None
    refresh: bool = False
    num_workers: int = 1
    num_references: Optional[int] = None  # overrides cfg.num_references when set


@dataclass(frozen=True)
class StepMetricRunConfig:
    input_dir: Path
    device: Optional[str] = None
    steps: Optional[list[int]] = None
    metrics: Optional[list[str]] = None
    refresh: bool = False
    num_workers: int = 1


@dataclass(frozen=True)
class GlobalMetricRunConfig:
    input_dir: Path
    metrics: Optional[list[str]] = None
    refresh: bool = False
    num_workers: int = 1
    fgw_no_align: bool = False
    mds_seed: int = 0


@dataclass(frozen=True)
class RenderRunConfig:
    input_dir: Path
    figure_id: str
    step: Optional[int] = None
    steps: Optional[list[int]] = None
    refresh: bool = False


@dataclass(frozen=True)
class VideoRunConfig:
    input_dir: Path
    figure_ids: Optional[list[str]] = None
    steps: Optional[list[int]] = None
    refresh: bool = False
    fps: int = 12
    num_workers: Optional[int] = None


@dataclass(frozen=True)
class StepExtractContext:
    step: int
    checkpoint_dir: Path
    data_path: Path


@dataclass(frozen=True)
class RenderSeedsRunConfig:
    input_dirs: list[Path]
    output_dir: Path
    figure_ids: list[str]
    refresh: bool = False
    num_workers: int = 1
