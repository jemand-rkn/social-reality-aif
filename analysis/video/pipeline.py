from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from analysis.core.types import RenderRunConfig, VideoRunConfig
from analysis.visualize.pipeline import FigurePipeline
from .manager import VideoFrameManager


class MissingFramesError(RuntimeError):
    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__("Missing required frames: " + ", ".join(self.missing))


def _parse_step_from_frame(path: Path) -> int:
    name = path.stem
    if not name.startswith("step_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return -1


@dataclass(frozen=True)
class _SingleVideoTask:
    input_dir: Path
    figure_id: str
    steps: Optional[List[int]]
    fps: int
    refresh: bool


class VideoPipeline:
    def __init__(self, config: VideoRunConfig):
        self.config = config
        self.figures_dir = config.input_dir / "analysis_figures"
        self.videos_dir = config.input_dir / "analysis_videos"

    def list_stepwise_figure_ids(self) -> List[str]:
        render_cfg = RenderRunConfig(
            input_dir=self.config.input_dir,
            figure_id="observations_and_creations",
        )
        figure_pipeline = FigurePipeline(render_cfg)
        return [
            figure_id
            for figure_id in figure_pipeline.registry.list_ids()
            if figure_pipeline.registry.get(figure_id).category == "stepwise"
        ]

    def _resolve_frame_map(self, figure_id: str) -> Dict[int, Path]:
        figure_dir = self.figures_dir / figure_id
        if not figure_dir.exists():
            raise MissingFramesError([f"figures:{figure_id}:missing_dir"])

        frame_map: Dict[int, Path] = {}
        for png_path in figure_dir.glob("step_*.png"):
            step = _parse_step_from_frame(png_path)
            if step >= 0:
                frame_map[step] = png_path

        if not frame_map:
            raise MissingFramesError([f"figures:{figure_id}:step_png_not_found"])
        return frame_map

    def _resolve_frame_paths(self, figure_id: str, steps: Optional[List[int]]) -> List[Path]:
        frame_map = self._resolve_frame_map(figure_id)
        if steps is None:
            selected_steps = sorted(frame_map.keys())
        else:
            selected_steps = list(steps)
            missing_steps = [step for step in selected_steps if step not in frame_map]
            if missing_steps:
                missing = [f"figures:{figure_id}:step_{step}.png" for step in missing_steps]
                raise MissingFramesError(missing)

        if not selected_steps:
            raise MissingFramesError([f"figures:{figure_id}:no_steps_selected"])

        return [frame_map[step] for step in selected_steps]

    def render_video(self, figure_id: str, steps: Optional[List[int]]) -> Path:
        return self.render_video_with_progress(figure_id=figure_id, steps=steps, progress_callback=None)

    def render_video_with_progress(
        self,
        figure_id: str,
        steps: Optional[List[int]],
        progress_callback=None,
    ) -> Path:
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.videos_dir / f"{figure_id}.mp4"
        if output_path.exists() and not self.config.refresh:
            return output_path

        frame_paths = self._resolve_frame_paths(figure_id, steps)
        manager = VideoFrameManager(fps=self.config.fps)
        return manager.write_mp4(frame_paths, output_path, progress_callback=progress_callback)


def _resolve_worker_count(task_count: int, num_workers: Optional[int]) -> int:
    if task_count <= 1:
        return 1
    if num_workers is None:
        return min(task_count, max(1, (os.cpu_count() or 1) - 1))
    return min(task_count, max(1, int(num_workers)))


def _render_video_task_worker(
    task: _SingleVideoTask,
) -> Tuple[str, str, Optional[str], Optional[List[str]], Optional[str]]:
    config = VideoRunConfig(
        input_dir=task.input_dir,
        steps=task.steps,
        fps=task.fps,
        refresh=task.refresh,
        num_workers=1,
    )
    pipeline = VideoPipeline(config)
    output_path = config.input_dir / "analysis_videos" / f"{task.figure_id}.mp4"

    progress_state = {"last_reported": -1}

    def _progress_logger(done: int, total: int) -> None:
        if total <= 0:
            return

        # Report at 0%, every 5%, and 100% to keep logs readable.
        percent = int((100 * done) / total)
        if done == total:
            percent = 100

        should_report = False
        if done == 0:
            should_report = True
        elif done == total:
            should_report = True
        elif percent >= progress_state["last_reported"] + 5:
            should_report = True

        if not should_report:
            return

        progress_state["last_reported"] = percent
        print(
            f"[render-video][{task.figure_id}] {done}/{total} frames ({percent}%)",
            flush=True,
        )

    try:
        if output_path.exists() and not task.refresh:
            return (task.figure_id, "skipped", str(output_path), None, None)
        output = pipeline.render_video_with_progress(
            figure_id=task.figure_id,
            steps=task.steps,
            progress_callback=_progress_logger,
        )
        return (task.figure_id, "ok", str(output), None, None)
    except MissingFramesError as exc:
        return (task.figure_id, "missing", None, list(exc.missing), None)
    except Exception as exc:
        return (task.figure_id, "error", None, None, str(exc))


def _run_tasks(
    tasks: List[_SingleVideoTask],
    num_workers: Optional[int],
) -> List[Tuple[str, str, Optional[str], Optional[List[str]], Optional[str]]]:
    if not tasks:
        return []

    worker_count = _resolve_worker_count(len(tasks), num_workers)
    results: List[Tuple[str, str, Optional[str], Optional[List[str]], Optional[str]]] = []

    if worker_count == 1:
        for task in tqdm(tasks, desc="render-video", unit="task"):
            results.append(_render_video_task_worker(task))
        return results

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=worker_count) as pool:
        for result in tqdm(
            pool.imap_unordered(_render_video_task_worker, tasks),
            total=len(tasks),
            desc="render-video",
            unit="task",
        ):
            results.append(result)
    return results


class VideoBatchRunner:
    @staticmethod
    def render_many(
        base_config: VideoRunConfig,
        figure_ids: Sequence[str],
    ) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, List[str]], Dict[str, str]]:
        tasks = [
            _SingleVideoTask(
                input_dir=base_config.input_dir,
                figure_id=figure_id,
                steps=base_config.steps,
                fps=base_config.fps,
                refresh=base_config.refresh,
            )
            for figure_id in figure_ids
        ]

        rendered: Dict[str, Path] = {}
        skipped: Dict[str, Path] = {}
        missing_by_figure: Dict[str, List[str]] = {}
        error_by_figure: Dict[str, str] = {}

        results = _run_tasks(tasks, base_config.num_workers)
        for figure_id, status, output, missing, error in results:
            if status == "ok" and output is not None:
                rendered[figure_id] = Path(output)
            elif status == "skipped" and output is not None:
                skipped[figure_id] = Path(output)
            elif status == "missing" and missing is not None:
                missing_by_figure[figure_id] = sorted(set(missing))
            elif status == "error" and error is not None:
                error_by_figure[figure_id] = error

        return rendered, skipped, missing_by_figure, error_by_figure
