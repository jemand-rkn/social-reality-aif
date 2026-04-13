from __future__ import annotations

import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from analysis.core.store import ArtifactStore
from analysis.core.types import AnalysisPaths, StepMetricRunConfig
from analysis.compute_analysis.step.metrics import DEFAULT_STEP_METRICS
from analysis.data_extraction.rebuild_society import parse_step


def _compute_single_step_metrics_worker(input_dir: str, step: int, metric_ids: List[str], refresh: bool) -> int:
    paths = AnalysisPaths.from_input_dir(Path(input_dir))
    store = ArtifactStore(paths)

    if not refresh and all(store.has_step_metric(metric_id, step) for metric_id in metric_ids):
        return step

    step_payload = store.load_extracted_step(step)
    runtime = {
        "cache": {},
        "step": step,
    }
    for metric_id in metric_ids:
        if store.has_step_metric(metric_id, step) and not refresh:
            continue
        metric_fn = DEFAULT_STEP_METRICS[metric_id]
        result = metric_fn(step_payload, runtime)
        if result is None:
            continue
        payload: Dict = {
            "step": step,
            "metric_id": metric_id,
            "data": result,
        }
        store.save_step_metric(metric_id, step, payload)
    return step


class StepMetricPipeline:
    def __init__(self, config: StepMetricRunConfig):
        self.config = config
        self.paths = AnalysisPaths.from_input_dir(config.input_dir)
        self.store = ArtifactStore(self.paths)

    def _resolve_steps(self) -> List[int]:
        extracted_files = sorted(self.paths.extracted_dir.glob("step_*.pt"), key=parse_step)
        steps = [parse_step(path) for path in extracted_files]
        steps = [step for step in steps if step >= 0]
        if self.config.steps:
            wanted = set(self.config.steps)
            steps = [step for step in steps if step in wanted]
        return steps

    def _resolve_metric_ids(self) -> List[str]:
        if not self.config.metrics:
            return list(DEFAULT_STEP_METRICS.keys())
        available = list(DEFAULT_STEP_METRICS.keys())
        include_pats = [p for p in self.config.metrics if not p.startswith("!")]
        exclude_pats = [p[1:] for p in self.config.metrics if p.startswith("!")]
        if include_pats:
            result: List[str] = []
            unknown: List[str] = []
            for pat in include_pats:
                if "*" in pat or "?" in pat:
                    matched = [mid for mid in available if fnmatch.fnmatch(mid, pat)]
                    if not matched:
                        unknown.append(pat)
                    else:
                        for mid in matched:
                            if mid not in result:
                                result.append(mid)
                else:
                    if pat not in available:
                        unknown.append(pat)
                    elif pat not in result:
                        result.append(pat)
            if unknown:
                raise ValueError(f"Unknown step metric ids: {unknown}")
        else:
            result = list(available)
        if exclude_pats:
            excluded: List[str] = []
            for pat in exclude_pats:
                if "*" in pat or "?" in pat:
                    excluded.extend(mid for mid in result if fnmatch.fnmatch(mid, pat))
                elif pat in result:
                    excluded.append(pat)
                else:
                    raise ValueError(f"Unknown step metric id in exclusion: {pat!r}")
            result = [mid for mid in result if mid not in excluded]
        return result

    def run(self) -> None:
        self.store.ensure_layout()

        metric_ids = self._resolve_metric_ids()
        print(f"Metrics ({len(metric_ids)}): {', '.join(metric_ids)}")
        steps = self._resolve_steps()
        if not steps:
            raise FileNotFoundError(
                f"No extracted step files found in {self.paths.extracted_dir}. Run extract first."
            )

        num_workers = max(1, int(self.config.num_workers))

        if num_workers < 2:
            for step in tqdm(steps, desc="Computing step metrics"):
                step_payload = self.store.load_extracted_step(step)
                runtime = {
                    "cache": {},
                    "step": step,
                }
                for metric_id in metric_ids:
                    if self.store.has_step_metric(metric_id, step) and not self.config.refresh:
                        continue
                    metric_fn = DEFAULT_STEP_METRICS[metric_id]
                    result = metric_fn(step_payload, runtime)
                    if result is None:
                        continue
                    payload: Dict = {
                        "step": step,
                        "metric_id": metric_id,
                        "data": result,
                    }
                    self.store.save_step_metric(metric_id, step, payload)
            return

        pending_steps = (
            steps
            if self.config.refresh
            else [step for step in steps if any(not self.store.has_step_metric(metric_id, step) for metric_id in metric_ids)]
        )
        if not pending_steps:
            return

        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = {
            executor.submit(
                _compute_single_step_metrics_worker,
                str(self.config.input_dir),
                step,
                metric_ids,
                bool(self.config.refresh),
            ): step
            for step in pending_steps
        }
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing step metrics (mt)"):
                step = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to compute step metrics for step={step}") from exc
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise
        finally:
            executor.shutdown(wait=False)
