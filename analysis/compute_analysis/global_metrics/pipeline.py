from __future__ import annotations

import fnmatch
from typing import Dict, List

from analysis.core.store import ArtifactStore
from analysis.core.types import AnalysisPaths, GlobalMetricRunConfig
from tqdm import tqdm
from .metrics import DEFAULT_GLOBAL_METRICS, GlobalRuntimeContext


class GlobalMetricPipeline:
    def __init__(self, config: GlobalMetricRunConfig):
        self.config = config
        self.paths = AnalysisPaths.from_input_dir(config.input_dir)
        self.store = ArtifactStore(self.paths)

    def _resolve_metric_ids(self) -> List[str]:
        if not self.config.metrics:
            return list(DEFAULT_GLOBAL_METRICS.keys())
        available = list(DEFAULT_GLOBAL_METRICS.keys())
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
                raise ValueError(f"Unknown global metric ids: {unknown}")
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
                    raise ValueError(f"Unknown global metric id in exclusion: {pat!r}")
            result = [mid for mid in result if mid not in excluded]
        return result

    def run(self) -> None:
        self.store.ensure_layout()
        metric_ids = self._resolve_metric_ids()
        print(f"Metrics ({len(metric_ids)}): {', '.join(metric_ids)}")

        context = GlobalRuntimeContext(
            config=self.config,
            paths=self.paths,
            store=self.store,
        )
        runtime: Dict = {"cache": {}}

        progress = tqdm(metric_ids, desc="Computing global metrics")
        for metric_id in progress:
            progress.set_postfix_str(metric_id)
            if self.store.has_global_metric(metric_id) and not self.config.refresh:
                continue
            metric_fn = DEFAULT_GLOBAL_METRICS[metric_id]
            data = metric_fn(context, runtime)
            if data is None:
                print(f"[WARNING] {metric_id} returned None — skipping save (missing required artifacts?)")
                continue
            payload: Dict = {
                "metric_id": metric_id,
                "data": data,
            }
            self.store.save_global_metric(metric_id, payload)
