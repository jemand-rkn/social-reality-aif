from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from analysis.core.store import ArtifactStore
from analysis.core.types import AnalysisPaths, RenderRunConfig, RenderSeedsRunConfig
from analysis.data_extraction.rebuild_society import parse_step
from . import figures_eval, figures_fgw, figures_rsa, figures_summary


class MissingArtifactsError(RuntimeError):
    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        message = "Missing required artifacts: " + ", ".join(self.missing)
        super().__init__(message)


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    category: str
    requires_step: bool
    requires_steps: bool
    required_step_metrics: List[str]
    required_global_metrics: List[str]
    loader: Callable[["FigurePipeline", Optional[int], Optional[List[int]]], Dict[str, Any]]
    renderer: Callable[[Dict[str, Any]], matplotlib.figure.Figure]
    frame_steps_resolver: Optional[Callable[["FigurePipeline", Optional[List[int]]], List[int]]] = None


class FigureRegistry:
    def __init__(self):
        self._specs: Dict[str, FigureSpec] = {}

    def register(self, spec: FigureSpec) -> None:
        if spec.figure_id in self._specs:
            raise ValueError(f"Duplicate figure_id: {spec.figure_id}")
        self._specs[spec.figure_id] = spec

    def get(self, figure_id: str) -> FigureSpec:
        if figure_id not in self._specs:
            known = ", ".join(sorted(self._specs.keys()))
            raise ValueError(f"Unknown figure_id: {figure_id}. Known: {known}")
        return self._specs[figure_id]

    def list_ids(self) -> List[str]:
        return sorted(self._specs.keys())


class FigurePipeline:
    def __init__(self, config: RenderRunConfig):
        self.config = config
        self.paths = AnalysisPaths.from_input_dir(config.input_dir)
        self.store = ArtifactStore(self.paths)
        self.registry = self._build_registry()

    def _build_registry(self) -> FigureRegistry:
        reg = FigureRegistry()

        # Stepwise figures
        reg.register(
            FigureSpec(
                figure_id="observations_and_creations_clustered",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=lambda p, step, steps: {
                    "step": step,
                    "data": p._load_step_metric_data("observations_and_creations_clustered", step),
                },
                renderer=lambda payload: figures_eval.render_observations_and_creations_clustered(
                    payload["data"], payload["step"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="observations_and_creations_clustered_legend",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=lambda p, step, steps: {
                    "data": p._load_step_metric_data("observations_and_creations_clustered", step)
                },
                renderer=lambda payload: figures_eval.render_observations_and_creations_clustered_legend(payload["data"]),
                frame_steps_resolver=lambda p, steps: [
                    (steps[0] if steps else sorted(p._available_steps_for_metric("observations_and_creations_clustered"))[0])
                ],
            )
        )
        reg.register(
            FigureSpec(
                figure_id="observations_and_creations_agents_clustered",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=lambda p, step, steps: {
                    "step": step,
                    "data": p._load_step_metric_data("observations_and_creations_clustered", step),
                },
                renderer=lambda payload: figures_eval.render_observations_and_creations_agents_clustered(
                    payload["data"], payload["step"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="observations_and_creations_agents_clustered_legend",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=lambda p, step, steps: {
                    "data": p._load_step_metric_data("observations_and_creations_clustered", step)
                },
                renderer=lambda payload: figures_eval.render_observations_and_creations_agents_clustered_legend(payload["data"]),
                frame_steps_resolver=lambda p, steps: [
                    (steps[0] if steps else sorted(p._available_steps_for_metric("observations_and_creations_clustered"))[0])
                ],
            )
        )
        reg.register(
            FigureSpec(
                figure_id="observations_agents_clustered",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=lambda p, step, steps: {
                    "step": step,
                    "data": p._load_step_metric_data("observations_and_creations_clustered", step),
                },
                renderer=lambda payload: figures_eval.render_observations_agents_clustered(
                    payload["data"], payload["step"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="wasserstein_similarity",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["wasserstein_similarity", "gromov_wasserstein_similarity"],
                required_global_metrics=[],
                loader=self._load_step_fgw_pair,
                renderer=lambda payload: figures_eval.render_wasserstein_similarity(payload["data"], payload["step"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_similarity",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["wasserstein_similarity", "gromov_wasserstein_similarity"],
                required_global_metrics=[],
                loader=self._load_step_fgw_pair,
                renderer=lambda payload: figures_eval.render_gromov_wasserstein_similarity(payload["data"], payload["step"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="latent_scatter_from_reference",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["latent_scatter_from_reference"],
                required_global_metrics=["latent_scatter_pca_alignment"],
                loader=self._load_latent_scatter_aligned,
                renderer=lambda payload: figures_eval.render_latent_scatter_from_reference(
                    payload["data"], payload["step"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="social_network",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=lambda p, step, steps: {
                    "step": step,
                    "data": p._load_step_metric_data("observations_and_creations_clustered", step),
                },
                renderer=lambda payload: figures_eval.render_social_network(payload["data"], payload["step"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="mhng_acceptance_matrix",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["mhng_edge_acceptance"],
                required_global_metrics=[],
                loader=self._load_mhng_step,
                renderer=lambda payload: figures_eval.render_mhng_acceptance_matrix(payload["data"], payload["step"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="mhng_acceptance_network",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["mhng_edge_acceptance", "observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=self._load_mhng_step,
                renderer=lambda payload: figures_eval.render_mhng_acceptance_network(payload["data"], payload["step"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="memorize_acceptance_matrix",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["memorize_edge_acceptance"],
                required_global_metrics=[],
                loader=self._load_memorize_step,
                renderer=lambda payload: figures_eval.render_memorize_acceptance_matrix(
                    payload["data"], payload["step"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="memorize_acceptance_network",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=["memorize_edge_acceptance", "observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=self._load_memorize_step,
                renderer=lambda payload: figures_eval.render_memorize_acceptance_network(
                    payload["data"], payload["step"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="both_acceptance_network",
                category="stepwise",
                requires_step=True,
                requires_steps=False,
                required_step_metrics=[
                    "mhng_edge_acceptance",
                    "memorize_edge_acceptance",
                    "observations_and_creations_clustered",
                ],
                required_global_metrics=[],
                loader=self._load_both_acceptance_step,
                renderer=lambda payload: figures_eval.render_both_acceptance_network(
                    payload["data"], payload["step"]
                ),
            )
        )

        # Whole-step figures
        reg.register(
            FigureSpec(
                figure_id="wasserstein_mds_trajectory",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=[],
                required_global_metrics=["wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(),
                },
                renderer=lambda payload: figures_fgw.render_mds_trajectory(payload["data"], payload["cluster_data"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_mds_trajectory",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=[],
                required_global_metrics=["gromov_wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("gromov_wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(),
                },
                renderer=lambda payload: figures_fgw.render_mds_trajectory(payload["data"], payload["cluster_data"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="wasserstein_mds_trajectory_cluster",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=["wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(required=True),
                },
                renderer=lambda payload: figures_fgw.render_mds_trajectory_clustered(
                    payload["data"], payload["cluster_data"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="wasserstein_mds_pc_trajectory",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=[],
                required_global_metrics=["wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(),
                },
                renderer=lambda payload: figures_fgw.render_mds_pc_trajectory(payload["data"], payload["cluster_data"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_mds_pc_trajectory",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=[],
                required_global_metrics=["gromov_wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("gromov_wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(),
                },
                renderer=lambda payload: figures_fgw.render_mds_pc_trajectory(payload["data"], payload["cluster_data"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="wasserstein_mds_pc_trajectory_cluster",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=["wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(required=True),
                },
                renderer=lambda payload: figures_fgw.render_mds_pc_trajectory_clustered(
                    payload["data"], payload["cluster_data"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_mds_pc_trajectory_cluster",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=["gromov_wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("gromov_wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(required=True),
                },
                renderer=lambda payload: figures_fgw.render_mds_pc_trajectory_clustered(
                    payload["data"], payload["cluster_data"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_mds_pc_trajectory_cluster_segmented",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=["gromov_wasserstein_mds"],
                loader=self._load_gw_mds_pc_segmented_data,
                renderer=lambda payload: figures_fgw.render_mds_pc_trajectory_clustered_segmented(
                    payload["data"], payload["cluster_data"], payload["segment_boundaries"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_mds_trajectory_cluster",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=["gromov_wasserstein_mds"],
                loader=lambda p, step, steps: {
                    "data": p._load_global_metric_data("gromov_wasserstein_mds"),
                    "cluster_data": p._load_any_cluster_data(required=True),
                },
                renderer=lambda payload: figures_fgw.render_mds_trajectory_clustered(
                    payload["data"], payload["cluster_data"]
                ),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="observations_agents_clustered_panel",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=["observations_and_creations_clustered"],
                required_global_metrics=[],
                loader=self._load_panel_data,
                renderer=lambda payload: figures_summary.render_observations_agents_clustered_panel(payload["panel_items"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="wasserstein_similarity_snapshot",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=["wasserstein_similarity", "gromov_wasserstein_similarity"],
                required_global_metrics=[],
                loader=self._load_fgw_snapshot_data,
                renderer=lambda payload: figures_summary.render_wasserstein_similarity_snapshot(payload["panel_items"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_similarity_snapshot",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=["wasserstein_similarity", "gromov_wasserstein_similarity"],
                required_global_metrics=[],
                loader=self._load_fgw_snapshot_data,
                renderer=lambda payload: figures_summary.render_gromov_wasserstein_similarity_snapshot(payload["panel_items"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="wasserstein_similarity_average_snapshot",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=["wasserstein_similarity", "gromov_wasserstein_similarity"],
                required_global_metrics=[],
                loader=self._load_fgw_average_snapshot_data,
                renderer=lambda payload: figures_summary.render_wasserstein_similarity_average_snapshot(payload["panel_items"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="gromov_wasserstein_similarity_average_snapshot",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=["wasserstein_similarity", "gromov_wasserstein_similarity"],
                required_global_metrics=[],
                loader=self._load_fgw_average_snapshot_data,
                renderer=lambda payload: figures_summary.render_gromov_wasserstein_similarity_average_snapshot(payload["panel_items"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="both_acceptance_network_snapshot",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=[
                    "mhng_edge_acceptance",
                    "memorize_edge_acceptance",
                    "observations_and_creations_clustered",
                ],
                required_global_metrics=[],
                loader=self._load_both_acceptance_snapshot_data,
                renderer=lambda payload: figures_summary.render_both_acceptance_network_snapshot(payload["panel_items"]),
            )
        )
        reg.register(
            FigureSpec(
                figure_id="both_acceptance_average_network_snapshot",
                category="snapshot",
                requires_step=False,
                requires_steps=True,
                required_step_metrics=[
                    "mhng_edge_acceptance",
                    "memorize_edge_acceptance",
                    "observations_and_creations_clustered",
                ],
                required_global_metrics=[],
                loader=self._load_both_acceptance_average_snapshot_data,
                renderer=lambda payload: figures_summary.render_both_acceptance_average_network_snapshot(
                    payload["panel_items"]
                ),
            )
        )
        # RSA time series figures
        reg.register(
            FigureSpec(
                figure_id="rsa_clusters",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=["rsa_within_clusters"],
                required_global_metrics=[],
                loader=self._load_rsa_series,
                renderer=lambda payload: figures_rsa.render_rsa_clusters(payload),
            )
        )
        rsa_shape = self._detect_rsa_shape()
        if rsa_shape is not None:
            num_agents, _num_clusters = rsa_shape
            for agent_idx in range(num_agents):
                reg.register(
                    FigureSpec(
                        figure_id=f"rsa_agent_{agent_idx}",
                        category="whole_step",
                        requires_step=False,
                        requires_steps=False,
                        required_step_metrics=["rsa_within_clusters"],
                        required_global_metrics=[],
                        loader=self._load_rsa_series,
                        renderer=lambda payload, idx=agent_idx: figures_rsa.render_rsa_agent(payload, idx),
                    )
                )

        return reg

    def _step_metric_file(self, metric_id: str, step: int) -> Path:
        return self.paths.step_metrics_dir / metric_id / f"step_{step}.npz"

    def _global_metric_file(self, metric_id: str) -> Path:
        return self.paths.global_metrics_dir / f"{metric_id}.npz"

    def _load_step_metric_data(self, metric_id: str, step: int) -> Dict[str, Any]:
        payload = self.store.load_step_metric(metric_id, step)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid step metric payload for {metric_id} step={step}")
        return data

    def _load_global_metric_data(self, metric_id: str) -> Dict[str, Any]:
        payload = self.store.load_global_metric(metric_id)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid global metric payload for {metric_id}")
        return data

    def _available_steps_for_metric(self, metric_id: str) -> List[int]:
        metric_dir = self.paths.step_metrics_dir / metric_id
        if not metric_dir.exists():
            return []
        return sorted([parse_step(path) for path in metric_dir.glob("step_*.npz")])

    def _load_any_cluster_data(self, required: bool = False) -> Optional[Dict[str, Any]]:
        steps = self._available_steps_for_metric("observations_and_creations_clustered")
        if not steps:
            if required:
                raise MissingArtifactsError(["step_metrics:observations_and_creations_clustered:any_step"])
            return None
        return self._load_step_metric_data("observations_and_creations_clustered", steps[0])

    def _load_latent_scatter_aligned(
        self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]
    ) -> Dict[str, Any]:
        assert step is not None
        data = self._load_step_metric_data("latent_scatter_from_reference", step)

        alignment = self._load_global_metric_data("latent_scatter_pca_alignment")
        align_steps = np.asarray(alignment["steps"], dtype=np.int64)
        align_signs = np.asarray(alignment["signs"])
        idx = int(np.searchsorted(align_steps, step))
        if align_steps[idx] != step:
            raise RuntimeError(f"latent_scatter_pca_alignment has no entry for step {step}")
        signs = align_signs[idx]  # shape (2,)
        if not np.all(signs == 1.0):
            aligned = [
                np.asarray(lat, dtype=np.float64) * signs[np.newaxis, :]
                for lat in data["latents_2d"]
            ]
            data = {**data, "latents_2d": np.asarray(aligned, dtype=object)}

        return {"step": step, "data": data}

    def _load_step_fgw_pair(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        assert step is not None
        wasserstein_data = self._load_step_metric_data("wasserstein_similarity", step)
        gromov_data = self._load_step_metric_data("gromov_wasserstein_similarity", step)
        return {
            "step": step,
            "data": figures_eval.compose_fgw_pair(wasserstein_data, gromov_data),
        }

    def _load_mhng_step(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        assert step is not None
        data = self._load_step_metric_data("mhng_edge_acceptance", step)
        cluster_data = self._load_step_metric_data("observations_and_creations_clustered", step)
        return {"step": step, "data": figures_eval.with_cluster_meta(data, cluster_data)}

    def _load_memorize_step(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        assert step is not None
        data = self._load_step_metric_data("memorize_edge_acceptance", step)
        cluster_data = self._load_step_metric_data("observations_and_creations_clustered", step)
        return {"step": step, "data": figures_eval.with_cluster_meta(data, cluster_data)}

    def _load_both_acceptance_step(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        assert step is not None
        mhng_data = self._load_step_metric_data("mhng_edge_acceptance", step)
        memorize_data = self._load_step_metric_data("memorize_edge_acceptance", step)
        cluster_data = self._load_step_metric_data("observations_and_creations_clustered", step)
        return {
            "step": step,
            "data": figures_eval.compose_acceptance_pair(
                figures_eval.with_cluster_meta(mhng_data, cluster_data),
                figures_eval.with_cluster_meta(memorize_data, cluster_data),
            ),
        }

    def _load_panel_data(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        if not steps:
            raise ValueError("observations_agents_clustered_panel requires --steps")
        panel_items: List[Tuple[int, Dict[str, Any]]] = []
        for step_id in steps:
            panel_items.append((step_id, self._load_step_metric_data("observations_and_creations_clustered", step_id)))
        return {"panel_items": panel_items}

    def _load_fgw_snapshot_data(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        if not steps:
            raise ValueError("similarity snapshot figures require --steps")
        panel_items: List[Tuple[int, Dict[str, Any]]] = []
        for step_id in steps:
            wasserstein_data = self._load_step_metric_data("wasserstein_similarity", step_id)
            gromov_data = self._load_step_metric_data("gromov_wasserstein_similarity", step_id)
            panel_items.append((step_id, figures_eval.compose_fgw_pair(wasserstein_data, gromov_data)))
        return {"panel_items": panel_items}

    def _load_gw_mds_pc_segmented_data(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        if not steps:
            raise ValueError("gromov_wasserstein_mds_pc_trajectory_cluster_segmented requires --steps")
        return {
            "data": self._load_global_metric_data("gromov_wasserstein_mds"),
            "cluster_data": self._load_any_cluster_data(required=True),
            "segment_boundaries": list(steps),
        }

    def _average_similarity_window(
        self,
        metric_id: str,
        start_step_exclusive: int,
        end_step_inclusive: int,
    ) -> Dict[str, Any]:
        available_steps = self._available_steps_for_metric(metric_id)
        window_steps = [s for s in available_steps if start_step_exclusive < s <= end_step_inclusive]
        if not window_steps:
            raise MissingArtifactsError(
                [
                    f"step_metrics:{metric_id}:window_{start_step_exclusive + 1}_{end_step_inclusive}",
                ]
            )

        matrix_stack: List[np.ndarray] = []
        for step_id in window_steps:
            data = self._load_step_metric_data(metric_id, step_id)
            matrix_stack.append(np.asarray(data["distance_matrix"], dtype=float))

        matrices = np.stack(matrix_stack, axis=0)
        valid_counts = np.sum(np.isfinite(matrices), axis=0)
        mean_matrix = np.divide(
            np.nansum(matrices, axis=0),
            valid_counts,
            out=np.full_like(matrices[0], np.nan, dtype=float),
            where=valid_counts > 0,
        )

        return {"distance_matrix": mean_matrix}

    def _load_fgw_average_snapshot_data(
        self,
        pipeline: "FigurePipeline",
        step: Optional[int],
        steps: Optional[List[int]],
    ) -> Dict[str, Any]:
        if not steps:
            raise ValueError("similarity average snapshot figures require --steps")

        boundaries = sorted({int(step_id) for step_id in steps})
        panel_items: List[Tuple[str, Dict[str, Any]]] = []
        prev_end = 0

        for end_step in boundaries:
            if end_step <= prev_end:
                raise ValueError("--steps for similarity average snapshot figures must be strictly increasing")

            wasserstein_avg = self._average_similarity_window("wasserstein_similarity", prev_end, end_step)
            gromov_avg = self._average_similarity_window("gromov_wasserstein_similarity", prev_end, end_step)
            panel_label = f"{prev_end + 1}-{end_step} step"
            panel_items.append((panel_label, figures_eval.compose_fgw_pair(wasserstein_avg, gromov_avg)))
            prev_end = end_step

        return {"panel_items": panel_items}

    def _load_both_acceptance_snapshot_data(
        self,
        pipeline: "FigurePipeline",
        step: Optional[int],
        steps: Optional[List[int]],
    ) -> Dict[str, Any]:
        if not steps:
            raise ValueError("both_acceptance_network_snapshot requires --steps")
        panel_items: List[Tuple[int, Dict[str, Any]]] = []
        for step_id in steps:
            mhng_data = self._load_step_metric_data("mhng_edge_acceptance", step_id)
            memorize_data = self._load_step_metric_data("memorize_edge_acceptance", step_id)
            cluster_data = self._load_step_metric_data("observations_and_creations_clustered", step_id)
            panel_items.append(
                (
                    step_id,
                    figures_eval.compose_acceptance_pair(
                        figures_eval.with_cluster_meta(mhng_data, cluster_data),
                        figures_eval.with_cluster_meta(memorize_data, cluster_data),
                    ),
                )
            )
        return {"panel_items": panel_items}

    def _average_acceptance_window(
        self,
        metric_id: str,
        start_step_exclusive: int,
        end_step_inclusive: int,
    ) -> Dict[str, Any]:
        available_steps = self._available_steps_for_metric(metric_id)
        window_steps = [s for s in available_steps if start_step_exclusive < s <= end_step_inclusive]
        if not window_steps:
            raise MissingArtifactsError(
                [
                    f"step_metrics:{metric_id}:window_{start_step_exclusive + 1}_{end_step_inclusive}",
                ]
            )

        first_data = self._load_step_metric_data(metric_id, window_steps[0])
        adjacency_matrix = np.asarray(first_data.get("adjacency_matrix"))

        proposed_sum = np.zeros_like(np.asarray(first_data["proposed_counts"], dtype=np.int64))
        accepted_sum = np.zeros_like(np.asarray(first_data["accepted_counts"], dtype=np.int64))
        zero_accept_counts = np.zeros_like(np.asarray(first_data["accepted_counts"], dtype=np.int64))
        rate_stack: List[np.ndarray] = []

        for step_id in window_steps:
            data = self._load_step_metric_data(metric_id, step_id)
            proposed_step = np.asarray(data["proposed_counts"], dtype=np.int64)
            accepted_step = np.asarray(data["accepted_counts"], dtype=np.int64)
            proposed_sum += proposed_step
            accepted_sum += accepted_step
            zero_accept_counts += ((proposed_step > 0) & (accepted_step == 0)).astype(np.int64)
            rate_stack.append(np.asarray(data["accept_rates"], dtype=float))

        rates = np.stack(rate_stack, axis=0)
        valid_counts = np.sum(np.isfinite(rates), axis=0)
        mean_rates = np.divide(
            np.nansum(rates, axis=0),
            valid_counts,
            out=np.full_like(rates[0], np.nan, dtype=float),
            where=valid_counts > 0,
        )
        zero_accept_fraction = np.asarray(zero_accept_counts, dtype=float) / float(len(window_steps))

        return {
            "adjacency_matrix": adjacency_matrix,
            "proposed_counts": proposed_sum,
            "accepted_counts": accepted_sum,
            "accept_rates": mean_rates,
            "zero_accept_fraction": zero_accept_fraction,
        }

    def _resolve_cluster_step_for_window(self, start_step_exclusive: int, end_step_inclusive: int) -> int:
        cluster_steps = self._available_steps_for_metric("observations_and_creations_clustered")
        in_window = [s for s in cluster_steps if start_step_exclusive < s <= end_step_inclusive]
        if in_window:
            return in_window[-1]

        older = [s for s in cluster_steps if s <= end_step_inclusive]
        if older:
            return older[-1]

        raise MissingArtifactsError(["step_metrics:observations_and_creations_clustered:any_step"])

    def _load_both_acceptance_average_snapshot_data(
        self,
        pipeline: "FigurePipeline",
        step: Optional[int],
        steps: Optional[List[int]],
    ) -> Dict[str, Any]:
        if not steps:
            raise ValueError("both_acceptance_average_network_snapshot requires --steps")

        boundaries = sorted({int(step_id) for step_id in steps})
        panel_items: List[Tuple[str, Dict[str, Any]]] = []
        prev_end = 0

        for end_step in boundaries:
            if end_step <= prev_end:
                raise ValueError("--steps for both_acceptance_average_network_snapshot must be strictly increasing")

            mhng_avg = self._average_acceptance_window("mhng_edge_acceptance", prev_end, end_step)
            memorize_avg = self._average_acceptance_window("memorize_edge_acceptance", prev_end, end_step)

            cluster_step = self._resolve_cluster_step_for_window(prev_end, end_step)
            cluster_data = self._load_step_metric_data("observations_and_creations_clustered", cluster_step)

            panel_label = f"{prev_end + 1}-{end_step} step"
            pair_data = figures_eval.compose_acceptance_pair(
                figures_eval.with_cluster_meta(mhng_avg, cluster_data),
                figures_eval.with_cluster_meta(memorize_avg, cluster_data),
            )
            panel_items.append(
                (
                    panel_label,
                    pair_data,
                )
            )
            prev_end = end_step

        return {"panel_items": panel_items}

    def _detect_rsa_shape(self) -> Optional[Tuple[int, int]]:
        """Returns (num_agents, num_clusters) from first available RSA metric step, or None."""
        steps = self._available_steps_for_metric("rsa_within_clusters")
        if not steps:
            return None
        try:
            data = self._load_step_metric_data("rsa_within_clusters", steps[0])
            agent_rsa = np.asarray(data.get("agent_rsa", []))
            cluster_rsa = np.asarray(data.get("cluster_rsa", []))
            if agent_rsa.ndim != 1 or cluster_rsa.ndim != 1:
                return None
            return int(agent_rsa.shape[0]), int(cluster_rsa.shape[0])
        except Exception:
            return None

    def _load_rsa_series(self, pipeline: "FigurePipeline", step: Optional[int], steps: Optional[List[int]]) -> Dict[str, Any]:
        available = self._available_steps_for_metric("rsa_within_clusters")
        if not available:
            raise MissingArtifactsError(["step_metrics:rsa_within_clusters:any_step"])
        agent_rsa_list = []
        cluster_rsa_list = []
        neighbor_latent_rsa_list = []
        agent_to_cluster: Optional[List[int]] = None
        num_clusters = 0
        for s in available:
            data = self._load_step_metric_data("rsa_within_clusters", s)
            agent_rsa_list.append(np.asarray(data.get("agent_rsa", [])))
            cluster_rsa_list.append(np.asarray(data.get("cluster_rsa", [])))
            neighbor_latent_rsa_list.append(np.asarray(data.get("neighbor_latent_rsa", [])))
            if agent_to_cluster is None:
                cluster_agents = data.get("cluster_agents", {})
                num_agents = int(np.asarray(data.get("agent_rsa", [])).shape[0])
                agent_to_cluster = [0] * num_agents
                if isinstance(cluster_agents, dict):
                    for cluster_id, members in cluster_agents.items():
                        cluster_idx = int(cluster_id)
                        for agent_idx in members:
                            agent_id = int(agent_idx)
                            if 0 <= agent_id < num_agents:
                                agent_to_cluster[agent_id] = cluster_idx
                    num_clusters = max(len(cluster_agents), max(agent_to_cluster) + 1 if agent_to_cluster else 1)
                else:
                    num_clusters = int(np.asarray(data.get("cluster_rsa", [])).shape[0])
        try:
            agent_rsa = np.stack(agent_rsa_list, axis=0)
        except ValueError:
            agent_rsa = np.array(agent_rsa_list, dtype=object)
        try:
            cluster_rsa = np.stack(cluster_rsa_list, axis=0)
        except ValueError:
            cluster_rsa = np.array(cluster_rsa_list, dtype=object)
        try:
            neighbor_latent_rsa = np.stack(neighbor_latent_rsa_list, axis=0)
        except ValueError:
            neighbor_latent_rsa = np.array(neighbor_latent_rsa_list, dtype=object)
        return {
            "steps": available,
            "agent_rsa": agent_rsa,
            "cluster_rsa": cluster_rsa,
            "neighbor_latent_rsa": neighbor_latent_rsa,
            "agent_to_cluster": agent_to_cluster or [],
            "num_clusters": int(num_clusters),
        }

    def _validate_dependencies(self, spec: FigureSpec, step: Optional[int]) -> None:
        missing: List[str] = []

        if spec.requires_step and step is None:
            raise ValueError(f"figure '{spec.figure_id}' requires --step")

        if spec.requires_steps and not self.config.steps:
            raise ValueError(f"figure '{spec.figure_id}' requires --steps")

        for metric_id in spec.required_global_metrics:
            if not self._global_metric_file(metric_id).exists():
                missing.append(f"global_metrics:{metric_id}")

        for metric_id in spec.required_step_metrics:
            if step is None:
                if not self._available_steps_for_metric(metric_id):
                    missing.append(f"step_metrics:{metric_id}:any_step")
            else:
                if not self._step_metric_file(metric_id, step).exists():
                    missing.append(f"step_metrics:{metric_id}:step_{step}")

        if spec.requires_steps and self.config.steps:
            for step_id in self.config.steps:
                for metric_id in spec.required_step_metrics:
                    if not self._step_metric_file(metric_id, step_id).exists():
                        missing.append(f"step_metrics:{metric_id}:step_{step_id}")

        if missing:
            raise MissingArtifactsError(missing)

    def _figure_base_path(self, spec: FigureSpec, step: Optional[int], steps: Optional[List[int]]) -> Path:
        out_dir = self.paths.figures_dir / spec.figure_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if spec.category == "stepwise":
            assert step is not None
            return out_dir / f"step_{step}"
        if spec.category == "snapshot" and steps:
            step_suffix = "_".join(str(v) for v in steps)
            return out_dir / f"steps_{step_suffix}"
        return out_dir / "figure"

    def _save_figure(self, fig: matplotlib.figure.Figure, base_path: Path) -> Path:
        png_path = Path(str(base_path) + ".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(Path(str(base_path) + ".pdf"), bbox_inches="tight")
        fig.savefig(Path(str(base_path) + ".svg"), bbox_inches="tight")
        plt.close(fig)
        return png_path

    def render(self) -> Path:
        self.store.ensure_layout()
        spec = self.registry.get(self.config.figure_id)
        if spec.category in {"whole_step", "snapshot"} and self.config.step is not None:
            print(f"[render] warning: ignoring --step for {spec.category} figure '{spec.figure_id}'")

        step = self.config.step if spec.category == "stepwise" else None
        self._validate_dependencies(spec, step)

        base_path = self._figure_base_path(spec, step, self.config.steps)
        png_path = Path(str(base_path) + ".png")
        if png_path.exists() and not self.config.refresh:
            return png_path

        payload = spec.loader(self, step, self.config.steps)
        fig = spec.renderer(payload)
        return self._save_figure(fig, base_path)

    def render_frames(self, steps: Optional[List[int]]) -> List[Path]:
        resolved_steps = self.resolve_frame_steps(steps)

        paths: List[Path] = []
        for step in resolved_steps:
            frame_config = RenderRunConfig(
                input_dir=self.config.input_dir,
                figure_id=self.config.figure_id,
                step=step,
                refresh=self.config.refresh,
            )
            paths.append(FigurePipeline(frame_config).render())
        return paths

    def resolve_frame_steps(self, steps: Optional[List[int]]) -> List[int]:
        spec = self.registry.get(self.config.figure_id)
        if spec.category != "stepwise":
            raise ValueError("render-frames is only supported for stepwise figures")

        if spec.frame_steps_resolver is not None:
            return list(spec.frame_steps_resolver(self, steps))

        resolved_steps = steps
        if resolved_steps is None:
            step_sets = [set(self._available_steps_for_metric(metric_id)) for metric_id in spec.required_step_metrics]
            common_steps = sorted(set.intersection(*step_sets)) if step_sets else []
            if not common_steps:
                missing = [f"step_metrics:{metric_id}:any_step" for metric_id in spec.required_step_metrics]
                raise MissingArtifactsError(missing)
            resolved_steps = common_steps
        return list(resolved_steps)

    def status(self) -> Dict[str, Any]:
        categorized: Dict[str, List[str]] = {}
        for figure_id in self.registry.list_ids():
            category = self.registry.get(figure_id).category
            categorized.setdefault(category, []).append(figure_id)
        return {
            "figures": self.registry.list_ids(),
            "figures_by_category": categorized,
            "figures_dir": str(self.paths.figures_dir),
        }


def _resolve_worker_count(task_count: int, num_workers: Optional[int]) -> int:
    if task_count <= 1:
        return 1
    if num_workers is None:
        return min(task_count, max(1, (os.cpu_count() or 1) - 1))
    return min(task_count, max(1, int(num_workers)))


def _render_task_worker(
    task: Tuple[Path, str, Optional[int], Optional[List[int]], bool],
) -> Tuple[str, Optional[int], str, Optional[str], Optional[List[str]], Optional[str]]:
    input_dir, figure_id, step, steps, refresh = task
    run_config = RenderRunConfig(
        input_dir=input_dir,
        figure_id=figure_id,
        step=step,
        steps=steps,
        refresh=refresh,
    )
    try:
        output = FigurePipeline(run_config).render()
        return (figure_id, step, "ok", str(output), None, None)
    except MissingArtifactsError as exc:
        return (figure_id, step, "missing", None, list(exc.missing), None)
    except Exception as exc:
        return (figure_id, step, "error", None, None, str(exc))


def _run_render_tasks_with_progress(
    tasks: List[Tuple[Path, str, Optional[int], Optional[List[int]], bool]],
    num_workers: Optional[int],
    label: str,
) -> List[Tuple[str, Optional[int], str, Optional[str], Optional[List[str]], Optional[str]]]:
    total = len(tasks)
    if total == 0:
        print(f"[{label}] no tasks to run")
        return []

    worker_count = _resolve_worker_count(total, num_workers)
    print(f"[{label}] starting {total} task(s) with {worker_count} worker(s)")

    results: List[Tuple[str, Optional[int], str, Optional[str], Optional[List[str]], Optional[str]]] = []

    if worker_count == 1:
        with tqdm(total=total, desc=label, unit="task", dynamic_ncols=True) as pbar:
            for task in tasks:
                results.append(_render_task_worker(task))
                pbar.update(1)
        return results

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=worker_count) as pool:
        with tqdm(total=total, desc=label, unit="task", dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(_render_task_worker, tasks):
                results.append(result)
                pbar.update(1)

    return results


class FigureBatchRunner:
    @staticmethod
    def render_many(
        base_config: RenderRunConfig,
        figure_ids: Sequence[str],
        num_workers: Optional[int] = None,
    ) -> Tuple[Dict[str, Path], Dict[str, List[str]], Dict[str, str]]:
        tasks: List[Tuple[Path, str, Optional[int], Optional[List[int]], bool]] = [
            (
                base_config.input_dir,
                figure_id,
                base_config.step,
                base_config.steps,
                base_config.refresh,
            )
            for figure_id in figure_ids
        ]
        rendered: Dict[str, Path] = {}
        missing_by_figure: Dict[str, List[str]] = {}
        error_by_figure: Dict[str, str] = {}

        results = _run_render_tasks_with_progress(tasks, num_workers, label="render")

        for figure_id, _step, status, output, missing, error in results:
            if status == "ok" and output is not None:
                rendered[figure_id] = Path(output)
            elif status == "missing" and missing is not None:
                missing_by_figure[figure_id] = missing
            elif status == "error" and error is not None:
                error_by_figure[figure_id] = error

        return rendered, missing_by_figure, error_by_figure

    @staticmethod
    def render_frames_many(
        input_dir: Path,
        figure_ids: Sequence[str],
        steps: Optional[List[int]],
        refresh: bool,
        num_workers: Optional[int] = None,
    ) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, str]]:
        rendered_counts: Dict[str, int] = {}
        missing_by_figure: Dict[str, List[str]] = {}
        error_by_figure: Dict[str, str] = {}
        tasks: List[Tuple[Path, str, Optional[int], Optional[List[int]], bool]] = []

        for figure_id in figure_ids:
            run_config = RenderRunConfig(
                input_dir=input_dir,
                figure_id=figure_id,
                refresh=refresh,
            )
            try:
                resolved_steps = FigurePipeline(run_config).resolve_frame_steps(steps)
            except MissingArtifactsError as exc:
                missing_by_figure[figure_id] = exc.missing
                continue
            except Exception as exc:
                error_by_figure[figure_id] = str(exc)
                continue

            if not resolved_steps:
                continue

            rendered_counts[figure_id] = 0
            for step in resolved_steps:
                tasks.append((input_dir, figure_id, step, None, refresh))

        results = _run_render_tasks_with_progress(tasks, num_workers, label="render-frames")

        for figure_id, step, status, _output, missing, error in results:
            if status == "ok":
                rendered_counts[figure_id] = rendered_counts.get(figure_id, 0) + 1
            elif status == "missing" and missing is not None:
                if figure_id not in missing_by_figure:
                    missing_by_figure[figure_id] = []
                missing_by_figure[figure_id].extend(missing)
            elif status == "error" and error is not None:
                if step is None:
                    error_by_figure[figure_id] = error
                else:
                    error_by_figure[figure_id] = f"step_{step}: {error}"

        # Keep missing items compact and deterministic for CLI reporting.
        for figure_id, missing in list(missing_by_figure.items()):
            missing_by_figure[figure_id] = sorted(set(missing))

        return rendered_counts, missing_by_figure, error_by_figure


def _render_multi_seed_task_worker(
    task: Tuple[Tuple[Path, ...], Path, str, bool],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    input_dirs, output_dir, figure_id, refresh = task
    run_config = RenderSeedsRunConfig(
        input_dirs=list(input_dirs),
        output_dir=output_dir,
        figure_ids=[figure_id],
        refresh=refresh,
        num_workers=1,
    )
    try:
        output = MultiSeedFigurePipeline(run_config).render(figure_id)
        return (figure_id, "ok", str(output), None)
    except MissingArtifactsError as exc:
        return (figure_id, "missing", None, f"missing: {', '.join(exc.missing)}")
    except Exception as exc:
        return (figure_id, "error", None, str(exc))


class MultiSeedFigurePipeline:
    def __init__(self, config: RenderSeedsRunConfig):
        self.config = config
        self.stores = [
            ArtifactStore(AnalysisPaths.from_input_dir(input_dir))
            for input_dir in config.input_dirs
        ]
        self.registry = self._build_registry()

    def _available_steps_for_seed(self, store: ArtifactStore, metric_id: str) -> List[int]:
        metric_dir = store.paths.step_metrics_dir / metric_id
        if not metric_dir.exists():
            return []
        return sorted([parse_step(path) for path in metric_dir.glob("step_*.npz")])

    def _detect_rsa_shape(self) -> Optional[Tuple[int, int]]:
        detected_shape: Optional[Tuple[int, int]] = None
        for store in self.stores:
            steps = self._available_steps_for_seed(store, "rsa_within_clusters")
            if not steps:
                continue
            try:
                payload = store.load_step_metric("rsa_within_clusters", steps[0])
            except Exception:
                return None
            data = payload.get("data")
            if not isinstance(data, dict):
                return None
            agent_rsa = np.asarray(data.get("agent_rsa", []))
            cluster_rsa = np.asarray(data.get("cluster_rsa", []))
            if agent_rsa.ndim != 1 or cluster_rsa.ndim != 1:
                return None
            shape = (int(agent_rsa.shape[0]), int(cluster_rsa.shape[0]))
            if detected_shape is None:
                detected_shape = shape
            elif detected_shape != shape:
                return None
        return detected_shape

    def _build_registry(self) -> FigureRegistry:
        reg = FigureRegistry()
        rsa_shape = self._detect_rsa_shape()
        if rsa_shape is None:
            return reg

        num_agents, _num_clusters = rsa_shape
        reg.register(
            FigureSpec(
                figure_id="rsa_clusters_seeds",
                category="whole_step",
                requires_step=False,
                requires_steps=False,
                required_step_metrics=["rsa_within_clusters"],
                required_global_metrics=[],
                loader=lambda p, step, steps: p._load_rsa_seeds(),
                renderer=lambda payload: figures_rsa.render_rsa_clusters_seeds(payload),
            )
        )
        for agent_idx in range(num_agents):
            reg.register(
                FigureSpec(
                    figure_id=f"rsa_agent_{agent_idx}_seeds",
                    category="whole_step",
                    requires_step=False,
                    requires_steps=False,
                    required_step_metrics=["rsa_within_clusters"],
                    required_global_metrics=[],
                    loader=lambda p, step, steps, idx=agent_idx: p._load_rsa_seeds(),
                    renderer=lambda payload, idx=agent_idx: figures_rsa.render_rsa_agent_seeds(payload, idx),
                )
            )

        return reg

    def _load_rsa_seeds(self) -> Dict[str, Any]:
        if not self.stores:
            raise MissingArtifactsError(["rsa_within_clusters:no_common_steps"])

        step_sets = [set(self._available_steps_for_seed(store, "rsa_within_clusters")) for store in self.stores]
        common_steps = sorted(set.intersection(*step_sets)) if step_sets else []
        if not common_steps:
            raise MissingArtifactsError(["rsa_within_clusters:no_common_steps"])

        expected_shape: Optional[Tuple[int, int]] = None
        agent_rsa_by_seed: List[np.ndarray] = []
        cluster_rsa_by_seed: List[np.ndarray] = []
        neighbor_cluster_rsa_by_seed: List[np.ndarray] = []

        for store in self.stores:
            agent_rsa_list: List[np.ndarray] = []
            cluster_rsa_list: List[np.ndarray] = []
            neighbor_cluster_rsa_list: List[np.ndarray] = []
            for step in common_steps:
                payload = store.load_step_metric("rsa_within_clusters", step)
                data = payload.get("data")
                if not isinstance(data, dict):
                    raise RuntimeError(f"Invalid step metric payload for rsa_within_clusters step={step}")
                agent_rsa = np.asarray(data.get("agent_rsa", []), dtype=np.float64)
                cluster_rsa = np.asarray(data.get("cluster_rsa", []), dtype=np.float64)
                neighbor_agent_rsa = np.asarray(data.get("neighbor_latent_rsa", []), dtype=np.float64)
                if agent_rsa.ndim != 1 or cluster_rsa.ndim != 1:
                    raise ValueError(
                        f"Inconsistent RSA shape across seeds: step={step} agent_rsa.ndim={agent_rsa.ndim} cluster_rsa.ndim={cluster_rsa.ndim}"
                    )
                if neighbor_agent_rsa.ndim != 1:
                    raise ValueError(
                        f"Inconsistent Neighbor RSA shape across seeds: step={step} neighbor_latent_rsa.ndim={neighbor_agent_rsa.ndim}"
                    )
                current_shape = (int(agent_rsa.shape[0]), int(cluster_rsa.shape[0]))
                if expected_shape is None:
                    expected_shape = current_shape
                elif expected_shape != current_shape:
                    raise ValueError(
                        "Inconsistent RSA shape across seeds: "
                        f"expected num_agents={expected_shape[0]}, num_clusters={expected_shape[1]} but got "
                        f"num_agents={current_shape[0]}, num_clusters={current_shape[1]}"
                    )

                num_agents = int(agent_rsa.shape[0])
                num_clusters = int(cluster_rsa.shape[0])
                if neighbor_agent_rsa.shape[0] != num_agents:
                    padded_neighbor = np.full(num_agents, np.nan, dtype=np.float64)
                    limit = min(num_agents, int(neighbor_agent_rsa.shape[0]))
                    padded_neighbor[:limit] = neighbor_agent_rsa[:limit]
                    neighbor_agent_rsa = padded_neighbor

                cluster_agents = data.get("cluster_agents", {})
                members_by_cluster: List[List[int]] = [[] for _ in range(max(num_clusters, 1))]
                if isinstance(cluster_agents, dict) and cluster_agents:
                    for cluster_id, members in cluster_agents.items():
                        cluster_idx = int(cluster_id)
                        if not (0 <= cluster_idx < len(members_by_cluster)):
                            continue
                        for agent_idx in members:
                            agent_id = int(agent_idx)
                            if 0 <= agent_id < num_agents:
                                members_by_cluster[cluster_idx].append(agent_id)
                else:
                    members_by_cluster[0] = list(range(num_agents))

                neighbor_cluster_rsa = np.full(max(num_clusters, 1), np.nan, dtype=np.float64)
                for cluster_idx, members in enumerate(members_by_cluster):
                    if not members:
                        continue
                    subset = neighbor_agent_rsa[np.asarray(members, dtype=int)]
                    if np.isfinite(subset).any():
                        neighbor_cluster_rsa[cluster_idx] = float(np.nanmean(subset))

                agent_rsa_list.append(agent_rsa)
                cluster_rsa_list.append(cluster_rsa)
                neighbor_cluster_rsa_list.append(neighbor_cluster_rsa)
            agent_rsa_by_seed.append(np.stack(agent_rsa_list, axis=0))
            cluster_rsa_by_seed.append(np.stack(cluster_rsa_list, axis=0))
            neighbor_cluster_rsa_by_seed.append(np.stack(neighbor_cluster_rsa_list, axis=0))

        agent_rsa_seeds = np.stack(agent_rsa_by_seed, axis=0)
        cluster_rsa_seeds = np.stack(cluster_rsa_by_seed, axis=0)
        neighbor_cluster_rsa_seeds = np.stack(neighbor_cluster_rsa_by_seed, axis=0)

        return {
            "steps": common_steps,
            "cluster_rsa_seeds": cluster_rsa_seeds,
            "cluster_rsa_mean": np.nanmean(cluster_rsa_seeds, axis=0),
            "cluster_rsa_std": np.nanstd(cluster_rsa_seeds, axis=0),
            "neighbor_cluster_rsa_seeds": neighbor_cluster_rsa_seeds,
            "neighbor_cluster_rsa_mean": np.nanmean(neighbor_cluster_rsa_seeds, axis=0),
            "neighbor_cluster_rsa_std": np.nanstd(neighbor_cluster_rsa_seeds, axis=0),
            "agent_rsa_seeds": agent_rsa_seeds,
            "agent_rsa_mean": np.nanmean(agent_rsa_seeds, axis=0),
            "agent_rsa_std": np.nanstd(agent_rsa_seeds, axis=0),
        }


    def _figure_base_path(self, figure_id: str) -> Path:
        out_dir = self.config.output_dir / "analysis_figures" / figure_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "figure"

    def _save_figure(self, fig: matplotlib.figure.Figure, base_path: Path) -> Path:
        png_path = Path(str(base_path) + ".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(Path(str(base_path) + ".pdf"), bbox_inches="tight")
        fig.savefig(Path(str(base_path) + ".svg"), bbox_inches="tight")
        plt.close(fig)
        return png_path

    def render(self, figure_id: str) -> Path:
        spec = self.registry.get(figure_id)
        base_path = self._figure_base_path(figure_id)
        png_path = Path(str(base_path) + ".png")
        if png_path.exists() and not self.config.refresh:
            return png_path
        payload = spec.loader(self, None, None)
        fig = spec.renderer(payload)
        return self._save_figure(fig, base_path)

    def render_many(
        self,
        figure_ids: List[str],
        num_workers: int = 1,
    ) -> Tuple[Dict[str, Path], Dict[str, str]]:
        rendered: Dict[str, Path] = {}
        error_by_figure: Dict[str, str] = {}

        if num_workers <= 1 or len(figure_ids) <= 1:
            for figure_id in figure_ids:
                try:
                    rendered[figure_id] = self.render(figure_id)
                except MissingArtifactsError as exc:
                    error_by_figure[figure_id] = f"missing: {', '.join(exc.missing)}"
                except Exception as exc:
                    error_by_figure[figure_id] = str(exc)
            return rendered, error_by_figure

        tasks = [
            (tuple(self.config.input_dirs), self.config.output_dir, figure_id, self.config.refresh)
            for figure_id in figure_ids
        ]
        worker_count = _resolve_worker_count(len(tasks), num_workers)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            with tqdm(total=len(tasks), desc="render-seeds", unit="task", dynamic_ncols=True) as pbar:
                for figure_id, status, output, error in pool.imap_unordered(_render_multi_seed_task_worker, tasks):
                    if status == "ok" and output is not None:
                        rendered[figure_id] = Path(output)
                    elif error is not None:
                        error_by_figure[figure_id] = error
                    pbar.update(1)
        return rendered, error_by_figure

    def list_ids(self) -> List[str]:
        return self.registry.list_ids()

