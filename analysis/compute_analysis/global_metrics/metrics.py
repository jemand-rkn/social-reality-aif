from __future__ import annotations

import multiprocessing as mp
import signal
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from analysis.core.store import ArtifactStore
from analysis.core.types import AnalysisPaths, GlobalMetricRunConfig
from analysis.data_extraction.rebuild_society import parse_step


def _get_mp_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


@dataclass
class GlobalRuntimeContext:
    config: GlobalMetricRunConfig
    paths: AnalysisPaths
    store: ArtifactStore


def _cache_get(runtime: Optional[Dict[str, Any]], key: str):
    if runtime is None:
        return None
    cache = runtime.setdefault("cache", {})
    return cache.get(key)


def _cache_set(runtime: Optional[Dict[str, Any]], key: str, value):
    if runtime is None:
        return value
    cache = runtime.setdefault("cache", {})
    cache[key] = value
    return value


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _extract_metric_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return {}



def _load_fgw_distance_matrices(context: GlobalRuntimeContext) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    dir_00 = context.paths.step_metrics_dir / "wasserstein_similarity"
    dir_10 = context.paths.step_metrics_dir / "gromov_wasserstein_similarity"
    files_00 = sorted(dir_00.glob("step_*.npz"), key=parse_step)
    if not files_00:
        raise FileNotFoundError(f"No wasserstein_similarity step metrics found in {dir_00}.")

    steps: List[int] = []
    matrices_00: List[np.ndarray] = []
    matrices_10: List[np.ndarray] = []

    for path_00 in files_00:
        step = parse_step(path_00)
        if not (dir_10 / f"step_{step}.npz").exists():
            continue

        p00 = context.store.load_step_metric("wasserstein_similarity", step)
        p10 = context.store.load_step_metric("gromov_wasserstein_similarity", step)
        d00 = _extract_metric_data(p00).get("distance_matrix")
        d10 = _extract_metric_data(p10).get("distance_matrix")
        if d00 is None or d10 is None:
            continue

        m00 = np.asarray(d00, dtype=np.float64)
        m10 = np.asarray(d10, dtype=np.float64)
        m00 = np.maximum(0.0, 0.5 * (m00 + m00.T))
        m10 = np.maximum(0.0, 0.5 * (m10 + m10.T))

        steps.append(step)
        matrices_00.append(m00)
        matrices_10.append(m10)

    if not steps:
        raise RuntimeError("No paired FGW distance matrices found in step metrics.")

    return steps, matrices_00, matrices_10


def _classical_mds(distance: np.ndarray, n_components: int = 2) -> np.ndarray:
    dist = np.asarray(distance, dtype=np.float64)
    n = dist.shape[0]
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    if n == 1:
        return np.zeros((1, n_components), dtype=np.float64)

    d2 = dist**2
    j = np.eye(n) - np.ones((n, n)) / float(n)
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    k = min(n_components, eigvals.size)
    coords = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    if k < n_components:
        coords = np.hstack([coords, np.zeros((n, n_components - k), dtype=coords.dtype)])
    return coords


def _compute_mds_eigvals(distance: np.ndarray) -> np.ndarray:
    dist = np.asarray(distance, dtype=np.float64)
    n = dist.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    d2 = dist**2
    j = np.eye(n) - np.ones((n, n)) / float(n)
    b = -0.5 * j @ d2 @ j
    eigvals = np.linalg.eigvalsh(b)
    eigvals = np.sort(eigvals)[::-1]
    return np.maximum(eigvals, 0.0)


def _mds_with_init(
    distance: np.ndarray,
    init: Optional[np.ndarray],
    n_components: int = 2,
    random_state: int = 0,
) -> np.ndarray:
    base = _classical_mds(distance, n_components=n_components)
    try:
        from sklearn.manifold import MDS
    except Exception:
        return base

    try:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            max_iter=500,
            n_init=1,
            random_state=random_state,
        )
        return mds.fit_transform(distance, init=init if init is not None else base)
    except Exception:
        return base


def _align_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    u, _, vt = np.linalg.svd(x_centered.T @ y_centered, full_matrices=False)
    r = u @ vt
    return x_centered @ r + y_mean


def _align_fgw_positions(positions: np.ndarray) -> np.ndarray:
    aligned = positions.copy()
    for idx in range(1, aligned.shape[0]):
        aligned[idx] = _align_to_reference(aligned[idx], aligned[idx - 1])
    return aligned


def _project_positions_to_global_pca(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    positions_np = np.asarray(positions, dtype=np.float64)
    if positions_np.ndim != 3 or positions_np.shape[-1] != 2:
        raise ValueError(f"Expected positions with shape [T, A, 2], got {positions_np.shape}.")

    flat = positions_np.reshape(-1, 2)
    finite_mask = np.all(np.isfinite(flat), axis=1)
    valid = flat[finite_mask]

    if valid.shape[0] < 2:
        return positions_np.copy(), np.array([np.nan, np.nan], dtype=np.float64)

    mean = valid.mean(axis=0, keepdims=True)
    centered = valid - mean
    cov = (centered.T @ centered) / float(max(1, valid.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    components = eigvecs[:, order]

    # Fix PCA sign ambiguity so axis directions remain deterministic.
    for axis_idx in range(components.shape[1]):
        max_abs_idx = int(np.argmax(np.abs(components[:, axis_idx])))
        if components[max_abs_idx, axis_idx] < 0:
            components[:, axis_idx] *= -1.0

    projected_flat = np.full_like(flat, np.nan)
    projected_flat[finite_mask] = (valid - mean) @ components

    eig_sum = float(np.sum(eigvals))
    if eig_sum > 0:
        contrib = eigvals[:2] / eig_sum
    else:
        contrib = np.array([np.nan, np.nan], dtype=np.float64)
    return projected_flat.reshape(positions_np.shape), contrib


def _compute_single_mds_payload(
    matrices: List[np.ndarray],
    align: bool,
    random_state: int,
) -> Dict[str, Any]:
    positions: List[np.ndarray] = []
    contributions: List[np.ndarray] = []

    for matrix in matrices:
        eigvals = _compute_mds_eigvals(matrix)
        eig_sum = float(np.sum(eigvals)) if eigvals.size else 0.0
        contributions.append(eigvals[:2] / eig_sum if eig_sum > 0 else np.array([np.nan, np.nan], dtype=np.float64))

        init = positions[-1] if positions else None
        positions.append(_mds_with_init(matrix, init=init, n_components=2, random_state=random_state))

    positions_raw = np.stack(positions, axis=0)
    positions_aligned = positions_raw if not align else _align_fgw_positions(positions_raw)
    positions_pc, contrib_pc = _project_positions_to_global_pca(positions_aligned)

    return {
        "positions": positions_aligned,
        "positions_raw": positions_raw,
        "contrib": np.stack(contributions, axis=0),
        "positions_pc": positions_pc,
        "contrib_pc": contrib_pc,
        "aligned": np.asarray(1 if align else 0, dtype=np.int64),
        "mds_seed": np.asarray(random_state, dtype=np.int64),
    }


def _compute_wasserstein_mds_payload(
    matrices_00: List[np.ndarray],
    align: bool,
    random_state: int,
) -> Dict[str, Any]:
    payload = _compute_single_mds_payload(matrices_00, align=align, random_state=random_state)
    payload["distance_type"] = np.asarray("wasserstein", dtype=object)
    return payload


def _compute_gromov_wasserstein_mds_payload(
    matrices_10: List[np.ndarray],
    align: bool,
    random_state: int,
) -> Dict[str, Any]:
    payload = _compute_single_mds_payload(matrices_10, align=align, random_state=random_state)
    payload["distance_type"] = np.asarray("gromov_wasserstein", dtype=object)
    return payload


def _get_fgw_distance_inputs(context: GlobalRuntimeContext, runtime: Optional[Dict[str, Any]] = None):
    cache_key = "fgw_distance_inputs"
    cached = _cache_get(runtime, cache_key)
    if cached is not None:
        return cached
    return _cache_set(runtime, cache_key, _load_fgw_distance_matrices(context))


def _get_wasserstein_mds_payload(context: GlobalRuntimeContext, runtime: Optional[Dict[str, Any]] = None):
    cache_key = f"wasserstein_mds_align_{0 if context.config.fgw_no_align else 1}_seed_{context.config.mds_seed}"
    cached = _cache_get(runtime, cache_key)
    if cached is not None:
        return cached

    steps, matrices_00, _ = _get_fgw_distance_inputs(context, runtime)
    payload = _compute_wasserstein_mds_payload(
        matrices_00=matrices_00,
        align=not context.config.fgw_no_align,
        random_state=context.config.mds_seed,
    )
    payload["steps"] = np.asarray(steps, dtype=np.int64)
    return _cache_set(runtime, cache_key, payload)


def _get_gromov_wasserstein_mds_payload(context: GlobalRuntimeContext, runtime: Optional[Dict[str, Any]] = None):
    cache_key = f"gromov_wasserstein_mds_align_{0 if context.config.fgw_no_align else 1}_seed_{context.config.mds_seed}"
    cached = _cache_get(runtime, cache_key)
    if cached is not None:
        return cached

    steps, _, matrices_10 = _get_fgw_distance_inputs(context, runtime)
    payload = _compute_gromov_wasserstein_mds_payload(
        matrices_10=matrices_10,
        align=not context.config.fgw_no_align,
        random_state=context.config.mds_seed,
    )
    payload["steps"] = np.asarray(steps, dtype=np.int64)
    return _cache_set(runtime, cache_key, payload)


def metric_wasserstein_mds(context: GlobalRuntimeContext, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _get_wasserstein_mds_payload(context, runtime)


def metric_gromov_wasserstein_mds(context: GlobalRuntimeContext, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _get_gromov_wasserstein_mds_payload(context, runtime)


def metric_latent_scatter_pca_alignment(
    context: GlobalRuntimeContext, runtime: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Compute sequential PCA sign-flip table for latent_scatter_from_reference frames.

    SVD has sign ambiguity: each principal component can flip independently between
    steps.  This metric loads the stored pca_components for every step, then walks
    through them in order and flips a component's sign whenever its dot product with
    the previous step's (already-oriented) component is negative.  The resulting
    sign table (shape T×2) is stored as a global metric so the render pipeline can
    apply it without reloading all steps per frame.
    """
    metric_dir = context.paths.step_metrics_dir / "latent_scatter_from_reference"
    step_files = sorted(metric_dir.glob("step_*.npz"), key=parse_step)
    if not step_files:
        raise FileNotFoundError(f"No latent_scatter_from_reference step metrics found in {metric_dir}.")

    steps: List[int] = []
    components_list: List[Optional[np.ndarray]] = []

    for path in step_files:
        step = parse_step(path)
        payload = context.store.load_step_metric("latent_scatter_from_reference", step)
        data = _extract_metric_data(payload)
        comp = data.get("pca_components")
        # pca_components is None when latent_dim==2 (no PCA applied)
        components_list.append(comp if (comp is not None and isinstance(comp, np.ndarray) and comp.ndim == 2) else None)
        steps.append(step)

    if not steps:
        return None

    n = len(steps)
    signs = np.ones((n, 2), dtype=np.float64)
    prev_comp: Optional[np.ndarray] = None
    prev_signs: Optional[np.ndarray] = None

    for i, comp in enumerate(components_list):
        if comp is None:
            # No PCA for this step — reset continuity chain
            prev_comp = None
            prev_signs = None
            continue

        if prev_comp is not None and prev_signs is not None:
            prev_oriented = prev_comp * prev_signs[:, np.newaxis]  # (2, D)
            dots = np.sum(comp * prev_oriented, axis=1)            # (2,)
            signs[i] = np.where(dots >= 0, 1.0, -1.0)
        # else: signs[i] stays 1.0 (first valid frame, no flip)

        prev_comp = comp
        prev_signs = signs[i]

    return {
        "steps": np.asarray(steps, dtype=np.int64),
        "signs": signs,
    }


DEFAULT_GLOBAL_METRICS: Dict[str, Callable[[GlobalRuntimeContext, Optional[Dict[str, Any]]], Dict[str, Any]]] = {
    "wasserstein_mds": metric_wasserstein_mds,
    "gromov_wasserstein_mds": metric_gromov_wasserstein_mds,
    "latent_scatter_pca_alignment": metric_latent_scatter_pca_alignment,
}
