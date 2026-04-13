# Analysis Pipeline Architecture Details

> Based on the implementation as of 2026-04-10

---

## Table of Contents

1. [Overall Structure](#1-overall-structure)
2. [Common Foundation — `core/`](#2-common-foundation--core)
3. [Layer 1 — Data Extraction (`data_extraction/`)](#3-layer-1--data-extraction-data_extraction)
4. [Layer 2 — Per-Step Metrics (`compute_analysis/step/`)](#4-layer-2--per-step-metrics-compute_analysisstep)
5. [Layer 3 — Global Metrics (`compute_analysis/global_metrics/`)](#5-layer-3--global-metrics-compute_analysisglobal_metrics)
6. [Layer 4 — Figure Generation (`visualize/`)](#6-layer-4--figure-generation-visualize)
7. [Layer 5 — Video Generation (`video/`)](#7-layer-5--video-generation-video)
8. [CLI Entry Point (`cli.py`)](#8-cli-entry-point-clipy)
9. [Artifact Output Location Reference](#9-artifact-output-location-reference)
10. [Extension Guidelines](#10-extension-guidelines)

---

## 1. Overall Structure

```
analysis/
├── cli.py                          # CLI entry point (integrates all layers)
├── core/                           # Common foundation (config types, store, registry, etc.)
│   ├── types.py                    # Config dataclasses · AnalysisPaths
│   ├── store.py                    # ArtifactStore (abstraction for file I/O)
│   ├── registry.py                 # ExtractorRegistry / ExtractionRuntimeContext
│   ├── eval_context.py             # EvalContext (data accessor with per-step cache)
│   ├── fgw.py                      # FGW distance matrix computation utilities
│   └── rsa.py                      # RSA (Representational Similarity Analysis) utilities
├── data_extraction/                # Layer 1: extraction of simulation output
│   ├── pipeline.py                 # StepExtractionPipeline (parallel extraction)
│   ├── step_extractors.py          # Extractor functions · build_default_registry()
│   └── rebuild_society.py          # Society reconstruction utilities
├── compute_analysis/               # Layers 2 & 3: metric computation
│   ├── step/
│   │   ├── metrics.py              # DEFAULT_STEP_METRICS (10 entries)
│   │   └── pipeline.py             # StepMetricPipeline (parallel computation)
│   └── global_metrics/
│       ├── metrics.py              # DEFAULT_GLOBAL_METRICS (3 entries)
│       └── pipeline.py             # GlobalMetricPipeline (sequential computation)
├── visualize/                      # Layer 4: figure generation
│   ├── pipeline.py                 # FigurePipeline · FigureBatchRunner · FigureRegistry
│   ├── figures_eval.py             # Renderers for per-step evaluation figures
│   ├── figures_fgw.py              # Renderers for FGW-MDS trajectory figures
│   ├── figures_rsa.py              # Renderers for RSA time-series figures
│   ├── figures_summary.py          # Renderers for summary panel figures
│   └── plotting.py                 # Common plotting helper functions
└── video/                          # Layer 5: video generation
    ├── pipeline.py                 # VideoPipeline · VideoBatchRunner
    └── manager.py                  # VideoFrameManager (MP4 writing)
```

### Full Pipeline Data Flow

```
Simulation output
  data/step_{N}.pt          ← per-step event data
  checkpoints/step_{N}/     ← per-step model checkpoints
  .hydra/config.yaml        ← Hydra config

          ↓  [Layer 1: extract]
analysis_artifacts/
  extracted/step_{N}.pt         ← integrated payload containing all agent info
  shared/{id}/step_{N}.npz      ← intermediate arrays directly referenced by Layer 2

          ↓  [Layer 2: compute-step]
analysis_artifacts/
  step_metrics/{metric_id}/step_{N}.npz

          ↓  [Layer 3: compute-global]
analysis_artifacts/
  global_metrics/{metric_id}.npz

          ↓  [Layer 4: render / render-frames]
analysis_figures/
  {figure_id}/step_{N}.png/.pdf/.svg   ← stepwise category
  {figure_id}/figure.png/.pdf/.svg     ← whole_step category

          ↓  [Layer 5: render-video]
analysis_videos/
  {figure_id}.mp4
```

---

## 2. Common Foundation — `core/`

### `core/types.py` — Config Dataclasses

| Class | Role |
|--------|------|
| `AnalysisPaths` | Immutable dataclass aggregating all directory paths. Created via `from_input_dir(input_dir)` |
| `ExtractRunConfig` | Layer 1 run config (device, steps, num_workers, refresh) |
| `StepMetricRunConfig` | Layer 2 run config (metrics, steps, num_workers, refresh) |
| `GlobalMetricRunConfig` | Layer 3 run config (metrics, refresh, num_workers, fgw_no_align, mds_seed) |
| `RenderRunConfig` | Layer 4 run config (figure_id, step, steps, refresh) |
| `VideoRunConfig` | Layer 5 run config (figure_ids, fps, steps, num_workers, refresh) |
| `StepExtractContext` | Per-step context during extraction (step number, checkpoint_dir, data_path) |

### `core/store.py` — `ArtifactStore`

All layers perform file I/O through this class. It centralizes path resolution and serialization.

| Method | Format | Description |
|----------|----------|------|
| `save_extracted_step(step, payload)` | `.pt` (torch.save) | Layer 1 extraction result |
| `load_extracted_step(step)` | `.pt` (torch.load) | Loaded by Layer 2 |
| `save_shared_step(shared_id, step, payload)` | `.npz` (np.savez) | Intermediate arrays during extraction (directly referenced by other layers) |
| `load_shared_step(shared_id, step)` | `.npz` | |
| `save_step_metric(metric_id, step, payload)` | `.npz` | Layer 2 computation result |
| `load_step_metric(metric_id, step)` | `.npz` | Loaded by Layer 4 |
| `save_global_metric(metric_id, payload)` | `.npz` | Layer 3 computation result |
| `load_global_metric(metric_id)` | `.npz` | Loaded by Layer 4 |
| `has_*` methods | — | Existence checks (used for refresh decisions) |

`_encode_npz_value` / `_decode_npz_value` transparently handles conversions between numpy/torch/scalar. Scalars are saved as shape=() ndarrays and restored to Python scalars on load.

### `core/registry.py` — Extractor Registration Mechanism

```python
class ExtractionRuntimeContext:
    step_context: StepExtractContext   # step number · file paths
    cfg: Any                           # Hydra config
    society: Society                   # reconstructed Society
    data_payload: Dict[str, Any]       # raw data from data/step_N.pt
    eval_context: EvalContext          # cached accessor
    store: ArtifactStore               # storage operations
```

```python
class ExtractorRegistry:
    def register(extractor_id, fn) -> None           # register (duplicates not allowed)
    def list_ids() -> List[str]                      # returns in registration order
    def run_all(context, payload) -> Dict            # applies all extractors sequentially
```

### `core/eval_context.py` — `EvalContext`

Accessor that retrieves various data from Society during extraction. Each method saves results to `_cache` to prevent recomputation.

| Method | Return type | Description |
|----------|--------|------|
| `get_observations()` | `List[np.ndarray]` | All buffer observations per agent |
| `get_creations()` | `List[np.ndarray]` | Generated observations per agent |
| `get_adjacency_matrix(include_self)` | `np.ndarray` | Adjacency matrix from NetworkX graph |
| `get_agent_to_cluster()` | `(List[int], int)` | Greedy modularity clustering |
| `get_reference_observations()` | `torch.Tensor` | Flattened reference observations |
| `get_reference_observations_grouped()` | `torch.Tensor [A, S, D]` | Agent × sample × dimension |
| `get_latent_stats(reference_obs)` | `(mu, std)` | Latent variable statistics for all agents |

---

## 3. Layer 1 — Data Extraction (`data_extraction/`)

### Role

Reads simulation output (checkpoints + event data) and saves a rich per-step payload to `analysis_artifacts/extracted/step_{N}.pt`.

### `rebuild_society.py` — Society Reconstruction

| Function | Description |
|------|------|
| `parse_step(path)` | Extracts step number from file path (`step_N.pt` → `N`) |
| `load_config(input_dir)` | Loads `.hydra/config.yaml` with OmegaConf |
| `build_society(cfg, device)` | Builds the agent list and Society from cfg and places initial data via DataFactory |
| `load_checkpoint(society, checkpoint_dir, num_agents)` | Loads `agent_{i}.pt` into each agent |

### `step_extractors.py` — Extractor Functions

Each extractor has the signature `(context: ExtractionRuntimeContext, payload: Dict) -> Dict` and returns the extended `payload`.

| extractor_id | Function name | Keys added | Saved as shared |
|---|---|---|---|
| `observations_creations` | `extract_observation_creation_views` | `observations`, `creations` | none |
| `network` | `extract_network_views` | `adjacency_matrix`, `adjacency_matrix_with_self`, `agent_to_cluster`, `num_clusters`, `network` | `adjacency_and_clusters` |
| `reference_obs` | `extract_reference_views` | `reference_obs` | `reference_obs` |
| `reference_latent_stats` | `extract_latent_stats` | `reference_latent_mu`, `reference_latent_std`, `reference_latent_stats`, `reference_latent_aligned_with_reference_obs` | `reference_latent_stats` |
| `events` | `extract_event_payloads` | `mhng_results`, `memorize_results`, `events` | none |

`build_default_registry()` registers these 5 in order.

**Note**: Extractors are executed in registration order. `reference_latent_stats` depends on `reference_obs`, so it must be registered after `reference_obs`.

### `pipeline.py` — `StepExtractionPipeline`

```
StepExtractionPipeline.__init__(config: ExtractRunConfig)
├── paths = AnalysisPaths.from_input_dir(config.input_dir)
├── store = ArtifactStore(paths)
└── registry = build_default_registry()

StepExtractionPipeline.run()
├── 1. ensure_layout()           — create directories
├── 2. _resolve_cfg()            — load config.yaml
├── 3. _resolve_steps()          — scan checkpoints/step_*/ directories
└── 4. process each step with _extract_single_step()
      ├── rebuild Society with build_society()
      ├── load model with load_checkpoint()
      ├── create EvalContext
      ├── create ExtractionRuntimeContext
      ├── run all extractors sequentially with registry.run_all()
      └── save with store.save_extracted_step()
```

**Parallelism**: When `num_workers >= 2`, each step is processed in parallel using `ProcessPoolExecutor`. The worker function `_extract_single_step_worker` creates its own `StepExtractionPipeline` instance, so there is no shared state. Always completes in a single pass of `run_all()`.

---

## 4. Layer 2 — Per-Step Metrics (`compute_analysis/step/`)

### Role

Reads Layer 1 extraction results (`extracted/step_{N}.pt`) and computes independent per-step numerical values and matrices, saving them to `step_metrics/{metric_id}/step_{N}.npz`.

### `metrics.py` — Step Metric Functions

All function signatures: `(step_payload: Dict, runtime: Optional[Dict]) -> Optional[Dict]`

Computation results within a step are saved to `runtime["cache"]`, allowing multiple metrics to share intermediate computations (e.g., `wasserstein_similarity` and `network_vs_wasserstein_similarity` use the same distance matrix).

| metric_id | Function | Description | Required payload keys |
|---|---|---|---|
| `observations_and_creations_clustered` | `metric_observations_and_creations_clustered` | Observation/creation data + cluster info | `observations`, `creations`, `adjacency_matrix`, `agent_to_cluster`, `num_clusters` |
| `network_vs_similarity_latent` | `metric_network_vs_similarity_latent` | Pearson correlation between inter-agent latent variable distance and network adjacency | `reference_latent_mu`, `reference_latent_std`, `adjacency_matrix_with_self` |
| `wasserstein_similarity` | `metric_wasserstein_similarity` | FGW distance matrix (α=0.0, Wasserstein) | `reference_latent_mu`, `reference_latent_std` |
| `gromov_wasserstein_similarity` | `metric_gromov_wasserstein_similarity` | FGW distance matrix (α=1.0, GW) | same as above |
| `network_vs_wasserstein_similarity` | `metric_network_vs_wasserstein_similarity` | Correlation between Wasserstein distance and network adjacency | wasserstein_similarity cache |
| `network_vs_gromov_wasserstein_similarity` | `metric_network_vs_gromov_wasserstein_similarity` | Correlation between GW distance and network adjacency | gromov_wasserstein_similarity cache |
| `latent_scatter_from_reference` | `metric_latent_scatter_from_reference` | 2D projection of latent variables (PCA or direct). Also saves `pca_components` (2×D) for aligning PCA axis directions across video frames | `reference_latent_mu`, `reference_obs` |
| `rsa_within_clusters` | `metric_rsa_within_clusters` | Within-cluster RSA (Representational Similarity Analysis) + neighbor latent RSA | `reference_obs`, `reference_latent_mu`, `agent_to_cluster` |
| `mhng_edge_acceptance` | `metric_mhng_edge_acceptance` | Per-edge MHNG acceptance rate matrix | `mhng_results`, `num_agents` |
| `memorize_edge_acceptance` | `metric_memorize_edge_acceptance` | Per-edge memorize acceptance rate matrix | `memorize_results`, `num_agents` |

### `pipeline.py` — `StepMetricPipeline`

```
StepMetricPipeline.run()
├── 1. _resolve_metric_ids()   — determine target metric_ids (all if not specified)
├── 2. _resolve_steps()        — scan step_*.pt in extracted/
└── 3. process each step
      ├── store.load_extracted_step(step)  — load payload
      ├── runtime = {"cache": {}, "step": step}
      └── for each metric_id
            ├── skip check with has_step_metric()
            ├── execute metric_fn(step_payload, runtime)
            └── save with store.save_step_metric()
```

**Parallelism**: When `num_workers >= 2`, steps are processed in parallel using `ProcessPoolExecutor`. The worker function `_compute_single_step_metrics_worker` processes all metric_ids before returning (serial within a step).

---

## 5. Layer 3 — Global Metrics (`compute_analysis/global_metrics/`)

### Role

Computes MDS trajectories of FGW distance matrices spanning all steps and saves them to `global_metrics/{metric_id}.npz`.

### Key Data Structures

**`GlobalRuntimeContext`** (dataclass)
```python
@dataclass
class GlobalRuntimeContext:
    config: GlobalMetricRunConfig
    paths: AnalysisPaths
    store: ArtifactStore
```

MDS results are cached in `runtime["cache"]` for reuse by subsequent metrics.

### `metrics.py` — Global Metric Functions

All function signatures: `(context: GlobalRuntimeContext, runtime: Optional[Dict]) -> Dict`

| metric_id | Function | Description | Required artifacts |
|---|---|---|---|
| `wasserstein_mds` | `metric_wasserstein_mds` | MDS trajectory of Wasserstein distance matrices (with Procrustes alignment) | step_metrics: `wasserstein_similarity` |
| `gromov_wasserstein_mds` | `metric_gromov_wasserstein_mds` | MDS trajectory of GW distance matrices | step_metrics: `gromov_wasserstein_similarity` |
| `latent_scatter_pca_alignment` | `metric_latent_scatter_pca_alignment` | PCA sign flip table between latent scatter video frames (shape T×2). Resolves sign indeterminacy from SVD based on continuity | step_metrics: `latent_scatter_from_reference` (requires `pca_components` key) |

#### Key Internal Processing

**MDS computation flow** (`_compute_single_mds_payload`):
1. Run classical MDS on each step's distance matrix (`_classical_mds`)
2. Further optimize with sklearn MDS (using the previous step's positions as initialization)
3. Apply Procrustes alignment between steps (`_align_fgw_positions`)
4. Project onto global PCA across all steps (`_project_positions_to_global_pca`)

### `pipeline.py` — `GlobalMetricPipeline`

```
GlobalMetricPipeline.run()
├── 1. _resolve_metric_ids()   — determine target metric_ids
├── 2. create GlobalRuntimeContext
├── 3. runtime = {"cache": {}}   ← shared cache across all metrics
└── 4. process each metric_id sequentially
      ├── skip check with has_global_metric()
      ├── execute metric_fn(context, runtime)
      └── save with store.save_global_metric()
```

**Note**: Unlike Layer 2, global metric computation is sequential (results are passed to the next metric via `runtime["cache"]`).

---

## 6. Layer 4 — Figure Generation (`visualize/`)

### Role

Reads Layer 2/3 metric results, generates figures with Matplotlib, and saves them as PNG/PDF/SVG under `analysis_figures/{figure_id}/`.

### Key Classes

#### `FigureSpec` (frozen dataclass)

Metadata object defining a single figure.

```python
@dataclass(frozen=True)
class FigureSpec:
    figure_id: str                     # unique ID
    category: str                      # "stepwise", "whole_step" or "snapshot"
    requires_step: bool                # whether --step argument is required
    required_step_metrics: List[str]   # required step_metric IDs
    required_global_metrics: List[str] # required global_metric IDs
    loader: Callable[[FigurePipeline, Optional[int], Optional[List[int]]], Dict]
    renderer: Callable[[Dict], matplotlib.figure.Figure]
    frame_steps_resolver: Optional[Callable[[FigurePipeline, Optional[List[int]]], List[int]]]
```

- **`loader`**: Function that reads data from storage and returns a payload
- **`renderer`**: Function that takes a payload and returns a `matplotlib.figure.Figure`

#### `FigureRegistry`

```python
class FigureRegistry:
    def register(spec: FigureSpec) -> None   # register (duplicates not allowed)
    def get(figure_id: str) -> FigureSpec
    def list_ids() -> List[str]              # returns sorted list
```

#### `FigurePipeline`

All `FigureSpec`s are registered in `_build_registry()`.

```
FigurePipeline.render()
├── 1. spec = registry.get(figure_id)
├── 2. _validate_dependencies()    — check existence of required artifacts
├── 3. if base_path .png exists and refresh=False → return (skip)
├── 4. payload = spec.loader(self, step, steps)
├── 5. fig = spec.renderer(payload)
└── 6. _save_figure(fig, base_path)   — write to PNG/PDF/SVG

FigurePipeline.render_frames(steps)
└── calls render() sequentially for each resolved_step

FigurePipeline.resolve_frame_steps(steps)
└── if steps is None, normally returns common steps of required step_metrics; for PCMCI figures returns global metric's `window_center_steps`
```

### Registered Figure List

#### stepwise category (requires step specification, can be made into video)

| figure_id | Required step_metrics | renderer |
|---|---|---|
| `observations_and_creations_clustered` | `observations_and_creations_clustered` | `figures_eval.render_observations_and_creations_clustered` |
| `observations_and_creations_clustered_legend` | `observations_and_creations_clustered` | `figures_eval.render_observations_and_creations_clustered_legend` |
| `observations_and_creations_agents_clustered` | `observations_and_creations_clustered` | `figures_eval.render_observations_and_creations_agents_clustered` |
| `observations_and_creations_agents_clustered_legend` | `observations_and_creations_clustered` | `figures_eval.render_observations_and_creations_agents_clustered_legend` |
| `observations_agents_clustered` | `observations_and_creations_clustered` | `figures_eval.render_observations_agents_clustered` |
| `wasserstein_similarity` | `wasserstein_similarity`, `gromov_wasserstein_similarity` | `figures_eval.render_wasserstein_similarity` |
| `gromov_wasserstein_similarity` | `wasserstein_similarity`, `gromov_wasserstein_similarity` | `figures_eval.render_gromov_wasserstein_similarity` |
| `latent_scatter_from_reference` | `latent_scatter_from_reference` (step), `latent_scatter_pca_alignment` (global) | `figures_eval.render_latent_scatter_from_reference` |
| `social_network` | `observations_and_creations_clustered` | `figures_eval.render_social_network` |
| `mhng_acceptance_matrix` | `mhng_edge_acceptance` | `figures_eval.render_mhng_acceptance_matrix` |
| `mhng_acceptance_network` | `mhng_edge_acceptance`, `observations_and_creations_clustered` | `figures_eval.render_mhng_acceptance_network` |
| `memorize_acceptance_matrix` | `memorize_edge_acceptance` | `figures_eval.render_memorize_acceptance_matrix` |
| `memorize_acceptance_network` | `memorize_edge_acceptance`, `observations_and_creations_clustered` | `figures_eval.render_memorize_acceptance_network` |
| `both_acceptance_network` | `mhng_edge_acceptance`, `memorize_edge_acceptance`, `observations_and_creations_clustered` | `figures_eval.render_both_acceptance_network` |

#### whole_step category (renders all steps at once)

| figure_id | Dependencies | renderer |
|---|---|---|
| `wasserstein_mds_trajectory` | global: `wasserstein_mds` | `figures_fgw.render_mds_trajectory` |
| `gromov_wasserstein_mds_trajectory` | global: `gromov_wasserstein_mds` | `figures_fgw.render_mds_trajectory` |
| `wasserstein_mds_trajectory_cluster` | step: `observations_and_creations_clustered`, global: `wasserstein_mds` | `figures_fgw.render_mds_trajectory_clustered` |
| `gromov_wasserstein_mds_trajectory_cluster` | step: `observations_and_creations_clustered`, global: `gromov_wasserstein_mds` | `figures_fgw.render_mds_trajectory_clustered` |
| `wasserstein_mds_pc_trajectory` | global: `wasserstein_mds` | `figures_fgw.render_mds_pc_trajectory` |
| `gromov_wasserstein_mds_pc_trajectory` | global: `gromov_wasserstein_mds` | `figures_fgw.render_mds_pc_trajectory` |
| `wasserstein_mds_pc_trajectory_cluster` | step: `observations_and_creations_clustered`, global: `wasserstein_mds` | `figures_fgw.render_mds_pc_trajectory_clustered` |
| `gromov_wasserstein_mds_pc_trajectory_cluster` | step: `observations_and_creations_clustered`, global: `gromov_wasserstein_mds` | `figures_fgw.render_mds_pc_trajectory_clustered` |
| `rsa_clusters` | step: `rsa_within_clusters` | `figures_rsa.render_rsa_clusters` |
| `rsa_agent_{agent_idx}` | step: `rsa_within_clusters` | `figures_rsa.render_rsa_agent` |

`rsa_agent_{agent_idx}` is dynamically registered at startup based on detected data shapes.

#### snapshot category (combines multiple steps into one image)

| figure_id | Dependencies | renderer |
|---|---|---|
| `observations_agents_clustered_panel` | step: `observations_and_creations_clustered` | `figures_summary.render_observations_agents_clustered_panel` |
| `wasserstein_similarity_snapshot` | step: `wasserstein_similarity` | `figures_summary.render_wasserstein_similarity_snapshot` |
| `gromov_wasserstein_similarity_snapshot` | step: `gromov_wasserstein_similarity` | `figures_summary.render_gromov_wasserstein_similarity_snapshot` |
| `wasserstein_similarity_average_snapshot` | step: `wasserstein_similarity` | `figures_summary.render_wasserstein_similarity_average_snapshot` |
| `gromov_wasserstein_similarity_average_snapshot` | step: `gromov_wasserstein_similarity` | `figures_summary.render_gromov_wasserstein_similarity_average_snapshot` |
| `both_acceptance_network_snapshot` | step: `mhng_edge_acceptance`, `memorize_edge_acceptance`, `observations_and_creations_clustered` | `figures_summary.render_both_acceptance_network_snapshot` |
| `both_acceptance_average_network_snapshot` | step: `mhng_edge_acceptance`, `memorize_edge_acceptance`, `observations_and_creations_clustered` | `figures_summary.render_both_acceptance_average_network_snapshot` |
| `gromov_wasserstein_mds_pc_trajectory_cluster_segmented` | step: `observations_and_creations_clustered`, global: `gromov_wasserstein_mds` | `figures_fgw.render_mds_pc_trajectory_clustered_segmented` |

### `FigureBatchRunner`

Utility for parallel rendering of multiple figures. Uses a process pool in the `spawn` context.

```python
FigureBatchRunner.render_many(base_config, figure_ids, num_workers)
    → (rendered: Dict[str, Path], missing_by_figure: Dict, error_by_figure: Dict)

FigureBatchRunner.render_frames_many(input_dir, figure_ids, steps, refresh, num_workers)
    → (rendered_counts: Dict[str, int], missing_by_figure: Dict, error_by_figure: Dict)
```

### Multi-seed RSA Rendering via `MultiSeedFigurePipeline`

A class that aligns `rsa_within_clusters` step metrics from multiple seeds on common steps, computes mean and standard deviation along the seed axis, and outputs whole-step figures. Implemented as a separate class from `FigurePipeline` in `analysis/visualize/pipeline.py`.

Config type `RenderSeedsRunConfig`:

```python
@dataclass(frozen=True)
class RenderSeedsRunConfig:
    input_dirs: list[Path]
    output_dir: Path
    figure_ids: list[str]
    refresh: bool = False
    num_workers: int = 1
```

Main responsibilities:
- Holds `ArtifactStore(AnalysisPaths.from_input_dir(d))` for each seed
- Enumerates available steps of `rsa_within_clusters` per seed and finds the intersection across seeds
- Stacks `agent_rsa [A]` and `cluster_rsa [C]` per seed to build `agent_rsa_seeds [S, T, A]` and `cluster_rsa_seeds [S, T, C]`
- Computes mean and standard deviation using `np.nanmean` / `np.nanstd`
- Saves output to `output_dir/analysis_figures/{figure_id}/figure.{png,pdf,svg}`

Registered figures (only registered when `num_agents` and `num_clusters` match across all seeds):

| figure_id | Description |
|---|---|
| `rsa_clusters_seeds` | Cluster-level RSA time series across multiple seeds (mean ±1σ band) |
| `rsa_agent_{agent_idx}_seeds` | Per-agent RSA time series across multiple seeds (dynamically registered) |

Parallelism: When `num_workers >= 2`, uses `multiprocessing.Pool` in the `spawn` context.

---

## 7. Layer 5 — Video Generation (`video/`)

### Role

Concatenates frame images from stepwise figures generated in Layer 4 and saves them as MP4 videos to `analysis_videos/{figure_id}.mp4`.

### `manager.py` — `VideoFrameManager`

```python
class VideoFrameManager:
    def __init__(fps: int = 12)
    def write_mp4(frame_paths: list[Path], output_path: Path, progress_callback=None) -> Path
```

- `_resolve_target_size()`: Computes the maximum size across all frames
- `_pad_to_shape()`: Center-pads frames of different sizes
- `_to_rgb_uint8()`: Converts grayscale, RGBA, float, etc. to RGB uint8

Writes MP4 using the `imageio` library.

### `pipeline.py` — `VideoPipeline` · `VideoBatchRunner`

```
VideoPipeline.render_video(figure_id, steps) -> Path
├── _resolve_frame_paths() identifies frame files
├── create VideoFrameManager(fps)
└── manager.write_mp4() writes the video

VideoBatchRunner.render_many(base_config, figure_ids)
    → (rendered, skipped, missing_by_figure, error_by_figure)
```

**Parallelism**: Renders multiple videos in parallel using `Pool.imap_unordered` in the `spawn` context. Each task prints progress every 5% via `print()`.

---

## 8. CLI Entry Point (`cli.py`)

| Subcommand | Corresponding class | Description |
|---|---|---|
| `extract` | `StepExtractionPipeline` | Layer 1: extract from checkpoints |
| `compute-step` | `StepMetricPipeline` | Layer 2: per-step metric computation |
| `compute-global` | `GlobalMetricPipeline` | Layer 3: global metric computation |
| `render` | `FigurePipeline` / `FigureBatchRunner.render_many()` | Layer 4: whole_step figure rendering |
| `render-frames` | `FigurePipeline.render_frames()` / `FigureBatchRunner.render_frames_many()` | Layer 4: full frame generation for stepwise figures |
| `render-snapshot` | `FigurePipeline` / `FigureBatchRunner.render_many()` | Layer 4: snapshot figure rendering (`--steps` required) |
| `render-seeds` | `MultiSeedFigurePipeline` | Layer 4: multi-seed aggregated RSA figure generation |
| `render-conditions` | — | Layer 4: multi-condition comparison RSA figure generation |
| `render-video` | `VideoPipeline` / `VideoBatchRunner.render_many()` | Layer 5: video generation |
| `status` | — | Display list of available figure IDs |

---

## 9. Artifact Output Location Reference

```
{input_dir}/
├── analysis_artifacts/
│   ├── extracted/
│   │   └── step_{N}.pt                         # Layer 1 output (torch format)
│   ├── shared/
│   │   ├── adjacency_and_clusters/step_{N}.npz
│   │   ├── reference_obs/step_{N}.npz
│   │   └── reference_latent_stats/step_{N}.npz
│   ├── step_metrics/
│   │   └── {metric_id}/step_{N}.npz            # Layer 2 output
│   └── global_metrics/
│       └── {metric_id}.npz                     # Layer 3 output
├── analysis_figures/
│   └── {figure_id}/
│       ├── step_{N}.png/.pdf/.svg              # stepwise figures
│       └── figure.png/.pdf/.svg                # whole_step figures
└── analysis_videos/
    └── {figure_id}.mp4                         # Layer 5 output
```

---

## 10. Extension Guidelines

### 10.1 Adding a New Extractor (Layer 1 Extension)

**Location**: `analysis/data_extraction/step_extractors.py`

#### Steps

1. **Define the extraction function**

```python
def extract_my_new_data(
    context: ExtractionRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    # Access Society data via context.eval_context
    # Optionally save intermediate data directly referenceable by other layers via context.store.save_shared_step()
    # Reference raw event data from context.data_payload
    my_data = context.eval_context.something()
    payload["my_key"] = my_data
    return payload
```

2. **Register in `build_default_registry()`**

```python
def build_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    # ... existing registrations ...
    registry.register("my_new_extractor", extract_my_new_data)
    return registry
```

#### Notes

- `payload` carries over the output of the previous extractor. Consider registration order if dependent keys are required.
- If data not in `EvalContext` is needed, add a method to `EvalContext` or access `context.society` or `context.data_payload` directly.
- Re-extraction with the `--refresh` flag is required after adding.

---

### 10.2 Adding a New Step Metric (Layer 2 Extension)

**Location**: `analysis/compute_analysis/step/metrics.py`

#### Steps

1. **Define the computation function**

```python
def metric_my_step_metric(
    step_payload: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    # Share intermediate results using runtime["cache"]
    cached = _cache_get(runtime, "my_step_metric")
    if cached is not None:
        return cached

    # Retrieve Layer 1 extraction data from step_payload
    mu = _to_numpy(step_payload.get("reference_latent_mu"))
    if mu is None:
        return None

    result = {"my_value": compute_something(mu)}
    return _cache_set(runtime, "my_step_metric", result)
```

2. **Add to `DEFAULT_STEP_METRICS`**

```python
DEFAULT_STEP_METRICS = {
    # ... existing ...
    "my_step_metric": metric_my_step_metric,
}
```

#### Notes

- Keys in `step_payload` are Layer 1 extracted payload keys (see `SCHEMA_REFERENCE.md`).
- Returning `None` skips saving (use when data is absent).
- `runtime["cache"]` is shared across all metrics for the same step. Use unique key names.
- Expensive computations (FGW distance matrices, etc.) must be cached.
- After adding, just re-run Layer 2 (no need to re-run Layer 1).

---

### 10.3 Adding a New Global Metric (Layer 3 Extension)

**Location**: `analysis/compute_analysis/global_metrics/metrics.py`

#### Steps

1. **Define the computation function**

```python
def metric_my_global_metric(
    context: GlobalRuntimeContext,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Load step_metrics from context.store
    # Reference parameters like window_size, lag from context.config
    # Reference directory paths from context.paths
    # Share data with other global metrics using runtime["cache"]

    # Example: load step_metric for all steps
    metric_dir = context.paths.step_metrics_dir / "some_metric"
    files = sorted(metric_dir.glob("step_*.npz"), key=parse_step)

    data_by_step = []
    for path in files:
        step = parse_step(path)
        payload = context.store.load_step_metric("some_metric", step)
        data_by_step.append(payload)

    result = {"aggregated": compute_global_thing(data_by_step)}
    return result
```

2. **Add to `DEFAULT_GLOBAL_METRICS`**

```python
DEFAULT_GLOBAL_METRICS = {
    # ... existing ...
    "my_global_metric": metric_my_global_metric,
}
```

#### Notes

- Global metrics are **executed sequentially**. Data accumulated in `runtime["cache"]` from previous metrics can be used.
- If depending on another global metric's result (e.g., `wasserstein_mds`), call the corresponding `_get_*` helper.
- Current fields of `GlobalMetricRunConfig` are `metrics`, `refresh`, `num_workers`, `fgw_no_align`, `mds_seed`.

---

### 10.4 Adding a New Figure (Layer 4 Extension)

**Location**: `analysis/visualize/pipeline.py` (registration) and `analysis/visualize/figures_*.py` (renderers)

#### Steps

1. **Define the renderer function in `figures_*.py`**

```python
# Add to analysis/visualize/figures_eval.py (or a new module)

def render_my_new_figure(data: Dict[str, Any], step: Optional[int] = None) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    # Extract values from data and plot
    ax.plot(data["my_value"])
    if step is not None:
        ax.set_title(f"My Figure (step={step})")
    return fig
```

2. **Register a `FigureSpec` in `FigurePipeline._build_registry()`**

```python
# Add inside _build_registry() in pipeline.py

reg.register(
    FigureSpec(
        figure_id="my_new_figure",
        category="stepwise",        # or "whole_step"
        requires_step=True,         # True for stepwise
        required_step_metrics=["my_step_metric"],
        required_global_metrics=[],
        loader=lambda p, step, steps: {
            "step": step,
            "data": p._load_step_metric_data("my_step_metric", step),
        },
        renderer=lambda payload: figures_eval.render_my_new_figure(
            payload["data"], payload["step"]
        ),
    )
)
```

#### Category Selection

| category | Use case | Can be made into video with render-frames | Save path |
|---|---|---|---|
| `stepwise` | Figures that differ per step (observation distributions, etc.) | Yes | `step_{N}.png` |
| `whole_step` | Figures spanning all steps (MDS trajectories, etc.) | No | `figure.png` |
| `snapshot` | Figures combining multiple specified steps into one image | No | `steps_{s1}_{s2}_....png` |

#### Using `steps` in whole_step / snapshot figures

For the `snapshot` category, set `requires_steps=True` and use the `steps` argument in `loader` to load data.

```python
reg.register(
    FigureSpec(
        figure_id="my_snapshot_figure",
        category="snapshot",
        requires_step=False,
        requires_steps=True,
        required_step_metrics=["my_metric"],
        required_global_metrics=[],
        loader=lambda p, step, steps: {
            "panel_items": [(s, p._load_step_metric_data("my_metric", s)) for s in (steps or [])]
        },
        renderer=lambda payload: figures_eval.render_my_snapshot(payload["panel_items"]),
    )
)
```

When a global metric is also needed (as in `gromov_wasserstein_mds_pc_trajectory_cluster_segmented`), combine `p._load_global_metric_data()` in the loader. Each value in `--steps` is treated as an interval boundary, generating one subplot per interval `{s_prev+1}–{s_i}`.

#### Notes

- The first argument `p` of `loader` is the `FigurePipeline` instance itself. Use `p._load_step_metric_data()` or `p._load_global_metric_data()`.
- If creating a new module (e.g., `figures_custom.py`), add it to the imports in `pipeline.py`.
- `renderer` returns a `matplotlib.figure.Figure`. No need to call `plt.close(fig)` — `_save_figure()` handles it.

---

### 10.5 Typical Flow When Combining Multiple Extensions

Example: "I want to visualize per-agent KL divergence as a time series"

```
1. [Layer 1 extension] Add extract_kl_divergence() to step_extractors.py
   → adds payload["kl_per_agent"]

2. [Layer 2 extension] Add metric_kl_divergence() to step/metrics.py
   → saves to step_metrics/kl_divergence/step_{N}.npz

3. [Layer 3 extension] Add metric_kl_divergence_timeseries() to global_metrics/metrics.py
   → loads kl_divergence for all steps and creates time-series array
   → saves to global_metrics/kl_divergence_timeseries.npz

4. [Layer 4 extension] Add render_kl_timeseries() to figures_eval.py
   Register FigureSpec(figure_id="kl_timeseries", category="whole_step", ...) 

5. Execute via CLI:
   uv run analysis/cli.py extract --input-dir ...
   uv run analysis/cli.py compute-step --input-dir ...
   uv run analysis/cli.py compute-global --input-dir ...
   uv run analysis/cli.py render --input-dir ... --figure-id kl_timeseries
```

If the Layer 1 extraction data is sufficient with existing keys, no Layer 1 changes are needed. If an existing Layer 2 step_metric can be reused, no Layer 2 changes are needed either.
