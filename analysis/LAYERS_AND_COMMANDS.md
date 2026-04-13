# Analysis Pipeline Complete Guide

This document summarizes the role, inputs/outputs, artifacts, execution commands, and available options for each layer of the `analysis` pipeline.

## 0. Overview

The 5-layer pipeline proceeds in the following order:

1. `extract` (Layer 1): Converts raw simulation output into extracted format for reuse in downstream layers
2. `compute-step` (Layer 2): Computes per-step metrics
3. `compute-global` (Layer 3): Computes time-series and flow metrics spanning all steps
4. `render` / `render-frames` (Layer 4): Outputs visualization images from Layer 2/3 metrics
5. `render-video` (Layer 5): Generates video from stepwise frame images produced in Layer 4

Main directory structure (`{input_dir}` is the Hydra output directory):

```text
{input_dir}/
├── analysis_artifacts/
│   ├── extracted/        # Layer 1
│   ├── shared/           # Layer 1 (shared auxiliary)
│   ├── step_metrics/     # Layer 2
│   └── global_metrics/   # Layer 3
├── analysis_figures/     # Layer 4
└── analysis_videos/      # Layer 5 video output
```

## 1. Layer 1 `extract`

### Role

- Creates a unified analysis payload per step from `checkpoints/step_*`, `data/step_*.pt`, and `.hydra/config.yaml`.
- Produces a state where downstream `compute-step` can run using only `extracted/step_*.pt`.

### Main Inputs

- `{input_dir}/.hydra/config.yaml`
- `{input_dir}/checkpoints/step_{t}/agent_{i}.pt`
- `{input_dir}/data/step_{t}.pt`

### Main Outputs

- `{input_dir}/analysis_artifacts/extracted/step_{t}.pt`
- `{input_dir}/analysis_artifacts/shared/adjacency_and_clusters/step_{t}.npz`
- `{input_dir}/analysis_artifacts/shared/reference_obs/step_{t}.npz`
- `{input_dir}/analysis_artifacts/shared/reference_latent_stats/step_{t}.npz`

### Execution Command

```bash
uv run python -m analysis.cli extract --input-dir /path/to/hydra/output
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--device` | `str` | `None` | Device override (e.g., `cpu`, `cuda:0`) |
| `--steps` | `str` | `all` | Target steps. `"1,100,500"` or `"all"` |
| `--refresh` | flag | `False` | Recompute even if existing artifacts are present |
| `--num-workers` | `int` | `16` | Number of worker processes for extraction. `2` or more enables multiprocessing parallelism |
| `--num-references` | `int` | `150` | Total number of reference observations to sample across all agents |

Notes:
- `--steps` requires comma-separated non-negative integers when not `all`.

## 2. Layer 2 `compute-step`

### Role

- Reads `extracted/step_{t}.pt` and computes per-step metrics, saving them to `step_metrics/`.

### Main Inputs

- `{input_dir}/analysis_artifacts/extracted/step_{t}.pt`

### Main Outputs

- `{input_dir}/analysis_artifacts/step_metrics/{metric_id}/step_{t}.npz`

`metric_id` (all available IDs):
- `observations_and_creations_clustered`
- `network_vs_similarity_latent`
- `wasserstein_similarity`
- `gromov_wasserstein_similarity`
- `network_vs_wasserstein_similarity`
- `network_vs_gromov_wasserstein_similarity`
- `latent_scatter_from_reference`
- `rsa_within_clusters`
- `mhng_edge_acceptance`
- `memorize_edge_acceptance`

### Execution Command

```bash
uv run python -m analysis.cli compute-step --input-dir /path/to/hydra/output
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--device` | `str` | `None` | Device override |
| `--steps` | `str` | `all` | Target steps. `"1,100,500"` or `"all"` |
| `--metrics` | `str` | `all` | Metric ID list (comma-separated) or `all`. Wildcard patterns (`*`/`?`) supported |
| `--refresh` | flag | `False` | Recompute even if existing metrics are present |
| `--num-workers` | `int` | `10` | Number of worker processes for step metric computation. `2` or more enables multiprocessing parallelism |

Notes:
- Specifying an unknown ID in `--metrics` results in an error.
- Patterns with wildcards (`*`, `?`) in `--metrics` are matched against registered metric IDs using `fnmatch`. An error occurs if no IDs match (e.g., `--metrics "network_vs_*"` to specify all network-related metrics at once).
- Prefixing an ID or pattern in `--metrics` with `!` excludes it. If only exclusions are specified, they are subtracted from all metrics (e.g., `--metrics "!rsa_within_clusters"` specifies all except rsa_within_clusters). Mixing include and exclude is supported (e.g., `--metrics "network_vs_*,!network_vs_gromov_wasserstein_similarity"`). Using `!` with an unknown ID without wildcards results in an error.

## 3. Layer 3 `compute-global`

### Role

- Aggregates Layer 2 step metrics across the full time series and computes FGW-MDS global metrics.

### Main Inputs

- `{input_dir}/analysis_artifacts/step_metrics/*`

### Main Outputs

- `{input_dir}/analysis_artifacts/global_metrics/{metric_id}.npz`

`metric_id` (all available IDs):
- `wasserstein_mds`
- `gromov_wasserstein_mds`
- `latent_scatter_pca_alignment`

### Execution Command

```bash
uv run python -m analysis.cli compute-global --input-dir /path/to/hydra/output
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--metrics` | `str` | `all` | Metric ID list (comma-separated) or `all`. Wildcard patterns (`*`/`?`) supported |
| `--refresh` | flag | `False` | Recompute even if existing metrics are present |
| `--num-workers` | `int` | `80` | Number of worker processes for internal global metric computation |
| `--fgw-no-align` | flag | `False` | Disable Procrustes alignment in FGW MDS |
| `--mds-seed` | `int` | `0` | Random seed for sklearn MDS initialization and iterative optimization in FGW MDS |

Notes:
- Specifying an unknown ID in `--metrics` results in an error.
- Patterns with wildcards (`*`, `?`) in `--metrics` are matched against registered metric IDs using `fnmatch`. An error occurs if no IDs match.
- Prefixing an ID or pattern in `--metrics` with `!` excludes it (e.g., `--metrics "!wasserstein_mds"` specifies all except wasserstein_mds).
- `--mds-seed` is the random seed used for sklearn MDS initialization and iterative optimization in FGW MDS.
- `latent_scatter_pca_alignment` will error if the `latent_scatter_from_reference` step metrics do not exist. This must be computed before rendering `latent_scatter_from_reference` figures with `render-frames`.

## 4. Layer 4 `visualize`

### Role

- Reads Layer 2/3 artifacts and saves images to `analysis_figures/`.
- Output formats are `png`, `pdf`, and `svg` with the same name.
- The subcommand used differs by figure category (`render` / `render-frames` / `render-snapshot` / `render-seeds` / `render-conditions`).

### Main Inputs

- `{input_dir}/analysis_artifacts/step_metrics/*`
- `{input_dir}/analysis_artifacts/global_metrics/*`

---

## 4.1 `render` — whole_step figures

Renders figures in the whole_step category. Loads all step data at once and generates a single figure.

### Main Outputs

- `{input_dir}/analysis_figures/{figure_id}/figure.png/.pdf/.svg`

### figure_id (all available IDs)

- `wasserstein_mds_trajectory`: Wasserstein MDS trajectory plot
- `gromov_wasserstein_mds_trajectory`: Gromov-Wasserstein MDS trajectory plot
- `wasserstein_mds_trajectory_cluster`: Cluster-colored Wasserstein MDS trajectory plot
- `gromov_wasserstein_mds_trajectory_cluster`: Cluster-colored Gromov-Wasserstein MDS trajectory plot
- `wasserstein_mds_pc_trajectory`: Wasserstein MDS trajectory projected onto global PCA axes
- `gromov_wasserstein_mds_pc_trajectory`: Gromov-Wasserstein MDS trajectory projected onto global PCA axes
- `wasserstein_mds_pc_trajectory_cluster`: Cluster-colored global PCA projection of Wasserstein MDS trajectory
- `gromov_wasserstein_mds_pc_trajectory_cluster`: Cluster-colored global PCA projection of Gromov-Wasserstein MDS trajectory
- `rsa_clusters`: Time-series RSA plot per cluster
- `rsa_agent_{agent_idx}`: RSA time-series plot per agent (dynamically registered, e.g., `rsa_agent_0`, `rsa_agent_1`, ...)

### Execution Command

```bash
uv run python -m analysis.cli render --input-dir /path/to/hydra/output --figure all
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--figure` | `str` | `all` | Figure ID (comma-separated, multiple allowed), `all`, or wildcard pattern (`*`/`?` supported, e.g., `rsa_agent_*`) |
| `--num-workers` | `int` | `30` | Number of rendering worker processes (actual parallelism is capped at number of tasks) |
| `--refresh` | flag | `False` | Re-render even if existing images are present |

Notes:
- `--figure all` resolves to all figures in the whole_step category.
- Explicitly specifying a stepwise figure results in the error `stepwise figure(s) must be rendered with 'render-frames': ...`.
- Explicitly specifying a snapshot figure results in the error `snapshot figure(s) must be rendered with 'render-snapshot': ...`.
- `*_mds_pc_trajectory*` figures will error if the Layer 3 artifacts lack `positions_pc` and `contrib_pc`.
- `rsa_agent_{agent_idx}` is dynamically registered; check available IDs with the `status` command.
- Patterns with wildcards (`*`, `?`) in `--figure` are matched using `fnmatch`. An error occurs if no IDs match.
- Prefixing an ID or pattern in `--figure` with `!` excludes it (e.g., `--figure "!gromov_wasserstein*"`). Mixing include and exclude is supported (e.g., `--figure "rsa_*,!rsa_agent_*"`).

---

## 4.2 `render-frames` — stepwise figures

Outputs figures in the stepwise category as per-step frame images. These become the input for `render-video`.

### Main Outputs

- `{input_dir}/analysis_figures/{figure_id}/step_{t}.png/.pdf/.svg`

### figure_id (all available IDs)

- `observations_and_creations_clustered`: Cluster-colored observation/creation distribution plot
- `observations_and_creations_clustered_legend`: Legend for the cluster-colored distribution plot
- `observations_and_creations_agents_clustered`: Cluster-colored observation/creation plot split by agent
- `observations_and_creations_agents_clustered_legend`: Legend for the per-agent cluster plot
- `observations_agents_clustered`: Cluster plot showing per-agent observation distributions side by side
- `wasserstein_similarity`: Heatmap of the inter-agent Wasserstein distance matrix
- `gromov_wasserstein_similarity`: Heatmap of the inter-agent Gromov-Wasserstein distance matrix
- `latent_scatter_from_reference`: Scatter plot of reference latent representations (requires `latent_scatter_pca_alignment` global metric in advance when creating video)
- `social_network`: Visualization of the social network structure
- `mhng_acceptance_matrix`: Heatmap of the MHNG acceptance rate matrix
- `mhng_acceptance_network`: MHNG acceptance rates overlaid on the network
- `memorize_acceptance_matrix`: Heatmap of the memorize acceptance rate matrix
- `memorize_acceptance_network`: Memorize acceptance rates overlaid on the network
- `both_acceptance_network`: MHNG and memorize acceptance rates side by side on the network

### Execution Command

```bash
uv run python -m analysis.cli render-frames --input-dir /path/to/hydra/output --figure social_network --steps all
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--figure` | `str` | `all` | Figure ID (comma-separated, multiple allowed), `all`, or wildcard pattern (`*`/`?` supported) |
| `--steps` | `str` | `all` | Step list. `all` automatically resolves available steps for each target figure |
| `--num-workers` | `int` | `30` | Number of rendering worker processes (actual parallelism is capped at number of tasks) |
| `--refresh` | flag | `False` | Re-render even if existing images are present |

Notes:
- `--figure all` resolves to all figures in the stepwise category.
- Explicitly specifying a whole_step figure results in the error `whole-step figure(s) must be rendered with 'render': ...`.
- Explicitly specifying a snapshot figure results in the error `snapshot figure(s) must be rendered with 'render-snapshot': ...`.
- Runs multiprocessing parallelism with `figure_ids × step` as one task each.
- With `--steps all`, available steps are resolved per figure. If resolution fails, it is reported as a missing dependency.
- Prefixing an ID or pattern in `--figure` with `!` excludes it (e.g., `--figure "!wasserstein*"`).

---

## 4.3 `render-snapshot` — snapshot figures

Renders figures in the snapshot category. Generates a figure with multiple specified steps arranged on a single image. `--steps` is required.

### Main Outputs

- `{input_dir}/analysis_figures/{figure_id}/steps_{s1}_{s2}_....png/.pdf/.svg`

### figure_id (all available IDs)

- `observations_agents_clustered_panel`: Panel plot of observation distributions across multiple steps
- `wasserstein_similarity_snapshot`: Snapshot plot of Wasserstein distance matrices from multiple steps placed side by side
- `gromov_wasserstein_similarity_snapshot`: Snapshot plot of Gromov-Wasserstein distance matrices from multiple steps placed side by side
- `wasserstein_similarity_average_snapshot`: Snapshot plot of averaged Wasserstein distance matrices from multiple steps
- `gromov_wasserstein_similarity_average_snapshot`: Snapshot plot of averaged Gromov-Wasserstein distance matrices from multiple steps
- `both_acceptance_network_snapshot`: Snapshot plot of MHNG/memorize acceptance rate networks from multiple steps placed side by side
- `both_acceptance_average_network_snapshot`: Snapshot plot of averaged MHNG/memorize acceptance rate networks from multiple steps
- `gromov_wasserstein_mds_pc_trajectory_cluster_segmented`: Cluster-colored global PCA-projected GW MDS trajectory split into intervals using `--steps` as boundaries, with each interval in a separate subplot

### Execution Command

```bash
uv run python -m analysis.cli render-snapshot \
	--input-dir /path/to/hydra/output \
	--figure observations_agents_clustered_panel \
	--steps 1,500,1000

# Segmented trajectory plot (--steps are used as interval boundaries)
uv run python -m analysis.cli render-snapshot \
	--input-dir /path/to/hydra/output \
	--figure gromov_wasserstein_mds_pc_trajectory_cluster_segmented \
	--steps 100,1000,5000
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--figure` | `str` | `all` | Figure ID (comma-separated, multiple allowed), `all`, or wildcard pattern (`*`/`?` supported) |
| `--steps` | `str` | required | Target step list (e.g., `1,500,1000`) |
| `--num-workers` | `int` | `30` | Number of rendering worker processes (actual parallelism is capped at number of tasks) |
| `--refresh` | flag | `False` | Re-render even if existing images are present |

Notes:
- `--figure all` resolves to all figures in the snapshot category.
- Prefixing an ID or pattern in `--figure` with `!` excludes it.

---

## 4.4 `render-seeds` — multi-seed aggregation

Renders RSA aggregation figures spanning multiple Hydra output directories (seeds). Aligns on steps common across seeds and computes mean and standard deviation, then outputs as a whole_step figure.

### Main Outputs

- `{output_dir}/analysis_figures/{figure_id}/figure.png/.pdf/.svg`

### figure_id (all available IDs)

- `rsa_clusters_seeds`: Overlaid cluster-level RSA time series across multiple seeds with mean line and ±1σ band
- `rsa_agent_{agent_idx}_seeds`: Per-agent RSA time series across multiple seeds (dynamically registered)

### Execution Command

```bash
uv run python -m analysis.cli render-seeds \
	--input-dirs /out/seed0 /out/seed1 /out/seed2 /out/seed3 \
	--output-dir /out/multi_seed \
	--figure all
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dirs` | `str ...` | required | List of Hydra output directories. At least 2 required |
| `--output-dir` | `str` | required | Output root. Images are saved under `output_dir/analysis_figures/` |
| `--figure` | `str` | `all` | Figure ID (comma-separated, multiple allowed), `all`, or wildcard pattern (`*`/`?` supported, e.g., `rsa_agent_*_seeds`) |
| `--num-workers` | `int` | `4` | Number of rendering worker processes |
| `--refresh` | flag | `False` | Re-render even if existing images are present |

Notes:
- Each seed requires at minimum `analysis_artifacts/step_metrics/rsa_within_clusters/step_{t}.npz`.
- The steps used are the intersection of available steps across all seeds. An error occurs if there are no common steps.
- `rsa_clusters_seeds` / `rsa_agent_{agent_idx}_seeds` are only registered when the number of agents and clusters match across all seeds.
- Output is created under `--output-dir`, not under each seed's `input_dir`.
- Available figure IDs depend on the number of seeds and data shapes; run `--figure all` as a dry run or check with the `status` command.
- Prefixing an ID or pattern in `--figure` with `!` excludes it (e.g., `--figure "!rsa_agent_*_seeds"`).

---

## 4.5 `render-conditions` — multi-condition comparison

Generates comparison plots of RSA time series overlaid across multiple experimental conditions (each with multiple seeds).

### Execution Command

```bash
uv run python -m analysis.cli render-conditions \
	--conditions \
	"w/ creation:/out/a/seed0,/out/a/seed1" \
	"w/o creation:/out/b/seed0,/out/b/seed1" \
	--output-dir /out/conditions
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--conditions` | `str ...` | required | Seed groups per condition. Each element is in `label:dir1,dir2,...` format |
| `--output-dir` | `str` | required | Output root. Images are saved under `output_dir/analysis_figures/rsa_clusters_conditions/` |
| `--refresh` | flag | `False` | Re-render even if existing images are present |

Notes:
- Currently dedicated to `rsa_clusters_conditions`.
- Each condition internally loads a multi-seed RSA aggregation and overlays them across conditions.
- An error occurs if no common steps can be obtained for any condition.

## 5. Layer 5 `video` (`render-video`)

### Role

- Generates video from stepwise frames (`analysis_figures/{figure_id}/step_{t}.png`) already produced in Layer 4.
- Does not recompute upstream; uses only existing frames.

### Main Inputs

- `{input_dir}/analysis_figures/{figure_id}/step_{t}.png`

### Main Outputs

- `{input_dir}/analysis_videos/{figure_id}.mp4`

### figure_id (all available IDs)

All stepwise category figures listed in `render-frames` (Section 4.2) can be specified.

### Execution Command

```bash
uv run python -m analysis.cli render-video --input-dir /path/to/hydra/output --figure all --steps all
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
| `--figure` | `str` | `all` | Stepwise figure ID (comma-separated, multiple allowed), `all`, or wildcard pattern (`*`/`?` supported) |
| `--steps` | `str` | `all` | Step list. `all` uses all frames under the figure directory in ascending step order |
| `--fps` | `int` | `60` | Output video FPS |
| `--num-workers` | `int` | `30` | Number of video encoding worker processes (actual parallelism is capped at number of tasks) |
| `--refresh` | flag | `False` | Regenerate even if existing video is present |

Notes:
- `--figure all` resolves to stepwise figures only.
- whole_step figures (e.g., `rsa_clusters`) cannot be specified.
- Patterns with wildcards (`*`, `?`) in `--figure` are matched using `fnmatch`. An error occurs if no IDs match.
- Prefixing an ID or pattern in `--figure` with `!` excludes it (e.g., `--figure "!wasserstein*"`).
- Only concatenates frame PNGs. Fails if the corresponding `analysis_figures/{figure_id}/step_{t}.png` has not been generated by `render-frames`.
- If frame sizes differ between steps, pads to the maximum size when generating the MP4.

## 6. Typical Execution Sequence

```bash
# Layer 1
uv run python -m analysis.cli extract --input-dir /path/to/hydra/output

# Layer 2
uv run python -m analysis.cli compute-step --input-dir /path/to/hydra/output

# Layer 3 (FGW-MDS + latent scatter PCA sign alignment)
uv run python -m analysis.cli compute-global --input-dir /path/to/hydra/output

# Layer 4.1 render (whole_step figures)
uv run python -m analysis.cli render --input-dir /path/to/hydra/output --figure all

# Layer 4.2 render-frames (stepwise figures)
uv run python -m analysis.cli render-frames --input-dir /path/to/hydra/output --figure all --steps all

# Layer 4.3 render-snapshot (snapshot figures)
uv run python -m analysis.cli render-snapshot --input-dir /path/to/hydra/output --figure all --steps 1,500,1000

# Layer 4.4 render-seeds (multi-seed aggregation)
uv run python -m analysis.cli render-seeds --input-dirs /out/seed0 /out/seed1 /out/seed2 /out/seed3 --output-dir /out/multi_seed --figure all

# Layer 4.5 render-conditions (multi-condition comparison)
uv run python -m analysis.cli render-conditions --conditions "w/ creation:/out/a/seed0,/out/a/seed1" "w/o creation:/out/b/seed0,/out/b/seed1" --output-dir /out/conditions

# Layer 5
uv run python -m analysis.cli render-video --input-dir /path/to/hydra/output --figure all --steps all
```

## 7. Auxiliary Command `status`

Displays the list of available figure IDs grouped by category. Useful for checking actual IDs of dynamically registered figures (e.g., `rsa_agent_*`).

```bash
uv run python -m analysis.cli status --input-dir /path/to/hydra/output
```

### Options (all available parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | required | Hydra output directory |
