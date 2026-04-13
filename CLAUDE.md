# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCENA (Social Creativity and Epistemic Norm Alignment) is a multi-agent simulation framework studying emergent social creativity. Agents with VAE-based generative models interact on social networks, creating and sharing observations via Metropolis-Hastings Name Game (MHNG) to develop shared epistemic norms.

## Commands

**Run simulation:**
```bash
uv run main.py
```

**Override Hydra config params at the command line:**
```bash
uv run main.py seed=0 device=cuda:1 num_creations=0 num_steps=1000
```

**Run multiple seeds in parallel (via screen):**
```bash
bash scripts/alife.bash
```

**Run analysis pipeline (extract → compute-step → compute-global → render):**
```bash
uv run python -m analysis.cli extract --input-dir /path/to/hydra/output
uv run python -m analysis.cli compute-step --input-dir /path/to/hydra/output
uv run python -m analysis.cli compute-global --input-dir /path/to/hydra/output
uv run python -m analysis.cli render --input-dir /path/to/hydra/output --figure all
uv run python -m analysis.cli render-frames --input-dir /path/to/hydra/output --figure all --steps all
```

**Install dependencies:**
```bash
uv sync
```

## Architecture

### Core simulation loop (`main.py`)
Uses [Hydra](https://hydra.cc/) for config management. Config is defined in [config.py](config.py) as dataclasses and loaded from [config/config.yaml](config/config.yaml). Outputs go to a Hydra-managed directory (default: NFS path); step data saved as `data/step_{N}.pt`, model checkpoints as `checkpoints/step_{N}/agent_{i}.pt`.

Two training phases:
1. **Initial phase** (`num_initial_steps`): Each agent trains a standard VAE on its private buffer (no communication).
2. **Social phase** (`num_steps`): Each step runs: Create → Communicate → MHNG → Memorize → Update Critics → Update Models.

### Agent (`srcaif/agent.py`)
Each agent has:
- **Encoder** → diagonal Gaussian in latent space
- **Decoder** → reconstructs observations from latents
- **Critic** (Wasserstein discriminator) — distinguishes self-generated vs social observations; score drives curiosity in `create()`
- **ObservationBuffer** — fixed-capacity replay buffer of memorized observations
- **TemporaryPool** (`local_shared_pool`) — stores received observations from neighbors each step
- **TemporaryPool** (`mhng_samples`) — stores MHNG-accepted samples used for critic/model updates

Key methods:
- `create()` — optimizes new observations in obs-space to maximize critic score (curiosity) and homeostasis
- `mhng()` — Metropolis-Hastings acceptance based on decoder log-likelihood ratio
- `memorize_observations()` — active inference-based selection of which received observations to add to buffer
- `update_critic()` / `update_model()` — standard training steps; `update_model_initial()` for VAE-only phase
- `save_model()` / `load_model()` — checkpoint to/from `checkpoints/step_{N}/agent_{i}.pt`; final saves go to `checkpoints/final/agent_{i}.pt`

### Society (`srcaif/society.py`)
Manages the agent collection and NetworkX graph. Supports multiple topologies: `small_world`, `scale_free`, `connected_caveman`, `fully_connected`, `ring`, `grid`, `random`.

`Society.step()` orchestrates all phases with `ThreadPoolExecutor` parallelism. `social_latents` (MHNG-accepted latents) persist across steps and drive creation in the next step.

### Models (`srcaif/models/`)
- `encoders.py` / `decoders.py` — MLP-based; 1-D (flat vector) observations only
- `critics.py` — discriminator taking `(obs, latent)` pairs
- `networks.py` — shared network building utilities

### Data (`data.py`)
`DataFactory` initializes agent buffers. Only `vector` mode: each agent gets samples from a distinct Gaussian cluster in R^d, with cluster centers arranged on a grid scaled by `data.vector_separation_scale`.

### Analysis pipeline (`analysis/`)
CLI pipeline for post-hoc analysis. See `analysis/LAYERS_AND_COMMANDS.md` for full command reference and `analysis/ARCHITECTURE.md` for data flow details.
- `extract` — extracts per-step artifacts from checkpoints and step data
- `compute-step` — computes per-step metrics (FGW similarity, RSA, acceptance rates, EFE landscape, etc.)
- `compute-global` — computes global metrics (FGW MDS trajectories) across all steps
- `render` / `render-frames` / `render-snapshot` / `render-seeds` / `render-conditions` — generates figures
- `render-video` — encodes step frames into MP4
- `status` — shows pipeline completion status for a run directory

**Multi-seed analysis scripts** in `scripts/`:
- `scripts/analysis_alife_seeds.bash` — runs the full pipeline across multiple seeds with `--parallel-extract` and `--seeds` options
- `scripts/alife_compute_global_metrics_for_seeds.bash` / `scripts/alife_render_figures_for_seeds.bash` — targeted per-phase wrappers

### Metrics / Analysis (`srcaif/metrics.py`)
- `compute_rsa_within_clusters()` — RSA between latent and observation RDMs within network communities
- `compute_fgw_similarity_vectorized()` — Fused Gromov-Wasserstein similarity between agent representation geometries
- TE (Gaussian and KSG estimators) and CCM with sliding-window wrappers

### Config (`config.py`, `config/config.yaml`)
All hyperparameters live in `Config` dataclass. Notable params:
- `creation_preference`: `"reconst"` (reconstruct own observations) or `"social"` (decode social prior)
- `beta_social` / `beta_individual`: KL weights in the social VAE loss
- `lambda_creative`: weight of homeostasis term in creation loss
- `memorization_alpha` / `memorization_temperature`: control selectivity of buffer memorization
- `network.network_type` + `network.network_params`: topology selection

WandB logging is controlled by `enable_wandb`; step metrics use custom step metrics (`initial_step`, `social_step`).
