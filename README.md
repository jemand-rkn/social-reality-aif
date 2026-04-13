# Social Reality Construction via Active Inference: Modeling the Dialectic of Conformity and Creativity

[![arXiv](https://img.shields.io/badge/arXiv-2604.09026-b31b1b.svg)](https://arxiv.org/abs/2604.09026)

This repository implements a multi-agent simulation in which agents equipped with VAE-based generative models interact on a social network. Agents create novel observations, communicate via the Metropolis-Hastings Naming Game (MHNG), and selectively memorize each other's creations—giving rise to emergent social groups and shared representational structure.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/jemand-rkn/social-reality-aif.git
cd social-reality-aif
uv sync

# 2. Run the simulation (default config: 14 agents, connected caveman network, 5000 steps)
uv run main.py

# 3. Run the full analysis pipeline on the output
bash scripts/alife2026_analysis.bash outputs/social_reality_construction_aif/14agents/<run-name>
```

Figures are saved under `<run-dir>/analysis_figures/` and videos under `<run-dir>/analysis_videos/`.

---

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- CUDA-capable GPU recommended (configurable via `device` parameter)

Install all dependencies:

```bash
uv sync
```

---

## Running Simulations

### Single run

```bash
uv run main.py
```

Config is managed by [Hydra](https://hydra.cc/). Override any parameter at the command line:

```bash
uv run main.py seed=0 device=cuda:1 num_steps=5000 num_creations=6
```

Outputs are saved to `outputs/social_reality_construction_aif/14agents/<run-name>/` by default, containing:
- `data/step_{N}.pt` — per-step event data
- `checkpoints/step_{N}/agent_{i}.pt` — model checkpoints per step
- `.hydra/config.yaml` — full config snapshot
- `train.log` — training log

### Multiple seeds in parallel

```bash
bash scripts/alife.bash
```

This launches seeds 0–4 in separate `screen` sessions.

---

## Analysis Pipeline

After a simulation run completes, use the analysis pipeline to compute metrics and generate figures.

### One-shot pipeline (recommended)

```bash
bash scripts/alife2026_analysis.bash <run-dir>
```

This runs all 6 stages in sequence and produces figures and videos. Optional arguments:

```bash
bash scripts/alife2026_analysis.bash <run-dir> 1000,2000,3000,4000,5000 --refresh --num-workers=32
```

### Step-by-step

```bash
# Layer 1: Extract per-step payloads from checkpoints
uv run python -m analysis.cli extract --input-dir <run-dir>

# Layer 2: Compute per-step metrics (Wasserstein/GW distances, RSA, acceptance rates, ...)
uv run python -m analysis.cli compute-step --input-dir <run-dir>

# Layer 3: Compute global metrics (MDS trajectories, PCA alignment)
uv run python -m analysis.cli compute-global --input-dir <run-dir>

# Layer 4a: Render whole-step figures (MDS trajectories, RSA time series)
uv run python -m analysis.cli render --input-dir <run-dir> --figure all

# Layer 4b: Render per-step frames (observation distributions, network, heatmaps, ...)
uv run python -m analysis.cli render-frames --input-dir <run-dir> --figure all --steps all

# Layer 4c: Render snapshot figures (side-by-side panels at selected steps)
uv run python -m analysis.cli render-snapshot --input-dir <run-dir> --figure all --steps 1000,2000,3000,4000,5000

# Layer 5: Encode frame sequences into MP4 videos
uv run python -m analysis.cli render-video --input-dir <run-dir> --figure all --steps all
```

### Multi-seed aggregation

To compute and render RSA statistics pooled across multiple seeds:

```bash
uv run python -m analysis.cli render-seeds \
    --input-dirs /out/seed0 /out/seed1 /out/seed2 /out/seed3 \
    --output-dir /out/multi_seed \
    --figure all
```

### Condition comparison

To compare two experimental conditions (e.g., w/ vs. w/o creation):

```bash
uv run python -m analysis.cli render-conditions \
    --conditions \
    "w/ creation:/out/creation/seed0,/out/creation/seed1" \
    "w/o creation:/out/no_creation/seed0,/out/no_creation/seed1" \
    --output-dir /out/comparison
```

### Check pipeline status

```bash
uv run python -m analysis.cli status --input-dir <run-dir>
```

For full documentation of all CLI options, see [analysis/LAYERS_AND_COMMANDS.md](analysis/LAYERS_AND_COMMANDS.md).

---

## Output Structure

```
<run-dir>/
├── data/step_{N}.pt                    # Raw step event data
├── checkpoints/step_{N}/agent_{i}.pt   # Model checkpoints
├── .hydra/config.yaml                  # Run configuration
├── analysis_artifacts/
│   ├── extracted/                      # Extracted per-step payloads
│   ├── step_metrics/                   # Per-step metrics (distances, RSA, ...)
│   └── global_metrics/                 # Global metrics (MDS trajectories, ...)
├── analysis_figures/
│   ├── wasserstein_mds_trajectory/     # MDS trajectory plots
│   ├── gromov_wasserstein_mds_*/       # GW-MDS trajectory plots
│   ├── rsa_clusters/                   # Cluster RSA time series
│   ├── observations_*/                 # Observation distribution frames
│   ├── social_network/                 # Network visualization frames
│   ├── *_acceptance_*/                 # Acceptance rate heatmaps and networks
│   └── ...
└── analysis_videos/
    └── {figure_id}.mp4                 # Animated videos from frame sequences
```

---

## Paper

> **[Social Reality Construction via Active Inference: Modeling the Dialectic of Conformity and Creativity](https://arxiv.org/abs/2604.09026)**
>
> Kentaro Nomura, Takato Horii

## Citation

```bibtex
@misc{nomura2026socialrealityconstructionactive,
      title={Social Reality Construction via Active Inference: Modeling the Dialectic of Conformity and Creativity}, 
      author={Kentaro Nomura and Takato Horii},
      year={2026},
      eprint={2604.09026},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2604.09026}, 
}
```
