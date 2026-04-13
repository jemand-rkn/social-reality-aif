from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from analysis.core.eval_context import EvalContext
from analysis.core.registry import ExtractionRuntimeContext
from analysis.core.store import ArtifactStore
from analysis.core.types import AnalysisPaths, ExtractRunConfig, StepExtractContext
from analysis.data_extraction.rebuild_society import (
    build_society,
    load_checkpoint,
    load_config,
    parse_step,
)
from analysis.data_extraction.step_extractors import build_default_registry
from utils import fix_seed


def _extract_single_step_worker(input_dir: str, step: int, device_str: str, refresh: bool) -> int:
    """Process worker for step extraction.

    The worker initializes its own pipeline/store to avoid sharing mutable state
    across processes.
    """
    input_path = Path(input_dir)
    worker_config = ExtractRunConfig(
        input_dir=input_path,
        device=device_str,
        refresh=refresh,
        num_workers=1,
    )
    pipeline = StepExtractionPipeline(worker_config)
    cfg, _ = pipeline._resolve_cfg()

    # Keep per-step randomness deterministic regardless of process scheduling.
    fix_seed(int(cfg.seed) + int(step))

    checkpoints_dir = input_path / "checkpoints"
    data_dir = input_path / "data"
    num_agents = int(cfg.num_agents)
    device = torch.device(device_str)

    pipeline._extract_single_step(
        cfg=cfg,
        step=step,
        checkpoints_dir=checkpoints_dir,
        data_dir=data_dir,
        num_agents=num_agents,
        device=device,
    )
    return step


class StepExtractionPipeline:
    def __init__(self, config: ExtractRunConfig):
        self.config = config
        self.paths = AnalysisPaths.from_input_dir(config.input_dir)
        self.store = ArtifactStore(self.paths)
        self.registry = build_default_registry()

    def _resolve_cfg(self):
        cfg = load_config(self.config.input_dir)
        obs_shape = tuple(cfg.data.obs_shape)
        return cfg, obs_shape

    def _resolve_steps(self, checkpoints_dir: Path) -> List[int]:
        step_dirs = [d for d in checkpoints_dir.glob("step_*") if d.is_dir()]
        steps = sorted(parse_step(d) for d in step_dirs)
        steps = [s for s in steps if s >= 0]
        if self.config.steps:
            wanted = set(self.config.steps)
            steps = [s for s in steps if s in wanted]
        return steps

    def _load_data_payload(self, data_dir: Path, step: int) -> Dict:
        data_path = data_dir / f"step_{step}.pt"
        if not data_path.exists():
            return {}
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        return payload if isinstance(payload, dict) else {}

    def _build_runtime_context(self, cfg, step: int, checkpoints_dir: Path, data_dir: Path, num_agents: int, device: torch.device):
        step_dir = checkpoints_dir / f"step_{step}"
        step_context = StepExtractContext(
            step=step,
            checkpoint_dir=step_dir,
            data_path=data_dir / f"step_{step}.pt",
        )
        society = build_society(cfg, device)
        load_checkpoint(society, step_dir, num_agents)
        data_payload = self._load_data_payload(data_dir, step)
        eval_context = EvalContext(
            society=society,
            creations=data_payload.get("creations"),
            mhng_results=data_payload.get("mhng_results"),
            memorize_results=data_payload.get("memorize_results"),
            step=step,
            obs_shape=tuple(cfg.data.obs_shape),
            num_references=self.config.num_references if self.config.num_references is not None else 150,
        )
        runtime_context = ExtractionRuntimeContext(
            step_context=step_context,
            cfg=cfg,
            society=society,
            data_payload=data_payload,
            eval_context=eval_context,
            store=self.store,
        )
        return runtime_context, num_agents

    def _extract_single_step(
        self,
        cfg,
        step: int,
        checkpoints_dir: Path,
        data_dir: Path,
        num_agents: int,
        device: torch.device,
    ) -> None:
        if self.store.has_extracted_step(step) and not self.config.refresh:
            return

        runtime_context, _ = self._build_runtime_context(cfg, step, checkpoints_dir, data_dir, num_agents, device)
        payload = {
            "step": step,
            "obs_shape": tuple(cfg.data.obs_shape),
            "num_agents": num_agents,
            "num_references": runtime_context.eval_context.num_references,
            "extractor_ids": self.registry.list_ids(),
        }
        payload = self.registry.run_all(runtime_context, payload)
        self.store.save_extracted_step(step, payload)

    def _extract_single_step_with_registry(
        self,
        registry,
        cfg,
        step: int,
        checkpoints_dir: Path,
        data_dir: Path,
        num_agents: int,
        device: torch.device,
    ) -> None:
        """Run a specific extractor registry for one step (no save_extracted_step)."""
        step_dir = checkpoints_dir / f"step_{step}"
        step_context = StepExtractContext(
            step=step,
            checkpoint_dir=step_dir,
            data_path=data_dir / f"step_{step}.pt",
        )

        society = build_society(cfg, device)
        load_checkpoint(society, step_dir, num_agents)

        data_payload = self._load_data_payload(data_dir, step)
        eval_context = EvalContext(
            society=society,
            creations=data_payload.get("creations"),
            mhng_results=data_payload.get("mhng_results"),
            memorize_results=data_payload.get("memorize_results"),
            step=step,
            obs_shape=tuple(cfg.data.obs_shape),
            num_references=self.config.num_references if self.config.num_references is not None else 150,
        )

        runtime_context = ExtractionRuntimeContext(
            step_context=step_context,
            cfg=cfg,
            society=society,
            data_payload=data_payload,
            eval_context=eval_context,
            store=self.store,
        )

        payload = {
            "step": step,
            "obs_shape": tuple(cfg.data.obs_shape),
            "num_agents": num_agents,
            "num_references": eval_context.num_references,
            "extractor_ids": registry.list_ids(),
        }
        registry.run_all(runtime_context, payload)

    def run(self) -> None:
        self.store.ensure_layout()

        cfg, obs_shape = self._resolve_cfg()
        if len(obs_shape) != 1 or obs_shape[0] < 2:
            raise ValueError(f"Unsupported obs_shape for extraction: {obs_shape}")

        checkpoints_dir = self.config.input_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

        data_dir = self.config.input_dir / "data"
        steps = self._resolve_steps(checkpoints_dir)
        if not steps:
            raise FileNotFoundError(f"No valid step_* checkpoints found in {checkpoints_dir}")

        cfg_seed = int(cfg.seed)
        fix_seed(cfg_seed)

        device_str = self.config.device if self.config.device else str(cfg.device)
        device = torch.device(device_str)
        num_agents = int(cfg.num_agents)
        num_workers = max(1, int(self.config.num_workers))

        if num_workers < 2:
            for step in tqdm(steps, desc="Extracting step artifacts"):
                self._extract_single_step(cfg=cfg, step=step, checkpoints_dir=checkpoints_dir,
                                          data_dir=data_dir, num_agents=num_agents, device=device)
            return

        pending_steps = steps if self.config.refresh else [step for step in steps if not self.store.has_extracted_step(step)]
        if not pending_steps:
            return

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {
                executor.submit(_extract_single_step_worker, str(self.config.input_dir),
                                step, device_str, bool(self.config.refresh)): step
                for step in pending_steps
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting step artifacts (mp)"):
                step = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to extract step={step} in multiprocessing mode") from exc

