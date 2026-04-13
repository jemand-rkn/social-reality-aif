from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.compute_analysis.global_metrics.pipeline import GlobalMetricPipeline
from analysis.compute_analysis.step.pipeline import StepMetricPipeline
from analysis.core.types import (
    ExtractRunConfig,
    GlobalMetricRunConfig,
    RenderRunConfig,
    RenderSeedsRunConfig,
    StepMetricRunConfig,
    VideoRunConfig,
)
from analysis.data_extraction.pipeline import StepExtractionPipeline
from analysis.video.pipeline import VideoBatchRunner, VideoPipeline
from analysis.visualize.pipeline import FigureBatchRunner, FigurePipeline, MultiSeedFigurePipeline


def _parse_steps(steps_arg: Optional[str]) -> Optional[List[int]]:
    if steps_arg is None or steps_arg.strip().lower() == "all":
        return None
    values = [item.strip() for item in steps_arg.split(",") if item.strip()]
    steps = [int(v) for v in values]
    if any(step < 0 for step in steps):
        raise ValueError("All steps must be non-negative integers.")
    return steps


def _normalize_cli_path(path_arg: str) -> Path:
    # Users sometimes over-escape '=' inside quoted shell paths, producing a literal '\\='.
    return Path(path_arg.replace("\\=", "="))


def _parse_figures(figures_arg: Optional[str], available: List[str]) -> List[str]:
    if figures_arg is None or figures_arg.strip().lower() == "all":
        return list(available)
    values = [item.strip() for item in figures_arg.split(",") if item.strip()]
    include_pats = [p for p in values if not p.startswith("!")]
    exclude_pats = [p[1:] for p in values if p.startswith("!")]
    if include_pats:
        result: List[str] = []
        unknown: List[str] = []
        for pat in include_pats:
            if "*" in pat or "?" in pat:
                matched = [fid for fid in available if fnmatch.fnmatch(fid, pat)]
                if not matched:
                    unknown.append(pat)
                else:
                    for fid in matched:
                        if fid not in result:
                            result.append(fid)
            else:
                if pat not in available:
                    unknown.append(pat)
                elif pat not in result:
                    result.append(pat)
        if unknown:
            raise ValueError(f"Unknown figure id(s): {', '.join(unknown)}")
    else:
        result = list(available)
    if exclude_pats:
        excluded: List[str] = []
        for pat in exclude_pats:
            if "*" in pat or "?" in pat:
                excluded.extend(fid for fid in result if fnmatch.fnmatch(fid, pat))
            elif pat in result:
                excluded.append(pat)
            else:
                raise ValueError(f"Unknown figure id in exclusion: {pat!r}")
        result = [fid for fid in result if fid not in excluded]
    return result


def _stepwise_figure_ids(pipeline: FigurePipeline) -> List[str]:
    return [figure_id for figure_id in pipeline.registry.list_ids() if pipeline.registry.get(figure_id).category == "stepwise"]


def _whole_step_figure_ids(pipeline: FigurePipeline) -> List[str]:
    return [
        figure_id
        for figure_id in pipeline.registry.list_ids()
        if pipeline.registry.get(figure_id).category == "whole_step"
    ]


def _snapshot_figure_ids(pipeline: FigurePipeline) -> List[str]:
    return [figure_id for figure_id in pipeline.registry.list_ids() if pipeline.registry.get(figure_id).category == "snapshot"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analysis pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Run data extraction stage")
    extract.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    extract.add_argument("--device", type=str, default=None, help="Override device")
    extract.add_argument("--steps", type=str, default="all", help='Step list, e.g. "1,100,500" or "all"')
    extract.add_argument("--refresh", action="store_true", help="Recompute even if extracted artifacts exist")
    extract.add_argument("--num-workers", type=int, default=16, help="Number of worker processes for extraction")
    extract.add_argument("--num-references", type=int, default=None, help="Override cfg.num_references (total reference obs sampled across all agents, default: 150)")

    compute_step = subparsers.add_parser("compute-step", help="Run step metric computation stage")
    compute_step.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    compute_step.add_argument("--device", type=str, default=None, help="Override device")
    compute_step.add_argument("--steps", type=str, default="all", help='Step list, e.g. "1,100,500" or "all"')
    compute_step.add_argument(
        "--metrics",
        type=str,
        default="all",
        help='Metric id list, e.g. "wasserstein_similarity,mhng_edge_acceptance" or "all"',
    )
    compute_step.add_argument("--refresh", action="store_true", help="Recompute even if metric files exist")
    compute_step.add_argument("--num-workers", type=int, default=10, help="Number of worker processes for step metrics")

    compute_global = subparsers.add_parser("compute-global", help="Run global metric computation stage")
    compute_global.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    compute_global.add_argument(
        "--metrics",
        type=str,
        default="all",
        help='Metric id list, e.g. "creation_te,wasserstein_mds" or "all"',
    )
    compute_global.add_argument("--refresh", action="store_true", help="Recompute even if metric files exist")
    compute_global.add_argument(
        "--num-workers",
        type=int,
        default=80,
        help="Worker processes for global metric computation (PCMCI)",
    )
    compute_global.add_argument(
        "--fgw-no-align",
        action="store_true",
        help="Disable Procrustes alignment for FGW MDS",
    )
    compute_global.add_argument(
        "--mds-seed",
        type=int,
        default=0,
        help="Random seed for sklearn MDS initialization/refinement",
    )
    render = subparsers.add_parser("render", help="Render figure(s)")
    render.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    render.add_argument(
        "--figure",
        type=str,
        default="all",
        help='Figure id list, e.g. "social_network,mhng_acceptance_matrix" or "all"',
    )
    render.add_argument("--step", type=int, default=None, help="Target step for stepwise figures")
    render.add_argument(
        "--steps",
        type=str,
        default=None,
        help='Deprecated for render; snapshot figures should use render-snapshot',
    )
    render.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Worker processes for rendering figures (default: cpu_count - 1)",
    )
    render.add_argument("--refresh", action="store_true", help="Re-render even if output file exists")

    render_frames = subparsers.add_parser("render-frames", help="Render multiple stepwise frames")
    render_frames.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    render_frames.add_argument(
        "--figure",
        type=str,
        default="all",
        help='Stepwise figure id list, e.g. "social_network,mhng_acceptance_matrix" or "all"',
    )
    render_frames.add_argument("--steps", type=str, default="all", help='Step list, e.g. "0,10,20" or "all"')
    render_frames.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Worker processes for rendering frames across figures x steps (default: cpu_count - 1)",
    )
    render_frames.add_argument("--refresh", action="store_true", help="Re-render even if output files exist")

    render_snapshot = subparsers.add_parser("render-snapshot", help="Render snapshot figure(s) from a selected step list")
    render_snapshot.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    render_snapshot.add_argument(
        "--figure",
        type=str,
        default="all",
        help='Snapshot figure id list, e.g. "observations_agents_clustered_panel" or "all"',
    )
    render_snapshot.add_argument(
        "--steps",
        type=str,
        required=True,
        help='Target step list, e.g. "1,500,1000,1800"',
    )
    render_snapshot.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Worker processes for rendering snapshot figures (default: cpu_count - 1)",
    )
    render_snapshot.add_argument("--refresh", action="store_true", help="Re-render even if output file exists")

    render_video = subparsers.add_parser("render-video", help="Build video(s) from rendered frame PNGs")
    render_video.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")
    render_video.add_argument(
        "--figure",
        type=str,
        default="all",
        help='Stepwise figure id list, e.g. "social_network,mhng_acceptance_matrix" or "all"',
    )
    render_video.add_argument("--steps", type=str, default="all", help='Step list, e.g. "0,10,20" or "all"')
    render_video.add_argument("--fps", type=int, default=60, help="Output video FPS")
    render_video.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Worker processes for figure-wise video encoding (default: cpu_count - 1)",
    )
    render_video.add_argument("--refresh", action="store_true", help="Rebuild even if output video exists")

    render_seeds = subparsers.add_parser("render-seeds", help="Render multi-seed aggregated RSA figures")
    render_seeds.add_argument(
        "--input-dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of Hydra output directories (one per seed)",
    )
    render_seeds.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures (figures saved under output_dir/analysis_figures/)",
    )
    render_seeds.add_argument(
        "--figure",
        type=str,
        default="all",
        help='Figure id list, e.g. "rsa_clusters_seeds,rsa_agent_0_seeds" or "all"',
    )
    render_seeds.add_argument("--num-workers", type=int, default=4, help="Worker processes for rendering")
    render_seeds.add_argument("--refresh", action="store_true", help="Re-render even if output file exists")

    render_conditions = subparsers.add_parser(
        "render-conditions",
        help="Render rsa_clusters comparing multiple conditions across seeds",
    )
    render_conditions.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        required=True,
        metavar="LABEL:DIR1,DIR2,...",
        help=(
            'One entry per condition, format "label:dir1,dir2,...". '
            'Example: "w/ creation:/path/seed0,/path/seed1" "w/o creation:/path/seed0,/path/seed1"'
        ),
    )
    render_conditions.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (figure saved under output_dir/analysis_figures/rsa_clusters_conditions/)",
    )
    render_conditions.add_argument("--refresh", action="store_true", help="Re-render even if output file exists")

    status = subparsers.add_parser("status", help="Show available analysis figures")
    status.add_argument("--input-dir", type=str, required=True, help="Hydra output directory")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        config = ExtractRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            device=args.device,
            steps=_parse_steps(args.steps),
            refresh=bool(args.refresh),
            num_workers=max(1, int(args.num_workers)),
            num_references=args.num_references,
        )
        pipeline = StepExtractionPipeline(config)
        pipeline.run()
        print(f"Extraction completed: {config.input_dir / 'analysis_artifacts' / 'extracted'}")
        return

    if args.command == "compute-step":
        metrics = None
        if args.metrics.strip().lower() != "all":
            metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
        config = StepMetricRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            device=args.device,
            steps=_parse_steps(args.steps),
            metrics=metrics,
            refresh=bool(args.refresh),
            num_workers=max(1, int(args.num_workers)),
        )
        pipeline = StepMetricPipeline(config)
        pipeline.run()
        print(f"Step metrics completed: {config.input_dir / 'analysis_artifacts' / 'step_metrics'}")
        return

    if args.command == "compute-global":
        metrics = None
        if args.metrics.strip().lower() != "all":
            metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

        num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 1) - 1)

        config = GlobalMetricRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            metrics=metrics,
            refresh=bool(args.refresh),
            num_workers=max(1, int(num_workers)),
            fgw_no_align=bool(args.fgw_no_align),
            mds_seed=int(args.mds_seed),
        )
        pipeline = GlobalMetricPipeline(config)
        pipeline.run()
        print(f"Global metrics completed: {config.input_dir / 'analysis_artifacts' / 'global_metrics'}")
        return

    if args.command == "render":
        base_config = RenderRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            figure_id="observations_and_creations",
            step=args.step,
            refresh=bool(args.refresh),
        )
        pipeline = FigurePipeline(base_config)
        if args.figure.strip().lower() == "all":
            figure_ids = _whole_step_figure_ids(pipeline)
        else:
            selected_figure_ids = _parse_figures(args.figure, pipeline.registry.list_ids())
            stepwise_figure_ids = [
                figure_id
                for figure_id in selected_figure_ids
                if pipeline.registry.get(figure_id).category == "stepwise"
            ]
            snapshot_figure_ids = [
                figure_id
                for figure_id in selected_figure_ids
                if pipeline.registry.get(figure_id).category == "snapshot"
            ]
            if stepwise_figure_ids:
                raise ValueError(
                    "stepwise figure(s) must be rendered with 'render-frames': "
                    + ", ".join(stepwise_figure_ids)
                )
            if snapshot_figure_ids:
                raise ValueError(
                    "snapshot figure(s) must be rendered with 'render-snapshot': "
                    + ", ".join(snapshot_figure_ids)
                )
            figure_ids = [
                figure_id
                for figure_id in selected_figure_ids
                if pipeline.registry.get(figure_id).category == "whole_step"
            ]

        print(f"Figures ({len(figure_ids)}): {', '.join(figure_ids)}")
        rendered, missing_by_figure, error_by_figure = FigureBatchRunner.render_many(
            base_config=base_config,
            figure_ids=figure_ids,
            num_workers=args.num_workers,
        )

        for figure_id, path in rendered.items():
            print(f"Rendered figure [{figure_id}]: {path}")

        if missing_by_figure:
            print("Missing artifacts by figure:")
            for figure_id in sorted(missing_by_figure.keys()):
                print(f"- {figure_id}")
                for item in missing_by_figure[figure_id]:
                    print(f"  - {item}")

        if error_by_figure:
            print("Render errors by figure:")
            for figure_id in sorted(error_by_figure.keys()):
                print(f"- {figure_id}: {error_by_figure[figure_id]}")

        if missing_by_figure or error_by_figure:
            raise SystemExit(2)
        return

    if args.command == "render-snapshot":
        steps = _parse_steps(args.steps)
        if steps is None:
            raise ValueError("render-snapshot requires an explicit --steps list")

        base_config = RenderRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            figure_id="observations_agents_clustered_panel",
            steps=steps,
            refresh=bool(args.refresh),
        )
        pipeline = FigurePipeline(base_config)
        available = _snapshot_figure_ids(pipeline)
        if args.figure.strip().lower() == "all":
            figure_ids = available
        else:
            figure_ids = _parse_figures(args.figure, available)

        print(f"Figures ({len(figure_ids)}): {', '.join(figure_ids)}")
        rendered, missing_by_figure, error_by_figure = FigureBatchRunner.render_many(
            base_config=base_config,
            figure_ids=figure_ids,
            num_workers=args.num_workers,
        )

        for figure_id, path in rendered.items():
            print(f"Rendered snapshot [{figure_id}]: {path}")

        if missing_by_figure:
            print("Missing artifacts by figure:")
            for figure_id in sorted(missing_by_figure.keys()):
                print(f"- {figure_id}")
                for item in missing_by_figure[figure_id]:
                    print(f"  - {item}")

        if error_by_figure:
            print("Render errors by figure:")
            for figure_id in sorted(error_by_figure.keys()):
                print(f"- {figure_id}: {error_by_figure[figure_id]}")

        if missing_by_figure or error_by_figure:
            raise SystemExit(2)
        return

    if args.command == "render-frames":
        steps = _parse_steps(args.steps)
        base_config = RenderRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            figure_id="observations_and_creations",
            refresh=bool(args.refresh),
        )
        pipeline = FigurePipeline(base_config)
        if args.figure.strip().lower() == "all":
            figure_ids = _stepwise_figure_ids(pipeline)
        else:
            selected_figure_ids = _parse_figures(args.figure, pipeline.registry.list_ids())
            whole_step_figure_ids = [
                figure_id
                for figure_id in selected_figure_ids
                if pipeline.registry.get(figure_id).category == "whole_step"
            ]
            snapshot_figure_ids = [
                figure_id
                for figure_id in selected_figure_ids
                if pipeline.registry.get(figure_id).category == "snapshot"
            ]
            if whole_step_figure_ids:
                raise ValueError(
                    "whole-step figure(s) must be rendered with 'render': "
                    + ", ".join(whole_step_figure_ids)
                )
            if snapshot_figure_ids:
                raise ValueError(
                    "snapshot figure(s) must be rendered with 'render-snapshot': "
                    + ", ".join(snapshot_figure_ids)
                )
            figure_ids = [
                figure_id
                for figure_id in selected_figure_ids
                if pipeline.registry.get(figure_id).category == "stepwise"
            ]

        print(f"Figures ({len(figure_ids)}): {', '.join(figure_ids)}")
        rendered, missing_by_figure, error_by_figure = FigureBatchRunner.render_frames_many(
            input_dir=base_config.input_dir,
            figure_ids=figure_ids,
            steps=steps,
            refresh=base_config.refresh,
            num_workers=args.num_workers,
        )

        for figure_id, count in rendered.items():
            print(f"Rendered {count} frames under {base_config.input_dir / 'analysis_figures' / figure_id}")

        if missing_by_figure:
            print("Missing artifacts by figure:")
            for figure_id in sorted(missing_by_figure.keys()):
                print(f"- {figure_id}")
                for item in missing_by_figure[figure_id]:
                    print(f"  - {item}")

        if error_by_figure:
            print("Render errors by figure:")
            for figure_id in sorted(error_by_figure.keys()):
                print(f"- {figure_id}: {error_by_figure[figure_id]}")

        if missing_by_figure or error_by_figure:
            raise SystemExit(2)
        return

    if args.command == "render-video":
        steps = _parse_steps(args.steps)
        base_config = VideoRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            steps=steps,
            refresh=bool(args.refresh),
            fps=max(1, int(args.fps)),
            num_workers=args.num_workers,
        )
        pipeline = VideoPipeline(base_config)
        if args.figure.strip().lower() == "all":
            figure_ids = pipeline.list_stepwise_figure_ids()
        else:
            figure_ids = _parse_figures(args.figure, pipeline.list_stepwise_figure_ids())

        print(f"Figures ({len(figure_ids)}): {', '.join(figure_ids)}")
        rendered, skipped, missing_by_figure, error_by_figure = VideoBatchRunner.render_many(
            base_config=base_config,
            figure_ids=figure_ids,
        )

        for figure_id, path in rendered.items():
            print(f"Rendered video [{figure_id}]: {path}")

        for figure_id, path in skipped.items():
            print(f"Skipped video [{figure_id}] (already exists): {path}")

        if missing_by_figure:
            print("Missing frames by figure:")
            for figure_id in sorted(missing_by_figure.keys()):
                print(f"- {figure_id}")
                for item in missing_by_figure[figure_id]:
                    print(f"  - {item}")

        if error_by_figure:
            print("Video render errors by figure:")
            for figure_id in sorted(error_by_figure.keys()):
                print(f"- {figure_id}: {error_by_figure[figure_id]}")

        if missing_by_figure or error_by_figure:
            raise SystemExit(2)
        return

    if args.command == "render-seeds":
        input_dirs = [_normalize_cli_path(path) for path in args.input_dirs]
        if len(input_dirs) < 2:
            raise ValueError("--input-dirs requires at least 2 directories")
        output_dir = _normalize_cli_path(args.output_dir)

        probe_config = RenderSeedsRunConfig(
            input_dirs=input_dirs,
            output_dir=output_dir,
            figure_ids=[],
            refresh=bool(args.refresh),
            num_workers=max(1, int(args.num_workers)),
        )
        probe_pipeline = MultiSeedFigurePipeline(probe_config)
        available = probe_pipeline.list_ids()

        if not available:
            print("No multi-seed figures available. Ensure rsa_within_clusters step metrics exist in all input dirs.")
            return

        if args.figure.strip().lower() == "all":
            figure_ids = available
        else:
            figure_ids = _parse_figures(args.figure, available)

        config = RenderSeedsRunConfig(
            input_dirs=input_dirs,
            output_dir=output_dir,
            figure_ids=figure_ids,
            refresh=bool(args.refresh),
            num_workers=max(1, int(args.num_workers)),
        )
        print(f"Figures ({len(figure_ids)}): {', '.join(figure_ids)}")
        pipeline = MultiSeedFigurePipeline(config)
        rendered, error_by_figure = pipeline.render_many(figure_ids, num_workers=config.num_workers)

        for figure_id, path in rendered.items():
            print(f"Rendered [{figure_id}]: {path}")

        if error_by_figure:
            print("Errors:")
            for figure_id, msg in sorted(error_by_figure.items()):
                print(f"- {figure_id}: {msg}")
            raise SystemExit(2)
        return

    if args.command == "render-conditions":
        import matplotlib.pyplot as plt
        from analysis.visualize import figures_rsa

        output_dir = _normalize_cli_path(args.output_dir)
        conditions_data: List[Dict[str, Any]] = []
        for cond_str in args.conditions:
            label, _, paths_str = cond_str.partition(":")
            seed_dirs = [_normalize_cli_path(p.strip()) for p in paths_str.split(",") if p.strip()]
            if not seed_dirs:
                raise ValueError(f"No directories found in condition spec: {cond_str!r}")
            probe_config = RenderSeedsRunConfig(
                input_dirs=seed_dirs,
                output_dir=output_dir,
                figure_ids=[],
                refresh=False,
            )
            pipeline = MultiSeedFigurePipeline(probe_config)
            try:
                data = pipeline._load_rsa_seeds()
            except Exception as exc:
                print(f"Error loading condition {label!r}: {exc}")
                raise SystemExit(2)
            data["label"] = label.strip()
            conditions_data.append(data)

        if not conditions_data:
            print("No condition data loaded.")
            raise SystemExit(2)

        figure_id = "rsa_clusters_conditions"
        out_dir = output_dir / "analysis_figures" / figure_id
        out_path = out_dir / "figure.png"
        if out_path.exists() and not args.refresh:
            print(f"Skipped (already exists): {out_path}")
            return

        fig = figures_rsa.render_rsa_clusters_conditions(conditions_data)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            p = out_dir / f"figure.{ext}"
            fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
            print(f"Saved: {p}")
        plt.close(fig)
        return

    if args.command == "status":
        config = RenderRunConfig(
            input_dir=_normalize_cli_path(args.input_dir),
            figure_id="observations_and_creations",
        )
        pipeline = FigurePipeline(config)
        status = pipeline.status()
        print(f"Figures dir: {status['figures_dir']}")
        print("Available figure IDs by category:")
        for category in ("stepwise", "whole_step", "snapshot"):
            figure_ids = status["figures_by_category"].get(category, [])
            if not figure_ids:
                continue
            print(f"- {category}")
            for figure_id in figure_ids:
                print(f"  - {figure_id}")
        return

    raise NotImplementedError(f"Command not implemented yet: {args.command}")


if __name__ == "__main__":
    main()
