"""Runner for side-by-side MP inversion benchmarking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_density_comparison
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng

from .common import build_population_spectrum, ensure_output_dir, log_summary, resolve_aspect_ratio, save_density
from .inversion_benchmark import benchmark_inverse_methods_from_population


def run_inversion_benchmark(config_path: str | Path) -> dict[str, Any]:
    """Benchmark all/selected inversion methods on one config-driven benchmark law."""
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_inversion_benchmark")
    out_dir = ensure_output_dir(cfg)

    rng = make_rng(cfg.global_settings.seed)
    population = build_population_spectrum(cfg, rng)
    aspect_ratio = resolve_aspect_ratio(cfg)
    grid = make_linear_grid(cfg.mp_forward.grid_min, cfg.mp_forward.grid_max, cfg.mp_forward.num_points)

    benchmark = benchmark_inverse_methods_from_population(
        population=population,
        aspect_ratio=aspect_ratio,
        grid=grid,
        forward_settings=cfg.mp_forward,
        inverse_settings=cfg.mp_inverse,
        methods=None,
        density_bandwidth=cfg.analysis.empirical_density_bandwidth or cfg.plotting.density_bandwidth,
    )

    summary_df = benchmark.summary_table
    if summary_df.empty:
        raise RuntimeError("No benchmark results were produced")

    if cfg.global_settings.save_arrays or cfg.global_settings.save_metadata:
        summary_df.to_csv(out_dir / "inversion_benchmark_summary.csv", index=False)
        summary_df.to_json(out_dir / "inversion_benchmark_summary.json", orient="records", indent=2)

    if cfg.global_settings.save_arrays:
        observed = benchmark.observed_density
        save_density(out_dir / "inversion_benchmark_observed_density.npz", observed)

        for method in benchmark.methods:
            result_bundle = benchmark.method_results[method]
            inversion = result_bundle["inversion"]
            estimated_density = result_bundle["estimated_density"]

            save_density(out_dir / f"inversion_benchmark_reconstructed_density_{method}.npz", inversion.reconstructed_observed)
            save_density(out_dir / f"inversion_benchmark_estimated_density_{method}.npz", estimated_density)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)

        observed = benchmark.observed_density
        methods = summary_df["method"].tolist()

        fig, _ = plot_density_comparison(
            densities=[observed] + [benchmark.method_results[m]["inversion"].reconstructed_observed for m in methods],
            labels=["observed"] + [f"reconstructed ({m})" for m in methods],
            title="Inversion Benchmark: Observed vs Reconstructed",
            figsize=cfg.plotting.figsize,
        )
        fig.savefig(out_dir / "inversion_benchmark_reconstruction.png", dpi=cfg.plotting.dpi)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=cfg.plotting.figsize)
        ax2.bar(summary_df["method"], summary_df["runtime_seconds"])
        ax2.set_ylabel("runtime (s)")
        ax2.set_title("Inversion Method Runtime")
        ax2.grid(alpha=0.2)
        fig2.tight_layout()
        fig2.savefig(out_dir / "inversion_benchmark_runtime.png", dpi=cfg.plotting.dpi)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=cfg.plotting.figsize)
        ax3.bar(summary_df["method"], summary_df["population_wasserstein_1"])
        ax3.set_ylabel("Wasserstein-1")
        ax3.set_title("Population Recovery Quality")
        ax3.grid(alpha=0.2)
        fig3.tight_layout()
        fig3.savefig(out_dir / "inversion_benchmark_quality.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig3)

    top = summary_df.iloc[0]
    summary = {
        "aspect_ratio": float(aspect_ratio),
        "best_method": str(top["method"]),
        "best_population_wasserstein_1": float(top["population_wasserstein_1"]),
        "best_runtime_seconds": float(top["runtime_seconds"]),
        "n_methods": int(summary_df.shape[0]),
    }

    if cfg.global_settings.save_metadata:
        with (out_dir / "inversion_benchmark_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config_path": str(config_path),
                    "summary": summary,
                    "methods": benchmark.methods,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    log_summary(logger, "Inversion benchmark summary", summary)
    return summary
