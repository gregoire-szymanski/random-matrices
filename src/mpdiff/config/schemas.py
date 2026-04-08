"""Dataclass-based configuration schemas for mpdiff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class EigenDistributionConfig:
    """Eigenvalue distribution settings for diagonal covariance models."""

    kind: str = "dirac"
    value: float = 1.0
    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    low: float = 0.5
    high: float = 1.5
    shape: float = 2.0
    scale: float = 1.0
    alpha: float = 2.0
    beta: float = 5.0
    a: float = 1.0
    b: float = 0.0


@dataclass(slots=True)
class OrthogonalConfig:
    """Configuration for orthogonal matrix generation."""

    method: str = "haar"


@dataclass(slots=True)
class LowRankConfig:
    """Parameters for ``B Σ B^T + D`` low-rank covariance models."""

    rank: int = 5
    factor_scale: float = 1.0
    latent_eigen_distribution: EigenDistributionConfig = field(
        default_factory=lambda: EigenDistributionConfig(kind="dirac", value=1.0)
    )
    diag_eigen_distribution: EigenDistributionConfig = field(
        default_factory=lambda: EigenDistributionConfig(kind="dirac", value=0.1)
    )


@dataclass(slots=True)
class CovarianceModelConfig:
    """Covariance model definition for each volatility segment."""

    kind: str = "diag_scalar"
    scalar: float = 1.0
    eigen_distribution: EigenDistributionConfig = field(default_factory=EigenDistributionConfig)
    orthogonal: OrthogonalConfig = field(default_factory=OrthogonalConfig)
    low_rank: LowRankConfig = field(default_factory=LowRankConfig)
    jitter: float = 1e-10


@dataclass(slots=True)
class PiecewiseSegmentConfig:
    """Piecewise interval model and optional scalar multiplier."""

    start: float = 0.0
    end: float = 1.0
    scalar: float = 1.0
    model: CovarianceModelConfig | None = None


@dataclass(slots=True)
class ScaledBaseVolatilityConfig:
    """Settings for piecewise scalar-times-base covariance structure."""

    base_model: CovarianceModelConfig = field(default_factory=CovarianceModelConfig)
    base_matrix_policy: str = "fixed_once"  # fixed_once | redraw_per_segment
    share_matrix_law_across_segments: bool = True


@dataclass(slots=True)
class VolatilityConfig:
    """Volatility specification across time."""

    mode: str = "constant"  # constant | piecewise | piecewise_scaled_base
    constant_model: CovarianceModelConfig = field(default_factory=CovarianceModelConfig)
    segments: list[PiecewiseSegmentConfig] = field(default_factory=list)
    scaled_base: ScaledBaseVolatilityConfig = field(default_factory=ScaledBaseVolatilityConfig)


@dataclass(slots=True)
class SimulationConfig:
    """Diffusion simulation controls."""

    d: int = 100
    T: float = 1.0
    n_steps: int = 1000
    drift: float | list[float] = 0.0
    initial_state: float | list[float] = 0.0
    time_grid: list[float] | None = None


@dataclass(slots=True)
class MPForwardConfig:
    """Numerical parameters for MP forward transform."""

    aspect_ratio: float | None = None
    grid_min: float = 0.0
    grid_max: float = 4.0
    num_points: int = 512
    eta: float = 1e-3
    tol: float = 1e-9
    max_iter: int = 500
    damping: float = 0.7


@dataclass(slots=True)
class MPInverseConfig:
    """Numerical parameters for MP inverse procedures."""

    method: str = "optimization"  # optimization | fixed_point | stieltjes_based | moment_based
    n_support: int = 50
    support_min: float | None = None
    support_max: float | None = None
    eta: float = 2e-3
    tol: float = 1e-6
    max_iter: int = 300
    regularization: float = 1e-3
    optimizer_max_iter: int = 120
    forward_max_iter: int = 250
    forward_tol: float = 1e-7


@dataclass(slots=True)
class PlottingConfig:
    """Plotting options for experiment scripts."""

    style: str = "default"
    figsize: tuple[float, float] = (8.0, 5.0)
    dpi: int = 120
    show: bool = True


@dataclass(slots=True)
class BenchmarkConfig:
    """Benchmarking and runtime reporting switches."""

    enabled: bool = True


@dataclass(slots=True)
class AnalysisConfig:
    """Optional analysis-specific settings."""

    population_model: CovarianceModelConfig | None = None
    reference_segment_index: int = 0


@dataclass(slots=True)
class GlobalConfig:
    """Top-level runtime settings."""

    seed: int | None = 1234
    output_dir: str = "outputs"
    log_level: str = "INFO"
    save_figures: bool = True
    save_arrays: bool = True


@dataclass(slots=True)
class ProjectConfig:
    """Root config object for the whole project."""

    global_settings: GlobalConfig = field(default_factory=GlobalConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    mp_forward: MPForwardConfig = field(default_factory=MPForwardConfig)
    mp_inverse: MPInverseConfig = field(default_factory=MPInverseConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(v) for v in value]


def _parse_eigen_distribution(data: Mapping[str, Any] | None) -> EigenDistributionConfig:
    data = {} if data is None else dict(data)
    return EigenDistributionConfig(
        kind=str(data.get("kind", "dirac")),
        value=float(data.get("value", 1.0)),
        values=_float_list(data.get("values")),
        weights=_float_list(data.get("weights")),
        low=float(data.get("low", 0.5)),
        high=float(data.get("high", 1.5)),
        shape=float(data.get("shape", 2.0)),
        scale=float(data.get("scale", 1.0)),
        alpha=float(data.get("alpha", 2.0)),
        beta=float(data.get("beta", 5.0)),
        a=float(data.get("a", 1.0)),
        b=float(data.get("b", 0.0)),
    )


def _parse_low_rank(data: Mapping[str, Any] | None) -> LowRankConfig:
    data = {} if data is None else dict(data)
    return LowRankConfig(
        rank=int(data.get("rank", 5)),
        factor_scale=float(data.get("factor_scale", 1.0)),
        latent_eigen_distribution=_parse_eigen_distribution(data.get("latent_eigen_distribution")),
        diag_eigen_distribution=_parse_eigen_distribution(data.get("diag_eigen_distribution")),
    )


def _parse_covariance_model(data: Mapping[str, Any] | None) -> CovarianceModelConfig:
    data = {} if data is None else dict(data)
    orth = data.get("orthogonal", {})
    return CovarianceModelConfig(
        kind=str(data.get("kind", "diag_scalar")),
        scalar=float(data.get("scalar", 1.0)),
        eigen_distribution=_parse_eigen_distribution(data.get("eigen_distribution")),
        orthogonal=OrthogonalConfig(method=str(orth.get("method", "haar"))),
        low_rank=_parse_low_rank(data.get("low_rank")),
        jitter=float(data.get("jitter", 1e-10)),
    )


def _parse_segment(data: Mapping[str, Any]) -> PiecewiseSegmentConfig:
    return PiecewiseSegmentConfig(
        start=float(data["start"]),
        end=float(data["end"]),
        scalar=float(data.get("scalar", 1.0)),
        model=_parse_covariance_model(data.get("model")) if data.get("model") is not None else None,
    )


def _parse_volatility(data: Mapping[str, Any]) -> VolatilityConfig:
    mode = str(data.get("mode", "constant"))
    scaled_base_data = dict(data.get("scaled_base", {}))
    scaled_base = ScaledBaseVolatilityConfig(
        base_model=_parse_covariance_model(scaled_base_data.get("base_model")),
        base_matrix_policy=str(scaled_base_data.get("base_matrix_policy", "fixed_once")),
        share_matrix_law_across_segments=bool(scaled_base_data.get("share_matrix_law_across_segments", True)),
    )
    segments = [_parse_segment(seg) for seg in data.get("segments", [])]
    return VolatilityConfig(
        mode=mode,
        constant_model=_parse_covariance_model(data.get("constant_model")),
        segments=segments,
        scaled_base=scaled_base,
    )


def project_config_from_dict(data: Mapping[str, Any]) -> ProjectConfig:
    """Build :class:`ProjectConfig` from a dictionary representation."""
    global_data = dict(data.get("global", {}))
    sim_data = dict(data.get("simulation", {}))
    vol_data = dict(data.get("volatility", {}))
    fwd_data = dict(data.get("mp_forward", {}))
    inv_data = dict(data.get("mp_inverse", {}))
    plot_data = dict(data.get("plotting", {}))
    bench_data = dict(data.get("benchmark", {}))
    analysis_data = dict(data.get("analysis", {}))

    plotting_size = plot_data.get("figsize", [8.0, 5.0])
    return ProjectConfig(
        global_settings=GlobalConfig(
            seed=global_data.get("seed", 1234),
            output_dir=str(global_data.get("output_dir", "outputs")),
            log_level=str(global_data.get("log_level", "INFO")),
            save_figures=bool(global_data.get("save_figures", True)),
            save_arrays=bool(global_data.get("save_arrays", True)),
        ),
        simulation=SimulationConfig(
            d=int(sim_data.get("d", 100)),
            T=float(sim_data.get("T", 1.0)),
            n_steps=int(sim_data.get("n_steps", 1000)),
            drift=float(sim_data["drift"]) if isinstance(sim_data.get("drift", 0.0), (int, float)) else _float_list(sim_data.get("drift")),
            initial_state=float(sim_data["initial_state"])
            if isinstance(sim_data.get("initial_state", 0.0), (int, float))
            else _float_list(sim_data.get("initial_state")),
            time_grid=_float_list(sim_data.get("time_grid")) if sim_data.get("time_grid") is not None else None,
        ),
        volatility=_parse_volatility(vol_data),
        mp_forward=MPForwardConfig(
            aspect_ratio=float(fwd_data["aspect_ratio"]) if fwd_data.get("aspect_ratio") is not None else None,
            grid_min=float(fwd_data.get("grid_min", 0.0)),
            grid_max=float(fwd_data.get("grid_max", 4.0)),
            num_points=int(fwd_data.get("num_points", 512)),
            eta=float(fwd_data.get("eta", 1e-3)),
            tol=float(fwd_data.get("tol", 1e-9)),
            max_iter=int(fwd_data.get("max_iter", 500)),
            damping=float(fwd_data.get("damping", 0.7)),
        ),
        mp_inverse=MPInverseConfig(
            method=str(inv_data.get("method", "optimization")),
            n_support=int(inv_data.get("n_support", 50)),
            support_min=float(inv_data["support_min"]) if inv_data.get("support_min") is not None else None,
            support_max=float(inv_data["support_max"]) if inv_data.get("support_max") is not None else None,
            eta=float(inv_data.get("eta", 2e-3)),
            tol=float(inv_data.get("tol", 1e-6)),
            max_iter=int(inv_data.get("max_iter", 300)),
            regularization=float(inv_data.get("regularization", 1e-3)),
            optimizer_max_iter=int(inv_data.get("optimizer_max_iter", 120)),
            forward_max_iter=int(inv_data.get("forward_max_iter", 250)),
            forward_tol=float(inv_data.get("forward_tol", 1e-7)),
        ),
        plotting=PlottingConfig(
            style=str(plot_data.get("style", "default")),
            figsize=(float(plotting_size[0]), float(plotting_size[1])),
            dpi=int(plot_data.get("dpi", 120)),
            show=bool(plot_data.get("show", True)),
        ),
        benchmark=BenchmarkConfig(enabled=bool(bench_data.get("enabled", True))),
        analysis=AnalysisConfig(
            population_model=_parse_covariance_model(analysis_data["population_model"])
            if analysis_data.get("population_model") is not None
            else None,
            reference_segment_index=int(analysis_data.get("reference_segment_index", 0)),
        ),
    )
