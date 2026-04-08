"""Dataclass-based configuration schemas and parsers for mpdiff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class EigenDistributionConfig:
    """Distribution parameters for eigenvalue/diagonal sampling."""

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
    beta_scale: float = 1.0
    beta_shift: float = 0.0


@dataclass(slots=True)
class DriftConfig:
    """Configurable drift models used in Euler diffusion simulation.

    Supported ``kind`` values:
    - ``zero``: no drift.
    - ``constant``: constant vector drift from ``vector``.
    - ``linear_mean_reversion``: ``b(t, x) = -theta * (x - target)``.
    - ``time_sine``: ``b(t, x) = amplitude * sin(2π frequency t + phase) * direction``.
    - ``callable``: import callable from ``callable_path`` as ``module:function``.
    """

    kind: str = "zero"
    vector: list[float] = field(default_factory=list)
    theta: float = 0.0
    target: float | list[float] = 0.0
    amplitude: float = 0.0
    frequency: float = 1.0
    phase: float = 0.0
    direction: list[float] = field(default_factory=list)
    callable_path: str | None = None
    callable_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrthogonalConfig:
    """Configuration for orthogonal matrix generation."""

    method: str = "haar"  # haar | identity


@dataclass(slots=True)
class LowRankFactorConfig:
    """Configuration for generating low-rank factor matrix ``B``."""

    method: str = "gaussian"  # gaussian | identity_block | from_file
    scale: float = 1.0
    normalize_columns: bool = True
    matrix_path: str | None = None


@dataclass(slots=True)
class DiagonalNoiseConfig:
    """Configuration for diagonal matrix ``D`` in ``B Σ B^T + D``."""

    kind: str = "scalar_identity"  # scalar_identity | distribution
    scalar: float = 0.1
    distribution: EigenDistributionConfig = field(
        default_factory=lambda: EigenDistributionConfig(kind="dirac", value=0.1)
    )


@dataclass(slots=True)
class LowRankConfig:
    """Parameters for ``B Σ B^T + D`` covariance model."""

    rank: int = 5
    latent_eigen_distribution: EigenDistributionConfig = field(
        default_factory=lambda: EigenDistributionConfig(kind="dirac", value=1.0)
    )
    factor: LowRankFactorConfig = field(default_factory=LowRankFactorConfig)
    diagonal_noise: DiagonalNoiseConfig = field(default_factory=DiagonalNoiseConfig)


@dataclass(slots=True)
class CovarianceModelConfig:
    """Covariance model definition for volatility segments."""

    kind: str = "diag_scalar"  # diag_scalar | diag_distribution | orthogonal_diag | low_rank_plus_diag
    scalar: float = 1.0
    eigen_distribution: EigenDistributionConfig = field(default_factory=EigenDistributionConfig)
    orthogonal: OrthogonalConfig = field(default_factory=OrthogonalConfig)
    low_rank: LowRankConfig = field(default_factory=LowRankConfig)
    sampling_policy: str = "draw_once"  # draw_once | redraw_per_segment
    jitter: float = 1e-10


@dataclass(slots=True)
class PiecewiseSegmentConfig:
    """Single segment in a piecewise-constant volatility schedule."""

    start: float = 0.0
    end: float = 1.0
    scalar: float = 1.0
    model: CovarianceModelConfig | None = None
    name: str = ""


@dataclass(slots=True)
class ScaledBaseVolatilityConfig:
    """Settings for segment model ``Cov_j = c_j * M_j``."""

    base_model: CovarianceModelConfig = field(default_factory=CovarianceModelConfig)
    base_matrix_policy: str = "common_fixed"  # common_fixed | redraw_per_segment
    share_matrix_law_across_segments: bool = True


@dataclass(slots=True)
class VolatilityConfig:
    """Volatility specification over time."""

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
    drift: float | list[float] | None = 0.0  # legacy compatibility field
    drift_model: DriftConfig = field(default_factory=DriftConfig)
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
class InverseFixedPointMethodConfig:
    """Method-specific options for fixed-point inverse."""

    smoothing_strength: float = 0.08
    min_kernel_density: float = 1e-16


@dataclass(slots=True)
class InverseOptimizationMethodConfig:
    """Method-specific options for optimization inverse."""

    optimizer: str = "L-BFGS-B"
    max_iter: int = 120


@dataclass(slots=True)
class InverseStieltjesMethodConfig:
    """Method-specific options for Stieltjes-based inverse."""

    quantile_min: float = 0.02
    quantile_max: float = 0.98


@dataclass(slots=True)
class InverseMomentMethodConfig:
    """Method-specific options for moment-based inverse."""

    family: str = "gamma"
    min_variance: float = 1e-10


@dataclass(slots=True)
class MPInverseConfig:
    """Numerical parameters for MP inverse procedures."""

    method: str = "optimization"  # optimization | fixed_point | stieltjes_based | moment_based
    compare_all_methods: bool = False
    compare_methods: list[str] = field(default_factory=list)
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
    fixed_point: InverseFixedPointMethodConfig = field(default_factory=InverseFixedPointMethodConfig)
    optimization: InverseOptimizationMethodConfig = field(default_factory=InverseOptimizationMethodConfig)
    stieltjes_based: InverseStieltjesMethodConfig = field(default_factory=InverseStieltjesMethodConfig)
    moment_based: InverseMomentMethodConfig = field(default_factory=InverseMomentMethodConfig)


@dataclass(slots=True)
class PlottingConfig:
    """Plotting options for simulation and spectral scripts."""

    style: str = "default"
    figsize: tuple[float, float] = (8.0, 5.0)
    dpi: int = 120
    show: bool = True
    plot_paths: bool = True
    max_path_dims: int = 5
    plot_eigen_hist: bool = True
    eigen_hist_bins: int = 45
    plot_eigen_density: bool = True
    density_bandwidth: float | None = None


@dataclass(slots=True)
class BenchmarkConfig:
    """Benchmarking and runtime reporting switches."""

    enabled: bool = True


@dataclass(slots=True)
class PopulationSpectrumConfig:
    """Optional direct spectral-law input for MP forward/inverse experiments."""

    source: str = "from_covariance_model"  # from_covariance_model | parametric | atomic | grid | empirical
    kind: str = "dirac"  # for source=parametric
    params: dict[str, Any] = field(default_factory=dict)
    atoms: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    grid: list[float] = field(default_factory=list)
    density: list[float] = field(default_factory=list)
    eigenvalues: list[float] = field(default_factory=list)
    eigenvalues_path: str | None = None
    n_atoms: int = 400


@dataclass(slots=True)
class AnalysisConfig:
    """Optional analysis-specific settings."""

    population_model: CovarianceModelConfig | None = None
    population_spectrum: PopulationSpectrumConfig | None = None
    reference_segment_index: int = 0
    realized_covariance_normalization: str = "total_time"  # total_time | n_steps | n_steps_minus_one | none
    empirical_density_bandwidth: float | None = None
    empirical_histogram_bins: int = 50


@dataclass(slots=True)
class GlobalConfig:
    """Top-level runtime settings."""

    seed: int | None = 1234
    output_dir: str = "outputs"
    log_level: str = "INFO"
    save_figures: bool = True
    save_arrays: bool = True
    save_metadata: bool = True


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


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _parse_eigen_distribution(data: Mapping[str, Any] | None) -> EigenDistributionConfig:
    data = dict(_mapping(data))
    kind = str(data.get("kind", "dirac"))

    # Backward compatibility for rescaled beta parameter names.
    beta_scale = float(data.get("beta_scale", data.get("a", data.get("rescale", 1.0))))
    if kind == "rescaled_beta" and "scale" in data and "beta_scale" not in data and "a" not in data:
        beta_scale = float(data["scale"])
    beta_shift = float(data.get("beta_shift", data.get("b", data.get("shift", 0.0))))

    return EigenDistributionConfig(
        kind=kind,
        value=float(data.get("value", 1.0)),
        values=_float_list(data.get("values")),
        weights=_float_list(data.get("weights")),
        low=float(data.get("low", 0.5)),
        high=float(data.get("high", 1.5)),
        shape=float(data.get("shape", 2.0)),
        scale=float(data.get("scale", 1.0)),
        alpha=float(data.get("alpha", 2.0)),
        beta=float(data.get("beta", 5.0)),
        beta_scale=beta_scale,
        beta_shift=beta_shift,
    )


def _parse_drift_model(data: Mapping[str, Any] | None, legacy_drift: Any) -> DriftConfig:
    drift_data = dict(_mapping(data))
    if drift_data:
        return DriftConfig(
            kind=str(drift_data.get("kind", "zero")),
            vector=_float_list(drift_data.get("vector")),
            theta=float(drift_data.get("theta", 0.0)),
            target=(
                float(drift_data.get("target", 0.0))
                if isinstance(drift_data.get("target", 0.0), (int, float))
                else _float_list(drift_data.get("target"))
            ),
            amplitude=float(drift_data.get("amplitude", 0.0)),
            frequency=float(drift_data.get("frequency", 1.0)),
            phase=float(drift_data.get("phase", 0.0)),
            direction=_float_list(drift_data.get("direction")),
            callable_path=str(drift_data["callable_path"]) if drift_data.get("callable_path") is not None else None,
            callable_kwargs=dict(_mapping(drift_data.get("callable_kwargs"))),
        )

    # Legacy mode: infer from scalar/list-valued simulation.drift.
    if legacy_drift is None:
        return DriftConfig(kind="zero")
    if isinstance(legacy_drift, (int, float)):
        if abs(float(legacy_drift)) < 1e-15:
            return DriftConfig(kind="zero")
        return DriftConfig(kind="constant", vector=[float(legacy_drift)])
    vector = _float_list(legacy_drift)
    if all(abs(v) < 1e-15 for v in vector):
        return DriftConfig(kind="zero")
    return DriftConfig(kind="constant", vector=vector)


def _parse_orthogonal(data: Mapping[str, Any] | None) -> OrthogonalConfig:
    data = dict(_mapping(data))
    return OrthogonalConfig(method=str(data.get("method", "haar")))


def _parse_factor(data: Mapping[str, Any] | None, legacy_factor_scale: float = 1.0) -> LowRankFactorConfig:
    data = dict(_mapping(data))
    return LowRankFactorConfig(
        method=str(data.get("method", data.get("kind", "gaussian"))),
        scale=float(data.get("scale", data.get("factor_scale", legacy_factor_scale))),
        normalize_columns=bool(data.get("normalize_columns", True)),
        matrix_path=str(data["matrix_path"]) if data.get("matrix_path") is not None else None,
    )


def _parse_diagonal_noise(data: Mapping[str, Any] | None, legacy_diag: Mapping[str, Any] | None) -> DiagonalNoiseConfig:
    data_dict = dict(_mapping(data))
    if data_dict:
        kind = str(data_dict.get("kind", "scalar_identity"))
        return DiagonalNoiseConfig(
            kind=kind,
            scalar=float(data_dict.get("scalar", 0.1)),
            distribution=_parse_eigen_distribution(data_dict.get("distribution")),
        )

    legacy_diag = dict(_mapping(legacy_diag))
    if legacy_diag:
        return DiagonalNoiseConfig(
            kind="distribution",
            distribution=_parse_eigen_distribution(legacy_diag),
        )

    return DiagonalNoiseConfig()


def _parse_low_rank(data: Mapping[str, Any] | None) -> LowRankConfig:
    data = dict(_mapping(data))
    factor_scale_legacy = float(data.get("factor_scale", 1.0))

    return LowRankConfig(
        rank=int(data.get("rank", 5)),
        latent_eigen_distribution=_parse_eigen_distribution(
            data.get("latent_eigen_distribution", data.get("sigma_distribution"))
        ),
        factor=_parse_factor(data.get("factor"), legacy_factor_scale=factor_scale_legacy),
        diagonal_noise=_parse_diagonal_noise(
            data.get("diagonal_noise"),
            data.get("diag_eigen_distribution"),
        ),
    )


def _normalize_covariance_kind(kind: str) -> str:
    aliases = {
        "diagonal_isotropic": "diag_scalar",
        "isotropic_diag": "diag_scalar",
        "rotated_diag": "orthogonal_diag",
    }
    return aliases.get(kind, kind)


def _parse_covariance_model(data: Mapping[str, Any] | None) -> CovarianceModelConfig:
    data = dict(_mapping(data))
    kind = _normalize_covariance_kind(str(data.get("kind", "diag_scalar")))
    return CovarianceModelConfig(
        kind=kind,
        scalar=float(data.get("scalar", 1.0)),
        eigen_distribution=_parse_eigen_distribution(data.get("eigen_distribution")),
        orthogonal=_parse_orthogonal(data.get("orthogonal")),
        low_rank=_parse_low_rank(data.get("low_rank")),
        sampling_policy=str(data.get("sampling_policy", data.get("draw_mode", "draw_once"))),
        jitter=float(data.get("jitter", 1e-10)),
    )


def _parse_segment(data: Mapping[str, Any], default_name: str) -> PiecewiseSegmentConfig:
    data = dict(_mapping(data))
    return PiecewiseSegmentConfig(
        start=float(data["start"]),
        end=float(data["end"]),
        scalar=float(data.get("scalar", 1.0)),
        model=_parse_covariance_model(data.get("model")) if data.get("model") is not None else None,
        name=str(data.get("name", default_name)),
    )


def _parse_scaled_base(data: Mapping[str, Any] | None) -> ScaledBaseVolatilityConfig:
    data = dict(_mapping(data))
    base_matrix_policy = str(data.get("base_matrix_policy", "common_fixed"))
    if base_matrix_policy == "fixed_once":
        base_matrix_policy = "common_fixed"
    return ScaledBaseVolatilityConfig(
        base_model=_parse_covariance_model(data.get("base_model")),
        base_matrix_policy=base_matrix_policy,
        share_matrix_law_across_segments=bool(data.get("share_matrix_law_across_segments", True)),
    )


def _parse_volatility(data: Mapping[str, Any]) -> VolatilityConfig:
    data = dict(_mapping(data))
    mode = str(data.get("mode", "constant"))
    if mode == "scaled_base":
        mode = "piecewise_scaled_base"

    segments = [
        _parse_segment(segment_data, default_name=f"segment_{idx}")
        for idx, segment_data in enumerate(data.get("segments", []))
    ]

    return VolatilityConfig(
        mode=mode,
        constant_model=_parse_covariance_model(data.get("constant_model", data.get("model"))),
        segments=segments,
        scaled_base=_parse_scaled_base(data.get("scaled_base")),
    )


def _parse_inverse_fixed_point(data: Mapping[str, Any] | None) -> InverseFixedPointMethodConfig:
    data = dict(_mapping(data))
    return InverseFixedPointMethodConfig(
        smoothing_strength=float(data.get("smoothing_strength", 0.08)),
        min_kernel_density=float(data.get("min_kernel_density", 1e-16)),
    )


def _parse_inverse_optimization(data: Mapping[str, Any] | None) -> InverseOptimizationMethodConfig:
    data = dict(_mapping(data))
    return InverseOptimizationMethodConfig(
        optimizer=str(data.get("optimizer", "L-BFGS-B")),
        max_iter=int(data.get("max_iter", data.get("optimizer_max_iter", 120))),
    )


def _parse_inverse_stieltjes(data: Mapping[str, Any] | None) -> InverseStieltjesMethodConfig:
    data = dict(_mapping(data))
    return InverseStieltjesMethodConfig(
        quantile_min=float(data.get("quantile_min", 0.02)),
        quantile_max=float(data.get("quantile_max", 0.98)),
    )


def _parse_inverse_moment(data: Mapping[str, Any] | None) -> InverseMomentMethodConfig:
    data = dict(_mapping(data))
    return InverseMomentMethodConfig(
        family=str(data.get("family", "gamma")),
        min_variance=float(data.get("min_variance", 1e-10)),
    )


def _parse_population_spectrum(data: Mapping[str, Any] | None) -> PopulationSpectrumConfig:
    data = dict(_mapping(data))
    params = dict(_mapping(data.get("params")))

    # Allow user-friendly inline parameter declarations.
    for key in (
        "value",
        "values",
        "weights",
        "low",
        "high",
        "shape",
        "scale",
        "alpha",
        "beta",
        "shift",
        "beta_shift",
        "beta_scale",
    ):
        if key in data and key not in params:
            params[key] = data[key]

    return PopulationSpectrumConfig(
        source=str(data.get("source", "from_covariance_model")),
        kind=str(data.get("kind", "dirac")),
        params=params,
        atoms=_float_list(data.get("atoms")),
        weights=_float_list(data.get("weights")),
        grid=_float_list(data.get("grid")),
        density=_float_list(data.get("density")),
        eigenvalues=_float_list(data.get("eigenvalues")),
        eigenvalues_path=str(data["eigenvalues_path"]) if data.get("eigenvalues_path") is not None else None,
        n_atoms=int(data.get("n_atoms", 400)),
    )


def project_config_from_dict(data: Mapping[str, Any]) -> ProjectConfig:
    """Build :class:`ProjectConfig` from a dictionary representation."""
    global_data = dict(_mapping(data.get("global")))
    sim_data = dict(_mapping(data.get("simulation")))
    vol_data = dict(_mapping(data.get("volatility")))
    fwd_data = dict(_mapping(data.get("mp_forward")))
    inv_data = dict(_mapping(data.get("mp_inverse")))
    plot_data = dict(_mapping(data.get("plotting")))
    bench_data = dict(_mapping(data.get("benchmark")))
    analysis_data = dict(_mapping(data.get("analysis")))

    plotting_size = plot_data.get("figsize", [8.0, 5.0])
    legacy_drift = sim_data.get("drift", 0.0)
    drift_model = _parse_drift_model(sim_data.get("drift_model"), legacy_drift=legacy_drift)

    initial_state_raw = sim_data.get("initial_state", 0.0)
    initial_state = (
        float(initial_state_raw)
        if isinstance(initial_state_raw, (int, float))
        else _float_list(initial_state_raw)
    )

    compare_methods_raw = inv_data.get("compare_methods", [])
    if isinstance(compare_methods_raw, str):
        compare_methods = [compare_methods_raw]
    else:
        compare_methods = [str(m) for m in compare_methods_raw]

    return ProjectConfig(
        global_settings=GlobalConfig(
            seed=global_data.get("seed", 1234),
            output_dir=str(global_data.get("output_dir", "outputs")),
            log_level=str(global_data.get("log_level", "INFO")),
            save_figures=bool(global_data.get("save_figures", True)),
            save_arrays=bool(global_data.get("save_arrays", True)),
            save_metadata=bool(global_data.get("save_metadata", True)),
        ),
        simulation=SimulationConfig(
            d=int(sim_data.get("d", 100)),
            T=float(sim_data.get("T", 1.0)),
            n_steps=int(sim_data.get("n_steps", 1000)),
            drift=(
                float(legacy_drift)
                if isinstance(legacy_drift, (int, float))
                else _float_list(legacy_drift)
            )
            if legacy_drift is not None
            else None,
            drift_model=drift_model,
            initial_state=initial_state,
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
            compare_all_methods=bool(inv_data.get("compare_all_methods", False)),
            compare_methods=compare_methods,
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
            fixed_point=_parse_inverse_fixed_point(inv_data.get("fixed_point")),
            optimization=_parse_inverse_optimization(inv_data.get("optimization")),
            stieltjes_based=_parse_inverse_stieltjes(inv_data.get("stieltjes_based")),
            moment_based=_parse_inverse_moment(inv_data.get("moment_based")),
        ),
        plotting=PlottingConfig(
            style=str(plot_data.get("style", "default")),
            figsize=(float(plotting_size[0]), float(plotting_size[1])),
            dpi=int(plot_data.get("dpi", 120)),
            show=bool(plot_data.get("show", True)),
            plot_paths=bool(plot_data.get("plot_paths", True)),
            max_path_dims=int(plot_data.get("max_path_dims", 5)),
            plot_eigen_hist=bool(plot_data.get("plot_eigen_hist", True)),
            eigen_hist_bins=int(plot_data.get("eigen_hist_bins", 45)),
            plot_eigen_density=bool(plot_data.get("plot_eigen_density", True)),
            density_bandwidth=(
                float(plot_data["density_bandwidth"])
                if plot_data.get("density_bandwidth") is not None
                else None
            ),
        ),
        benchmark=BenchmarkConfig(enabled=bool(bench_data.get("enabled", True))),
        analysis=AnalysisConfig(
            population_model=_parse_covariance_model(analysis_data["population_model"])
            if analysis_data.get("population_model") is not None
            else None,
            population_spectrum=_parse_population_spectrum(analysis_data.get("population_spectrum"))
            if analysis_data.get("population_spectrum") is not None
            else None,
            reference_segment_index=int(analysis_data.get("reference_segment_index", 0)),
            realized_covariance_normalization=str(analysis_data.get("realized_covariance_normalization", "total_time")),
            empirical_density_bandwidth=(
                float(analysis_data["empirical_density_bandwidth"])
                if analysis_data.get("empirical_density_bandwidth") is not None
                else None
            ),
            empirical_histogram_bins=int(analysis_data.get("empirical_histogram_bins", 50)),
        ),
    )
