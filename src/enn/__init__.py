from __future__ import annotations

from ._lazy import lazy_getattr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "SemanticOutputProjection": (".semantic_outputs", "SemanticOutputProjection"),
    "SemanticInputTransform": (".semantic", "SemanticInputTransform"),
    "EpistemicNearestNeighbors": (".enn.enn_class", "EpistemicNearestNeighbors"),
    "ENNStatefulFitter": (".enn.enn_fitter", "ENNStatefulFitter"),
    "ENNOutputBlend": ("._rust", "ENNOutputBlend"),
    "AffineCalibrator": (".enn.affine_calibrator", "AffineCalibrator"),
    "fit_affine_calibrator": (".enn.affine_calibrator", "fit_affine_calibrator"),
    "create_optimizer": (".turbo.rust_optimizer", "create_optimizer"),
    "Optimizer": (".turbo.rust_optimizer", "Optimizer"),
    "Telemetry": (".turbo.types.telemetry", "Telemetry"),
    "OptimizerConfig": (".turbo.config", "OptimizerConfig"),
    "turbo_zero_config": (".turbo.config", "turbo_zero_config"),
    "turbo_enn_config": (".turbo.config", "turbo_enn_config"),
    "lhd_only_config": (".turbo.config", "lhd_only_config"),
    "TurboTRConfig": (".turbo.config.trust_region", "TurboTRConfig"),
    "MorboTRConfig": (".turbo.config.trust_region", "MorboTRConfig"),
    "NoTRConfig": (".turbo.config.trust_region", "NoTRConfig"),
    "CandidateRV": (".turbo.config", "CandidateRV"),
    "InitStrategy": (".turbo.config", "InitStrategy"),
    "AcqType": (".turbo.config", "AcqType"),
    "CatEncoder": (".enn.cat_codes", "CatEncoder"),
    "CatEncoding": (".enn.cat_codes", "CatEncoding"),
    "DimKind": (".enn.cat_codes", "DimKind"),
    "ENNParams": (".enn.enn_params", "ENNParams"),
    "PosteriorFlags": (".enn.enn_params", "PosteriorFlags"),
    "ENNNormal": (".enn.enn_normal", "ENNNormal"),
    "ENNSurrogateConfig": (".turbo.config.surrogate", "ENNSurrogateConfig"),
    "ENNFitConfig": (".turbo.config.enn_fit_config", "ENNFitConfig"),
    "NoSurrogateConfig": (".turbo.config.surrogate", "NoSurrogateConfig"),
    "MultiObjectiveConfig": (
        ".turbo.config.multi_objective_config",
        "MultiObjectiveConfig",
    ),
    "CandidateGenConfig": (".turbo.config.candidate_gen_config", "CandidateGenConfig"),
    "TRLengthConfig": (".turbo.config.tr_length_config", "TRLengthConfig"),
    "Rescalarize": (".turbo.config.rescalarize", "Rescalarize"),
    "RAASPDriver": (".turbo.config.raasp_driver", "RAASPDriver"),
    "ENNMetricLearning": (".turbo.config.enn_x_scaling", "ENNMetricLearning"),
}


def __getattr__(name: str):
    return lazy_getattr(
        name=name,
        module_name=__name__,
        package=__package__,
        mapping=_LAZY_ATTRS,
        extra="`pip install 'ennbo[with-deps]'`",
    )


__all__: list[str] = list(_LAZY_ATTRS)
