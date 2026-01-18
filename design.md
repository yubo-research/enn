# Design 20260118


## Proposed Config Hierarchy, semantically

Search space & sampling
  candidate_rv (SOBOL, UNIFORM)
  num_candidates

Initialization policy
  init_strategy, num_init

Surrogate model
  k
  ENN fitting
    num_fit_samples, num_fit_candidates
  scale_x

Acquisition / decision rule
  AcqType (THOMPSON, PARETO, UCB)

Trust region / locality
  TR length schedule
    length_init, length_min, length_max
  multi-objective shaping
    num_metrics, alpha
  rescalarize (ON_RESTART, ON_PROPOSE)

Observation handling
  trailing_obs


## Proposd Config Hierarchy

Search space & sampling
  CandidateSamplingConfig
    candidate_rv
    num_candidates

Initialization policy
  InitPolicyConfig
    init_strategy
    num_init

Surrogate model
  SurrogateConfig
    k
    ENNFitConfig
      num_fit_samples
      num_fit_candidates
    scale_x

Acquisition / decision rule
  AcquisitionPolicyConfig
    AcqType

Trust region / locality
  TrustRegionConfig
    TRLengthConfig
      length_init
      length_min
      length_max
    MultiObjectiveConfig
      num_metrics
      alpha
    RescalePolicyConfig
      rescalarize

Observation handling
  ObservationHistoryConfig
    trailing_obs
