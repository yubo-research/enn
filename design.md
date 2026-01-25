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


----

# TuRBO-ENN is O(N)
N is the number of observations at the time of ask() or tell(). These call should be no worse than O(N). Maybe O(NlnK). Maybe even O(NK) if we have to.

Not O(NlnN) or O(N^2).  O(N).


## Incumbent: compute once, reuse everywhere

There is **exactly one incumbent** at any time, and it must be **selected once** and then **reused**:

- **Trust region length update**: determine success/failure (improvement) and update TR length.
- **Trust region center**: the incumbent `x_center` is the center of the trust region.
- **Trailing window retention**: the incumbent must be kept in the trailing window no matter how old it is.

There are not multiple “paths” or multiple “methods” for finding the incumbent. The incumbent is found **once** and used for these three purposes. This is efficient, maintainable, and sensible.
