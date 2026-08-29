//! Process-wide BPANN tuning values.
//!
//! Defaults match the historical hardcoded defaults. When the `ennbo` crate is
//! used, it installs a provider that reads `~/.ennbo/config.toml`.

use std::sync::RwLock;

/// Minimum allowed `index_compact_fragment_max` (matches compaction clamp floor).
pub const INDEX_COMPACT_FRAGMENT_MAX_MIN: usize = 3;

/// Default rows of pending observations before an index flush is scheduled.
pub const DEFAULT_PENDING_FLUSH_THRESHOLD: usize = 250;

/// Default hard cap on pending rows before soft sync runs on the calling thread.
///
/// Equal to `4 × DEFAULT_PENDING_FLUSH_THRESHOLD`. Must stay `>=` the soft threshold.
pub const DEFAULT_PENDING_HARD_FLUSH_THRESHOLD: usize = 3000;

/// Default max indexed rows for exhaustive leaf search (and no skip edges at build).
pub const DEFAULT_EXHAUSTIVE_SEARCH_ROW_LIMIT: usize = 2500;

/// Default max indexed rows for skip-refinement search (and skip-edge build).
pub const DEFAULT_SKIP_REFINEMENT_ROW_LIMIT: usize = 150_000;

/// Default max batch size for row-id-only leaf builds (mmap score path).
///
/// Matches the historical hardcoded cutoff in `build_batch` on main. Batches
/// larger than this use full in-memory leaf vectors.
pub const DEFAULT_STRUCTURED_BUILD_ROW_LIMIT: usize = 1024;

/// Snapshot of tunable BPANN parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpannTuning {
    pub index_compact_rows_per_fragment: usize,
    pub index_compact_fragment_max: usize,
    pub search_rows_per_fragment: usize,
    pub small_fragment_merge_rows: usize,
    pub search_fragment_budget_max: usize,
    pub build_seed: Option<u64>,
    pub pending_flush_threshold: usize,
    /// Hard pending cap: soft-sync on the caller when `pending >=` this value.
    pub pending_hard_flush_threshold: usize,
    pub structured_build_row_limit: usize,
    pub search_beam_width: usize,
    /// Max indexed rows using exhaustive leaf search; build stores no skip edges at or below.
    pub exhaustive_search_row_limit: usize,
    /// Max indexed rows using skip-refinement search; build stores skip edges in
    /// `(exhaustive_search_row_limit, skip_refinement_row_limit]`.
    pub skip_refinement_row_limit: usize,
}

impl Default for BpannTuning {
    fn default() -> Self {
        Self {
            index_compact_rows_per_fragment: 10_000,
            index_compact_fragment_max: 32,
            search_rows_per_fragment: 80_000,
            small_fragment_merge_rows: 15_000,
            search_fragment_budget_max: 1,
            build_seed: None,
            pending_flush_threshold: DEFAULT_PENDING_FLUSH_THRESHOLD,
            pending_hard_flush_threshold: DEFAULT_PENDING_HARD_FLUSH_THRESHOLD,

            structured_build_row_limit: DEFAULT_STRUCTURED_BUILD_ROW_LIMIT,
            search_beam_width: 1,
            exhaustive_search_row_limit: DEFAULT_EXHAUSTIVE_SEARCH_ROW_LIMIT,
            skip_refinement_row_limit: DEFAULT_SKIP_REFINEMENT_ROW_LIMIT,
        }
    }
}

impl BpannTuning {
    /// Validate all tunable fields. Returns an error describing the first violation.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            (
                self.index_compact_rows_per_fragment == 0,
                "index_compact_rows_per_fragment must be >= 1".to_string(),
            ),
            (
                self.index_compact_fragment_max < INDEX_COMPACT_FRAGMENT_MAX_MIN,
                format!("index_compact_fragment_max must be >= {INDEX_COMPACT_FRAGMENT_MAX_MIN}"),
            ),
            (
                self.search_rows_per_fragment == 0,
                "search_rows_per_fragment must be >= 1".to_string(),
            ),
            (
                self.small_fragment_merge_rows == 0,
                "small_fragment_merge_rows must be >= 1".to_string(),
            ),
            (
                self.search_fragment_budget_max == 0,
                "search_fragment_budget_max must be >= 1".to_string(),
            ),
            (
                self.pending_flush_threshold == 0,
                "pending_flush_threshold must be >= 1".to_string(),
            ),
            (
                self.pending_hard_flush_threshold == 0,
                "pending_hard_flush_threshold must be >= 1".to_string(),
            ),
            (
                self.pending_hard_flush_threshold < self.pending_flush_threshold,
                "pending_hard_flush_threshold must be >= pending_flush_threshold".to_string(),
            ),
            (
                self.structured_build_row_limit == 0,
                "structured_build_row_limit must be >= 1".to_string(),
            ),
            (
                self.search_beam_width == 0,
                "search_beam_width must be >= 1".to_string(),
            ),
            (
                self.exhaustive_search_row_limit == 0,
                "exhaustive_search_row_limit must be >= 1".to_string(),
            ),
            (
                self.skip_refinement_row_limit < self.exhaustive_search_row_limit,
                "skip_refinement_row_limit must be >= exhaustive_search_row_limit".to_string(),
            ),
        ];
        for (invalid, message) in checks {
            if invalid {
                return Err(message);
            }
        }
        Ok(())
    }

    /// Whether build should store skip edges for this indexed-row count.
    ///
    /// On-disk skip edges reflect build-time limits; search uses call-time limits
    /// and may take the skip-refinement path with empty edges until rebuild.
    pub fn rows_need_skip_edges(&self, row_count: usize) -> bool {
        row_count > self.exhaustive_search_row_limit
            && row_count <= self.skip_refinement_row_limit
    }

    /// Whether search should scan all leaves exhaustively for this row count.
    pub fn use_exhaustive_search(&self, rows: usize) -> bool {
        rows <= self.exhaustive_search_row_limit
    }

    /// Whether search should use skip-refinement (non-exhaustive, within skip band).
    pub fn use_skip_refinement_search(&self, rows: usize) -> bool {
        !self.use_exhaustive_search(rows) && rows <= self.skip_refinement_row_limit
    }
}

type TuningProvider = Box<dyn Fn() -> BpannTuning + Send + Sync>;

static TUNING_PROVIDER: RwLock<Option<TuningProvider>> = RwLock::new(None);

/// Install a provider consulted on every tuning access.
pub fn set_tuning_provider(provider: TuningProvider) {
    *TUNING_PROVIDER.write().expect("tuning provider lock") = Some(provider);
}

/// Clear any installed provider (tests).
pub fn clear_tuning_provider() {
    *TUNING_PROVIDER.write().expect("tuning provider lock") = None;
}

/// Current tuning: from the provider if set, otherwise compiled-in defaults.
///
/// If the provider returns an invalid snapshot, falls back to defaults.
pub fn current_tuning() -> BpannTuning {
    let tuning = if let Some(provider) = TUNING_PROVIDER.read().expect("tuning provider lock").as_ref()
    {
        provider()
    } else {
        BpannTuning::default()
    };
    if tuning.validate().is_ok() {
        tuning
    } else {
        BpannTuning::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tuning_is_valid() {
        assert!(BpannTuning::default().validate().is_ok());
    }

    #[test]
    fn default_pending_flush_threshold_is_250() {
        assert_eq!(DEFAULT_PENDING_FLUSH_THRESHOLD, 250);
        assert_eq!(
            BpannTuning::default().pending_flush_threshold,
            DEFAULT_PENDING_FLUSH_THRESHOLD
        );
        assert_eq!(
            current_tuning().pending_flush_threshold,
            DEFAULT_PENDING_FLUSH_THRESHOLD
        );
    }

    #[test]
    fn default_pending_hard_flush_threshold_is_3000() {
        assert_eq!(DEFAULT_PENDING_HARD_FLUSH_THRESHOLD, 3000);

        assert_eq!(
            BpannTuning::default().pending_hard_flush_threshold,
            DEFAULT_PENDING_HARD_FLUSH_THRESHOLD
        );
        assert_eq!(
            current_tuning().pending_hard_flush_threshold,
            DEFAULT_PENDING_HARD_FLUSH_THRESHOLD
        );
    }

    #[test]
    fn metamorphic_default_threshold_independent_of_other_fields() {

        let base = BpannTuning::default();
        let variants = [
            BpannTuning {
                index_compact_rows_per_fragment: 5_000,
                ..base
            },
            BpannTuning {
                search_beam_width: 8,
                ..base
            },
            BpannTuning {
                structured_build_row_limit: 4_096,
                ..base
            },
        ];
        for v in variants {
            assert_eq!(v.pending_flush_threshold, DEFAULT_PENDING_FLUSH_THRESHOLD);
            assert_eq!(
                v.pending_hard_flush_threshold,
                DEFAULT_PENDING_HARD_FLUSH_THRESHOLD
            );
            assert!(v.validate().is_ok());
        }
    }

    #[test]
    fn fuzz_pending_flush_threshold_validation_all_seeds() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = 0x5045_4e44_u64;
        println!("fuzz_pending_flush_threshold_validation seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..64 {
            let soft = rng.gen_range(0usize..10_000);
            let hard = rng.gen_range(0usize..10_000);
            let t = BpannTuning {
                pending_flush_threshold: soft,
                pending_hard_flush_threshold: hard,
                ..Default::default()
            };
            let expect_ok = soft >= 1 && hard >= 1 && hard >= soft;
            assert_eq!(t.validate().is_ok(), expect_ok, "soft={soft} hard={hard}");
        }
    }

    #[test]
    fn rejects_hard_below_soft_pending_threshold() {
        let t = BpannTuning {
            pending_flush_threshold: 500,
            pending_hard_flush_threshold: 499,
            ..Default::default()
        };
        assert!(t
            .validate()
            .unwrap_err()
            .contains("pending_hard_flush_threshold"));
    }

    #[test]
    fn rejects_zero_hard_pending_threshold() {
        let t = BpannTuning {
            pending_hard_flush_threshold: 0,
            pending_flush_threshold: 1,
            ..Default::default()
        };
        assert!(t
            .validate()
            .unwrap_err()
            .contains("pending_hard_flush_threshold"));
    }

    #[test]
    fn rejects_zero_divisors_and_beam() {
        let t = BpannTuning {
            index_compact_rows_per_fragment: 0,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("index_compact_rows_per_fragment"));
        let t = BpannTuning {
            search_rows_per_fragment: 0,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("search_rows_per_fragment"));
        let t = BpannTuning {
            search_beam_width: 0,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("search_beam_width"));
    }

    #[test]
    fn rejects_fragment_max_below_compact_floor() {
        let t = BpannTuning {
            index_compact_fragment_max: INDEX_COMPACT_FRAGMENT_MAX_MIN - 1,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("index_compact_fragment_max"));
    }

    #[test]
    fn current_tuning_falls_back_on_invalid_provider() {
        set_tuning_provider(Box::new(|| BpannTuning {
            pending_flush_threshold: 0,
            ..Default::default()
        }));
        assert_eq!(current_tuning(), BpannTuning::default());
        clear_tuning_provider();
    }

    #[test]
    fn default_search_row_limits_match_historical_cliffs() {
        let t = BpannTuning::default();
        assert_eq!(t.exhaustive_search_row_limit, DEFAULT_EXHAUSTIVE_SEARCH_ROW_LIMIT);
        assert_eq!(t.skip_refinement_row_limit, DEFAULT_SKIP_REFINEMENT_ROW_LIMIT);
        assert_eq!(DEFAULT_EXHAUSTIVE_SEARCH_ROW_LIMIT, 2500);
        assert_eq!(DEFAULT_SKIP_REFINEMENT_ROW_LIMIT, 150_000);
        assert_eq!(t.structured_build_row_limit, DEFAULT_STRUCTURED_BUILD_ROW_LIMIT);
        assert_eq!(DEFAULT_STRUCTURED_BUILD_ROW_LIMIT, 1024);

        assert_eq!(t.search_fragment_budget_max, 1);
    }

    #[test]
    fn default_needs_skip_edges_boundaries() {
        let t = BpannTuning::default();
        assert!(!t.rows_need_skip_edges(0));
        assert!(!t.rows_need_skip_edges(DEFAULT_EXHAUSTIVE_SEARCH_ROW_LIMIT));
        assert!(t.rows_need_skip_edges(DEFAULT_EXHAUSTIVE_SEARCH_ROW_LIMIT + 1));
        assert!(t.rows_need_skip_edges(DEFAULT_SKIP_REFINEMENT_ROW_LIMIT));
        assert!(!t.rows_need_skip_edges(DEFAULT_SKIP_REFINEMENT_ROW_LIMIT + 1));
    }

    #[test]
    fn rejects_zero_exhaustive_and_skip_below_exhaustive() {
        let t = BpannTuning {
            exhaustive_search_row_limit: 0,
            ..Default::default()
        };
        assert!(t
            .validate()
            .unwrap_err()
            .contains("exhaustive_search_row_limit"));
        let t = BpannTuning {
            exhaustive_search_row_limit: 100,
            skip_refinement_row_limit: 99,
            ..Default::default()
        };
        assert!(t
            .validate()
            .unwrap_err()
            .contains("skip_refinement_row_limit"));
    }

    #[test]
    fn equal_limits_are_valid_empty_skip_band() {
        let t = BpannTuning {
            exhaustive_search_row_limit: 500,
            skip_refinement_row_limit: 500,
            ..Default::default()
        };
        assert!(t.validate().is_ok());
        assert!(!t.rows_need_skip_edges(500));
        assert!(!t.rows_need_skip_edges(501));
        assert!(t.use_exhaustive_search(500));
        assert!(!t.use_skip_refinement_search(501));
    }

    #[test]
    fn metamorphic_search_mode_matches_needs_skip_edges() {

        let base = BpannTuning::default();
        let pairs = [
            (1usize, 1usize),
            (1, 10),
            (2500, 150_000),
            (100, 100),
            (10_000, usize::MAX),
        ];
        for (ex, skip) in pairs {
            let t = BpannTuning {
                exhaustive_search_row_limit: ex,
                skip_refinement_row_limit: skip,
                ..base
            };
            assert!(t.validate().is_ok(), "ex={ex} skip={skip}");
            for rows in [0, 1, ex.saturating_sub(1), ex, ex.saturating_add(1), skip, skip.saturating_add(1)] {
                assert_eq!(
                    t.rows_need_skip_edges(rows),
                    t.use_skip_refinement_search(rows),
                    "rows={rows} ex={ex} skip={skip}"
                );
                assert_eq!(
                    t.use_exhaustive_search(rows),
                    rows <= ex,
                    "exhaustive rows={rows}"
                );
            }
        }
    }

    #[test]
    fn fuzz_search_row_limit_validation_all_seeds() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = 0x524f_5753_u64;
        println!("fuzz_search_row_limit_validation seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..128 {
            let ex = rng.gen_range(0usize..5_000);
            let skip = rng.gen_range(0usize..10_000);
            let t = BpannTuning {
                exhaustive_search_row_limit: ex,
                skip_refinement_row_limit: skip,
                ..Default::default()
            };
            let ok = t.validate().is_ok();
            if ex == 0 || skip < ex {
                assert!(!ok, "ex={ex} skip={skip} should be invalid");
            } else {
                assert!(ok, "ex={ex} skip={skip} should be valid");
            }
        }
    }

    #[test]
    fn tuning_provider_switches_search_mode_at_fixed_rows() {
        clear_tuning_provider();
        let rows = 3_000usize;

        assert!(current_tuning().rows_need_skip_edges(rows));
        assert!(current_tuning().use_skip_refinement_search(rows));

        set_tuning_provider(Box::new(|| BpannTuning {
            exhaustive_search_row_limit: 10_000,
            skip_refinement_row_limit: 150_000,
            ..Default::default()
        }));
        assert!(!current_tuning().rows_need_skip_edges(rows));
        assert!(current_tuning().use_exhaustive_search(rows));

        set_tuning_provider(Box::new(|| BpannTuning {
            exhaustive_search_row_limit: 100,
            skip_refinement_row_limit: 200,
            ..Default::default()
        }));
        assert!(!current_tuning().rows_need_skip_edges(rows));
        assert!(!current_tuning().use_exhaustive_search(rows));
        assert!(!current_tuning().use_skip_refinement_search(rows));

        clear_tuning_provider();
    }
}
