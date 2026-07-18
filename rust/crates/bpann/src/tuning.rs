//! Process-wide BPANN tuning values.
//!
//! Defaults match the historical hardcoded defaults. When the `ennbo` crate is
//! used, it installs a provider that reads `~/.ennbo/config.toml`.

use std::sync::RwLock;

/// Minimum allowed `index_compact_fragment_max` (matches compaction clamp floor).
pub const INDEX_COMPACT_FRAGMENT_MAX_MIN: usize = 3;

/// Default rows of pending observations before an index flush is scheduled.
pub const DEFAULT_PENDING_FLUSH_THRESHOLD: usize = 250;

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
    pub structured_build_row_limit: usize,
    pub search_beam_width: usize,
}

impl Default for BpannTuning {
    fn default() -> Self {
        Self {
            index_compact_rows_per_fragment: 10_000,
            index_compact_fragment_max: 32,
            search_rows_per_fragment: 80_000,
            small_fragment_merge_rows: 15_000,
            search_fragment_budget_max: 3,
            build_seed: None,
            pending_flush_threshold: DEFAULT_PENDING_FLUSH_THRESHOLD,
            structured_build_row_limit: 1_024,
            search_beam_width: 1,
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
                self.structured_build_row_limit == 0,
                "structured_build_row_limit must be >= 1".to_string(),
            ),
            (
                self.search_beam_width == 0,
                "search_beam_width must be >= 1".to_string(),
            ),
        ];
        for (invalid, message) in checks {
            if invalid {
                return Err(message);
            }
        }
        Ok(())
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
    fn metamorphic_default_threshold_independent_of_other_fields() {
        // Changing unrelated fields must not change the pending_flush default.
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
            assert!(v.validate().is_ok());
        }
    }

    #[test]
    fn fuzz_pending_flush_threshold_validation_all_seeds() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = 0x5045_4e44_u64; // "PEND"
        println!("fuzz_pending_flush_threshold_validation seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..64 {
            let thr = rng.gen_range(0usize..10_000);
            let t = BpannTuning {
                pending_flush_threshold: thr,
                ..Default::default()
            };
            if thr == 0 {
                assert!(t.validate().is_err());
            } else {
                assert!(t.validate().is_ok());
            }
        }
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
}
