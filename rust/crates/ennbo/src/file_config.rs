//! File-backed ennbo configuration (`~/.ennbo/config.toml`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

static CONFIG_PATH_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Override the config file path (tests / ops tools). Pass `None` to restore the default.
pub fn set_config_path(path: Option<PathBuf>) {
    *CONFIG_PATH_OVERRIDE.write().expect("config path lock") = path;
    install_bpann_tuning_from_config();
}

/// Default path: `~/.ennbo/config.toml`.
pub fn default_config_path() -> PathBuf {
    home_dir().join(".ennbo").join("config.toml")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn active_config_path() -> PathBuf {
    CONFIG_PATH_OVERRIDE
        .read()
        .expect("config path lock")
        .clone()
        .unwrap_or_else(default_config_path)
}

/// Tunable BPANN parameters persisted under `[bpann]` in the config file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BpannConfig {
    pub index_compact_rows_per_fragment: usize,
    pub index_compact_fragment_max: usize,
    pub search_rows_per_fragment: usize,
    pub small_fragment_merge_rows: usize,
    pub search_fragment_budget_max: usize,
    /// When set, used as the k-means build seed; otherwise the batch start row is used.
    pub build_seed: Option<u64>,
    /// Rows of pending observations before an index flush is scheduled.
    pub pending_flush_threshold: usize,
    /// Hard pending cap (soft-sync on caller). `None` means the key was absent in TOML;
    /// resolved to `max(DEFAULT_PENDING_HARD_FLUSH_THRESHOLD, soft)` on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_hard_flush_threshold: Option<usize>,
    /// Batch size at or below which builds use a single row-ID leaf (no k-means tree).
    pub structured_build_row_limit: usize,
    /// Beam width used during approximate tree traversal.
    pub search_beam_width: usize,
    /// Max indexed rows using exhaustive leaf search; build stores no skip edges at or below.
    pub exhaustive_search_row_limit: usize,
    /// Max indexed rows using skip-refinement search; build stores skip edges in the middle band.
    pub skip_refinement_row_limit: usize,
}

impl Default for BpannConfig {
    /// Mirror the compiled-in BPANN defaults so the values live in exactly one place.
    fn default() -> Self {
        bpann::BpannTuning::default().into()
    }
}

impl From<bpann::BpannTuning> for BpannConfig {
    fn from(t: bpann::BpannTuning) -> Self {
        Self {
            index_compact_rows_per_fragment: t.index_compact_rows_per_fragment,
            index_compact_fragment_max: t.index_compact_fragment_max,
            search_rows_per_fragment: t.search_rows_per_fragment,
            small_fragment_merge_rows: t.small_fragment_merge_rows,
            search_fragment_budget_max: t.search_fragment_budget_max,
            build_seed: t.build_seed,
            pending_flush_threshold: t.pending_flush_threshold,
            pending_hard_flush_threshold: Some(t.pending_hard_flush_threshold),
            structured_build_row_limit: t.structured_build_row_limit,
            search_beam_width: t.search_beam_width,
            exhaustive_search_row_limit: t.exhaustive_search_row_limit,
            skip_refinement_row_limit: t.skip_refinement_row_limit,
        }
    }
}

impl BpannConfig {
    /// Validate all tunable fields. Returns an error describing the first violation.
    pub fn validate(&self) -> Result<(), String> {
        self.to_tuning().validate()
    }

    /// Resolve the hard pending cap: absent key → `max(DEFAULT_HARD, soft)`.
    pub fn resolved_pending_hard_flush_threshold(&self) -> usize {
        self.pending_hard_flush_threshold.unwrap_or_else(|| {
            std::cmp::max(
                bpann::DEFAULT_PENDING_HARD_FLUSH_THRESHOLD,
                self.pending_flush_threshold,
            )
        })
    }

    fn to_tuning(&self) -> bpann::BpannTuning {
        bpann::BpannTuning {
            index_compact_rows_per_fragment: self.index_compact_rows_per_fragment,
            index_compact_fragment_max: self.index_compact_fragment_max,
            search_rows_per_fragment: self.search_rows_per_fragment,
            small_fragment_merge_rows: self.small_fragment_merge_rows,
            search_fragment_budget_max: self.search_fragment_budget_max,
            build_seed: self.build_seed,
            pending_flush_threshold: self.pending_flush_threshold,
            pending_hard_flush_threshold: self.resolved_pending_hard_flush_threshold(),
            structured_build_row_limit: self.structured_build_row_limit,
            search_beam_width: self.search_beam_width,
            exhaustive_search_row_limit: self.exhaustive_search_row_limit,
            skip_refinement_row_limit: self.skip_refinement_row_limit,
        }
    }
}

/// Root document for `~/.ennbo/config.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigFile {
    pub bpann: BpannConfig,
}

/// File-backed configuration object.
///
/// Each parameter accessor re-reads `~/.ennbo/config.toml` (or the path from
/// [`set_config_path`]). If the file is missing, it is created with defaults.
/// Invalid files fall back to defaults.
#[derive(Debug, Clone)]
pub struct Config {
    path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Config for the active path (`set_config_path` override or `~/.ennbo/config.toml`).
    pub fn new() -> Self {
        Self {
            path: active_config_path(),
        }
    }

    /// Config for an explicit path (does not change the process-wide override).
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ensure the config file exists, then load and validate it.
    ///
    /// Parse errors or invalid parameter values fall back to defaults.
    pub fn load(&self) -> ConfigFile {
        self.ensure_exists();
        let file = match fs::read_to_string(&self.path) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => ConfigFile::default(),
        };
        if file.bpann.validate().is_ok() {
            file
        } else {
            ConfigFile::default()
        }
    }

    /// Write `file` to this config path, creating parent directories as needed.
    ///
    /// Rejects invalid BPANN parameter values.
    pub fn save(&self, file: &ConfigFile) -> Result<(), String> {
        file.bpann.validate()?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = toml::to_string_pretty(file).map_err(|e| e.to_string())?;
        fs::write(&self.path, text).map_err(|e| e.to_string())
    }

    fn ensure_exists(&self) {
        if self.path.exists() {
            return;
        }
        let _ = self.save(&ConfigFile::default());
    }

    pub fn index_compact_rows_per_fragment(&self) -> usize {
        self.load().bpann.index_compact_rows_per_fragment
    }

    pub fn index_compact_fragment_max(&self) -> usize {
        self.load().bpann.index_compact_fragment_max
    }

    pub fn search_rows_per_fragment(&self) -> usize {
        self.load().bpann.search_rows_per_fragment
    }

    pub fn small_fragment_merge_rows(&self) -> usize {
        self.load().bpann.small_fragment_merge_rows
    }

    pub fn search_fragment_budget_max(&self) -> usize {
        self.load().bpann.search_fragment_budget_max
    }

    pub fn build_seed(&self) -> Option<u64> {
        self.load().bpann.build_seed
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.load().bpann.pending_flush_threshold
    }

    pub fn pending_hard_flush_threshold(&self) -> usize {
        self.load().bpann.resolved_pending_hard_flush_threshold()
    }

    pub fn structured_build_row_limit(&self) -> usize {
        self.load().bpann.structured_build_row_limit
    }

    pub fn search_beam_width(&self) -> usize {
        self.load().bpann.search_beam_width
    }

    pub fn exhaustive_search_row_limit(&self) -> usize {
        self.load().bpann.exhaustive_search_row_limit
    }

    pub fn skip_refinement_row_limit(&self) -> usize {
        self.load().bpann.skip_refinement_row_limit
    }
}

/// Install BPANN tuning so the `bpann` crate reads values via [`Config`].
///
/// Snapshot once at install time. Reloading the TOML inside the provider (old
/// behavior) put disk I/O on every `current_tuning()` — including per-query
/// search mode selection — and dominated TuRBO ask time.
pub fn install_bpann_tuning_from_config() {
    let cached = Config::new().load().bpann.to_tuning();
    bpann::set_tuning_provider(Box::new(move || cached));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn creates_default_file_when_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        assert!(!path.exists());
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.index_compact_rows_per_fragment(), 10_000);
        assert_eq!(cfg.pending_flush_threshold(), 250);
        assert_eq!(cfg.pending_hard_flush_threshold(), 3000);
        assert_eq!(cfg.structured_build_row_limit(), 1_024);
        assert_eq!(cfg.search_beam_width(), 1);
        assert_eq!(cfg.exhaustive_search_row_limit(), 2500);
        assert_eq!(cfg.skip_refinement_row_limit(), 150_000);
        assert!(path.exists());
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("index_compact_rows_per_fragment"));
        assert!(text.contains("pending_flush_threshold"));
        assert!(text.contains("pending_hard_flush_threshold"));
        assert!(text.contains("structured_build_row_limit"));
        assert!(text.contains("search_beam_width"));
        assert!(text.contains("exhaustive_search_row_limit"));
        assert!(text.contains("skip_refinement_row_limit"));
        assert!(text.contains("10000"));
        assert!(text.contains("pending_flush_threshold = 250"));
        assert!(text.contains("pending_hard_flush_threshold = 3000"));
        assert!(text.contains("exhaustive_search_row_limit = 2500"));
        assert!(text.contains("skip_refinement_row_limit = 150000"));
    }

    #[test]
    fn accessors_reread_updated_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.search_fragment_budget_max(), 3);
        let mut file = ConfigFile::default();
        file.bpann.search_fragment_budget_max = 7;
        file.bpann.search_beam_width = 4;
        file.bpann.pending_flush_threshold = 2_000;
        file.bpann.pending_hard_flush_threshold = Some(4_000);
        file.bpann.structured_build_row_limit = 2_048;
        file.bpann.exhaustive_search_row_limit = 5_000;
        file.bpann.skip_refinement_row_limit = 200_000;
        cfg.save(&file).unwrap();
        assert_eq!(cfg.search_fragment_budget_max(), 7);
        assert_eq!(cfg.search_beam_width(), 4);
        assert_eq!(cfg.pending_flush_threshold(), 2_000);
        assert_eq!(cfg.pending_hard_flush_threshold(), 4_000);
        assert_eq!(cfg.structured_build_row_limit(), 2_048);
        assert_eq!(cfg.exhaustive_search_row_limit(), 5_000);
        assert_eq!(cfg.skip_refinement_row_limit(), 200_000);
    }

    #[test]
    fn missing_hard_key_with_soft_above_default_preserves_soft() {
        // Q5: absent hard must not full-default-fallback when soft is elevated.
        // Resolve policy: hard = max(DEFAULT_HARD, soft). With DEFAULT_HARD=3000
        // and soft=2000, hard becomes 3000; soft stays 2000.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[bpann]\npending_flush_threshold = 2000\nsearch_beam_width = 1\n",
        )
        .unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.pending_flush_threshold(), 2_000);
        assert_eq!(cfg.pending_hard_flush_threshold(), 3_000);
    }

    #[test]
    fn missing_hard_key_with_soft_above_default_hard_uses_soft() {
        // When soft exceeds DEFAULT_HARD, absent hard resolves to soft.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[bpann]\npending_flush_threshold = 5000\nsearch_beam_width = 1\n",
        )
        .unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.pending_flush_threshold(), 5_000);
        assert_eq!(cfg.pending_hard_flush_threshold(), 5_000);
    }

    #[test]
    fn missing_hard_key_with_default_soft_uses_default_hard() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[bpann]\npending_flush_threshold = 250\nsearch_beam_width = 1\n",
        )
        .unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.pending_flush_threshold(), 250);
        assert_eq!(cfg.pending_hard_flush_threshold(), 3000);
    }

    #[test]
    fn explicit_hard_below_soft_falls_back_to_defaults() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[bpann]\npending_flush_threshold = 2000\npending_hard_flush_threshold = 1000\nsearch_beam_width = 1\n",
        )
        .unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.pending_flush_threshold(), 250);
        assert_eq!(cfg.pending_hard_flush_threshold(), 3000);
    }

    #[test]
    fn save_rejects_hard_below_soft() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        let mut file = ConfigFile::default();
        file.bpann.pending_flush_threshold = 500;
        file.bpann.pending_hard_flush_threshold = Some(100);
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("pending_hard_flush_threshold"));
    }

    #[test]
    fn save_rejects_invalid_parameters() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        let mut file = ConfigFile::default();
        file.bpann.search_beam_width = 0;
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("search_beam_width"));

        let mut file = ConfigFile::default();
        file.bpann.exhaustive_search_row_limit = 0;
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("exhaustive_search_row_limit"));

        let mut file = ConfigFile::default();
        file.bpann.exhaustive_search_row_limit = 100;
        file.bpann.skip_refinement_row_limit = 50;
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("skip_refinement_row_limit"));
    }

    #[test]
    fn load_falls_back_to_defaults_on_invalid_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[bpann]\nindex_compact_rows_per_fragment = 0\nsearch_beam_width = 1\n",
        )
        .unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.index_compact_rows_per_fragment(), 10_000);
        assert_eq!(cfg.search_beam_width(), 1);
    }

    #[test]
    fn set_config_path_overrides_default() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("override.toml");
        set_config_path(Some(path.clone()));
        let cfg = Config::new();
        assert_eq!(cfg.path(), path.as_path());
        assert_eq!(cfg.index_compact_fragment_max(), 32);
        set_config_path(None);
    }

    #[test]
    fn default_bpann_config_is_valid() {
        assert!(BpannConfig::default().validate().is_ok());
    }

    #[test]
    fn default_config_path_is_home_ennbo() {
        let expected = home_dir().join(".ennbo").join("config.toml");
        assert_eq!(default_config_path(), expected);
        assert!(default_config_path().ends_with("config.toml"));
    }

    #[test]
    fn active_config_path_reflects_override() {
        assert_eq!(active_config_path(), default_config_path());
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("override.toml");
        set_config_path(Some(path.clone()));
        assert_eq!(active_config_path(), path);
        set_config_path(None);
        assert_eq!(active_config_path(), default_config_path());
    }

    #[test]
    fn accessors_read_search_merge_and_seed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.search_rows_per_fragment(), 80_000);
        assert_eq!(cfg.small_fragment_merge_rows(), 15_000);
        assert_eq!(cfg.build_seed(), None);

        let mut file = ConfigFile::default();
        file.bpann.search_rows_per_fragment = 50_000;
        file.bpann.small_fragment_merge_rows = 9_000;
        file.bpann.build_seed = Some(42);
        cfg.save(&file).unwrap();

        assert_eq!(cfg.search_rows_per_fragment(), 50_000);
        assert_eq!(cfg.small_fragment_merge_rows(), 9_000);
        assert_eq!(cfg.build_seed(), Some(42));
    }

    #[test]
    fn install_tuning_picks_up_row_limit_overrides() {
        // Avoid set_config_path here: it is process-global and races other tests.
        // Mirror install_bpann_tuning_from_config by loading via with_path → to_tuning.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        let mut file = ConfigFile::default();
        file.bpann.exhaustive_search_row_limit = 7_000;
        file.bpann.skip_refinement_row_limit = 90_000;
        cfg.save(&file).unwrap();

        let t = cfg.load().bpann.to_tuning();
        assert_eq!(t.exhaustive_search_row_limit, 7_000);
        assert_eq!(t.skip_refinement_row_limit, 90_000);
        assert!(!t.rows_need_skip_edges(5_000));
        assert!(t.rows_need_skip_edges(8_000));
        assert!(!t.rows_need_skip_edges(100_000));
    }
}
