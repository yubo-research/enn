use super::forest::ForestIndex;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const DEFAULT_SPLIT_RULE: SplitRule = SplitRule::Pivot;
pub(crate) const DEFAULT_PAR_ABOVE: u32 = 4096;
pub(crate) const DEFAULT_DEFER_ROWS: u32 = 0;
pub(crate) const DEFAULT_LOCAL_BYTES: u32 = 4 << 20;
pub(crate) const CODED_MAX_DIM: usize = 100;
pub(crate) const DEFAULT_RERANK: u32 = 30;
const PAGE_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRule {
    Pivot,
    PivotI8,
}

impl SplitRule {
    pub(crate) fn payload(self, dim: usize) -> usize {
        match self {
            Self::PivotI8 => dim,
            Self::Pivot => 0,
        }
    }
}

impl std::str::FromStr for SplitRule {
    type Err = ENNError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pivot" => Ok(Self::Pivot),
            "pivot_i8" => Ok(Self::PivotI8),
            _ => Err(ENNError::InvalidParameter(format!(
                "unknown split rule {s:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitConfig {
    pub rule: SplitRule,
    pub par_above: u32,
    pub defer_rows: u32,
    pub local_bytes: u32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            rule: DEFAULT_SPLIT_RULE,
            par_above: DEFAULT_PAR_ABOVE,
            defer_rows: DEFAULT_DEFER_ROWS,
            local_bytes: DEFAULT_LOCAL_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForestRules {
    pub split: SplitConfig,
    pub bound: f64,
    pub rerank: u32,
}

impl Default for ForestRules {
    fn default() -> Self {
        Self {
            split: SplitConfig::default(),
            bound: 1.0,
            rerank: DEFAULT_RERANK,
        }
    }
}

impl SplitConfig {
    pub(crate) fn for_dim(self, dim: usize) -> Self {
        let coded = self.rule == DEFAULT_SPLIT_RULE && dim <= CODED_MAX_DIM;
        Self {
            rule: if coded { SplitRule::PivotI8 } else { self.rule },
            ..self
        }
    }
}

impl ForestRules {
    pub(crate) fn for_shape(self, dim: usize, leaf_cap: usize) -> Self {
        let small = 8 * dim * leaf_cap <= PAGE_BYTES;
        Self {
            split: self.split.for_dim(dim),
            rerank: if small && self.rerank == DEFAULT_RERANK {
                0
            } else {
                self.rerank
            },
            ..self
        }
    }
}

impl ForestIndex {
    pub(crate) fn set_split(
        &mut self,
        rule: SplitRule,
        rows: &MmapColumnStore,
    ) -> Result<(), ENNError> {
        self.params.rules.split.rule = rule;
        let split = self.params.rules.split;
        self.trees
            .iter_mut()
            .chain(self.repair.iter_mut().map(|r| &mut r.shadow))
            .for_each(|t| t.split = split);
        if self
            .trees
            .iter()
            .any(|t| !t.nodes.holds(split.rule, rows.ncols))
        {
            return self.rebuild_all(rows);
        }
        Ok(())
    }

    pub(crate) fn set_trees(&mut self, n: usize, rows: &MmapColumnStore) -> Result<(), ENNError> {
        if n == 0 {
            return Err(ENNError::InvalidParameter(
                "O1F needs at least one tree".into(),
            ));
        }
        self.params.num_trees = n;
        self.rebuild_all(rows)
    }

    pub(crate) fn set_par_rebuild(&mut self, above: u32) {
        self.params.rules.split.par_above = above;
        self.trees
            .iter_mut()
            .chain(self.repair.iter_mut().map(|r| &mut r.shadow))
            .for_each(|t| t.split.par_above = above);
    }

    pub(crate) fn set_defer(&mut self, defer_rows: u32, local_bytes: u32) {
        self.params.rules.split.defer_rows = defer_rows;
        self.params.rules.split.local_bytes = local_bytes.max(1);
        let split = self.params.rules.split;
        self.trees
            .iter_mut()
            .chain(self.repair.iter_mut().map(|r| &mut r.shadow))
            .for_each(|t| {
                t.split.defer_rows = split.defer_rows;
                t.split.local_bytes = split.local_bytes;
            });
    }

    pub(crate) fn set_bound(&mut self, bound: f64) {
        self.params.rules.bound = bound.max(0.0);
    }

    pub(crate) fn set_rerank(&mut self, rerank: u32) {
        self.params.rules.rerank = rerank;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_parse_and_default_by_shape() {
        assert_eq!(
            ForestRules::default().split,
            SplitConfig {
                rule: SplitRule::Pivot,
                par_above: 4096,
                defer_rows: DEFAULT_DEFER_ROWS,
                local_bytes: DEFAULT_LOCAL_BYTES
            }
        );
        assert_eq!(
            (ForestRules::default().bound, ForestRules::default().rerank),
            (1.0, 30)
        );
        assert_eq!("pivot".parse::<SplitRule>().unwrap(), SplitRule::Pivot);
        assert_eq!("pivot_i8".parse::<SplitRule>().unwrap(), SplitRule::PivotI8);
        assert_eq!(SplitConfig::default().for_dim(100).rule, SplitRule::PivotI8);
        assert_eq!(
            (
                ForestRules::default().for_shape(10, 32).rerank,
                ForestRules::default().for_shape(30, 32).rerank
            ),
            (0, 30)
        );
        assert_eq!(
            ForestRules::default().for_shape(100, 32).split.rule,
            SplitRule::PivotI8
        );
        assert_eq!(SplitConfig::default().for_dim(300).rule, SplitRule::Pivot);
        assert!("bogus".parse::<SplitRule>().is_err() && "seeded".parse::<SplitRule>().is_err());
    }
}
