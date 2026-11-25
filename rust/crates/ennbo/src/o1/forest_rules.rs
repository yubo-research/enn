use super::forest::ForestIndex;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const DEFAULT_SPLIT_CANDIDATES: u32 = 1;
pub(crate) const DEFAULT_SPLIT_RULE: SplitRule = SplitRule::Pivot;
const MAX_LEAN: u32 = 20;
pub(crate) const DEFAULT_PAR_ABOVE: u32 = 4096;
pub(crate) const DEFAULT_LOCAL_KIB: u32 = 0;
pub(crate) const DEFAULT_GATHER_PREAD: u8 = 0;
pub(crate) const CODED_MAX_DIM: usize = 100;
pub(crate) const DEFAULT_RERANK: u32 = 30;
const PAGE_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeRule {
    MaxMargin,
    SumSquares,
}

impl ProbeRule {
    pub(crate) fn cost(self, parent: f64, margin: f64) -> f64 {
        match self {
            Self::MaxMargin => parent.max(margin),
            Self::SumSquares => parent + margin * margin,
        }
    }
}

impl std::str::FromStr for ProbeRule {
    type Err = ENNError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "max_margin" => Ok(Self::MaxMargin),
            "sum_squares" => Ok(Self::SumSquares),
            _ => Err(ENNError::InvalidParameter(format!("unknown probe rule {s:?}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectRule {
    FirstSeen,
    Votes,
}

impl std::str::FromStr for SelectRule {
    type Err = ENNError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "first_seen" => Ok(Self::FirstSeen),
            "votes" => Ok(Self::Votes),
            _ => Err(ENNError::InvalidParameter(format!("unknown select rule {s:?}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRule {
    Seeded,
    Pivot,
    PivotI8,
    PivotF32,
}

impl SplitRule {
    pub(crate) fn is_pivot(self) -> bool {
        self != Self::Seeded
    }

    pub(crate) fn coded(self) -> Option<bool> {
        match self {
            Self::PivotI8 => Some(false),
            Self::PivotF32 => Some(true),
            _ => None,
        }
    }

    pub(crate) fn payload(self, dim: usize) -> usize {
        match self.coded() {
            Some(true) => 4 * dim,
            Some(false) => dim,
            None => 0,
        }
    }
}

impl std::str::FromStr for SplitRule {
    type Err = ENNError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "seeded" => Ok(Self::Seeded),
            "pivot" => Ok(Self::Pivot),
            "pivot_i8" => Ok(Self::PivotI8),
            "pivot_f32" => Ok(Self::PivotF32),
            _ => Err(ENNError::InvalidParameter(format!("unknown split rule {s:?}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitConfig {
    pub rule: SplitRule,
    pub candidates: u32,
    pub stage_above: u32,
    pub stage_work: u32,
    pub lean: u32,
    pub par_above: u32,
    pub local_kib: u32,
    pub pread: u8,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self { rule: DEFAULT_SPLIT_RULE, candidates: DEFAULT_SPLIT_CANDIDATES, stage_above: 0, stage_work: 0, lean: 0, par_above: DEFAULT_PAR_ABOVE, local_kib: DEFAULT_LOCAL_KIB, pread: DEFAULT_GATHER_PREAD }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForestRules {
    pub probe: ProbeRule,
    pub select: SelectRule,
    pub split: SplitConfig,
    pub bound: f64,
    pub patience: u32,
    pub rerank: u32,
}

impl Default for ForestRules {
    fn default() -> Self {
        Self { probe: ProbeRule::MaxMargin, select: SelectRule::FirstSeen, split: SplitConfig::default(), bound: 1.0, patience: 0, rerank: DEFAULT_RERANK }
    }
}

impl SplitConfig {
    pub(crate) fn for_dim(self, dim: usize) -> Self {
        let coded = self.rule == DEFAULT_SPLIT_RULE && dim <= CODED_MAX_DIM;
        Self { rule: if coded { SplitRule::PivotI8 } else { self.rule }, ..self }
    }
}

impl ForestRules {
    pub(crate) fn for_shape(self, dim: usize, leaf_cap: usize) -> Self {
        let small = 8 * dim * leaf_cap <= PAGE_BYTES;
        Self { split: self.split.for_dim(dim), rerank: if small && self.rerank == DEFAULT_RERANK { 0 } else { self.rerank }, ..self }
    }
}

impl ForestIndex {
    pub(crate) fn set_split(&mut self, rule: Option<SplitRule>, candidates: Option<u32>, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let split = &mut self.params.rules.split;
        split.rule = rule.unwrap_or(split.rule);
        split.candidates = candidates.map_or(split.candidates, |k| k.max(1));
        let split = *split;
        self.trees.iter_mut().chain(self.repair.iter_mut().map(|r| &mut r.shadow)).for_each(|t| t.split = split);
        if self.trees.iter().any(|t| !t.nodes.holds(split.rule, rows.ncols)) {
            return self.rebuild_all(rows);
        }
        Ok(())
    }

    pub(crate) fn set_trees(&mut self, n: usize, rows: &MmapColumnStore) -> Result<(), ENNError> {
        if n == 0 {
            return Err(ENNError::InvalidParameter("O1F needs at least one tree".into()));
        }
        self.params.num_trees = n;
        self.rebuild_all(rows)
    }

    pub(crate) fn set_staging(&mut self, above: u32, work: u32) {
        let split = &mut self.params.rules.split;
        (split.stage_above, split.stage_work) = (above, work);
        self.trees.iter_mut().chain(self.repair.iter_mut().map(|r| &mut r.shadow)).for_each(|t| (t.split.stage_above, t.split.stage_work) = (above, work));
    }

    pub(crate) fn set_lean(&mut self, lean: u32) -> Result<(), ENNError> {
        if lean >= MAX_LEAN {
            return Err(ENNError::InvalidParameter(format!("lean must be below {MAX_LEAN} percent")));
        }
        self.params.rules.split.lean = lean;
        self.trees.iter_mut().chain(self.repair.iter_mut().map(|r| &mut r.shadow)).for_each(|t| t.split.lean = lean);
        Ok(())
    }

    pub(crate) fn set_par_rebuild(&mut self, above: u32) {
        self.params.rules.split.par_above = above;
        self.trees.iter_mut().chain(self.repair.iter_mut().map(|r| &mut r.shadow)).for_each(|t| t.split.par_above = above);
    }

    pub(crate) fn set_local_rebuild(&mut self, kib: u32, pread: u8) {
        let split = &mut self.params.rules.split;
        (split.local_kib, split.pread) = (kib, pread);
        self.trees.iter_mut().chain(self.repair.iter_mut().map(|r| &mut r.shadow)).for_each(|t| (t.split.local_kib, t.split.pread) = (kib, pread));
    }

    pub(crate) fn set_probe_rule(&mut self, rule: ProbeRule, select: SelectRule, stop: (f64, u32)) {
        self.params.rules.probe = rule;
        self.params.rules.select = select;
        self.params.rules.bound = stop.0.max(0.0);
        self.params.rules.patience = stop.1;
    }

    pub(crate) fn set_rerank(&mut self, rerank: u32) {
        self.params.rules.rerank = rerank;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_parse_and_cost() {
        assert_eq!(ProbeRule::SumSquares.cost(1.0, 2.0), 5.0);
        assert_eq!(ProbeRule::MaxMargin.cost(1.0, 2.0), 2.0);
        assert!("bogus".parse::<ProbeRule>().is_err() && "bogus".parse::<SelectRule>().is_err());
        assert_eq!("votes".parse::<SelectRule>().unwrap(), SelectRule::Votes);
        assert_eq!("sum_squares".parse::<ProbeRule>().unwrap(), ProbeRule::SumSquares);
        assert_eq!(ForestRules::default().split, SplitConfig { rule: SplitRule::Pivot, candidates: 1, stage_above: 0, stage_work: 0, lean: 0, par_above: 4096, local_kib: DEFAULT_LOCAL_KIB, pread: DEFAULT_GATHER_PREAD });
        assert_eq!((ForestRules::default().bound, ForestRules::default().patience, ForestRules::default().rerank), (1.0, 0, 30));
        assert_eq!("pivot".parse::<SplitRule>().unwrap(), SplitRule::Pivot);
        assert_eq!(SplitConfig::default().for_dim(100).rule, SplitRule::PivotI8);
        assert_eq!((ForestRules::default().for_shape(10, 32).rerank, ForestRules::default().for_shape(30, 32).rerank), (0, 30));
        assert_eq!(ForestRules::default().for_shape(100, 32).split.rule, SplitRule::PivotI8);
        assert_eq!(SplitConfig::default().for_dim(300).rule, SplitRule::Pivot);
        assert_eq!(SplitConfig { rule: SplitRule::Seeded, ..SplitConfig::default() }.for_dim(10).rule, SplitRule::Seeded);
        assert!("bogus".parse::<SplitRule>().is_err());
    }
}
