use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::forest::ForestIndex;
use super::forest_jobbuild::levels;
use super::forest_tree::Tree;
use super::lsh::{io_error, table_seed};
use super::row_order::Rows;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const FOREST_METRICS_FILE: &str = "o1f_metrics.json";

pub(crate) fn scale_drift(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(u, v)| (u.ln() - v.ln()).abs())
        .fold(0.0, f64::max)
}

pub(crate) fn inverse(scale: &[f64]) -> Vec<f64> {
    scale.iter().map(|s| 1.0 / s).collect()
}

pub(crate) fn shadow_path(work_dir: &Path, t: usize) -> PathBuf {
    work_dir.join(format!("o1f_tree_{t}.new.bin"))
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ForestMetrics {
    pub version: u64,
    pub target: Vec<f64>,
    pub tree_scales: Vec<Vec<f64>>,
    pub repair: Option<(usize, Vec<f64>)>,
}

pub(crate) struct Repair {
    pub tree: usize,
    pub inv: Vec<f64>,
    pub shadow: Tree,
    pub cursor: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct RepairStats {
    pub started: u64,
    pub finished: u64,
    pub abandoned: u64,
    pub rows: u64,
}

fn catchup_rate() -> Option<usize> {
    static RATE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *RATE.get_or_init(|| {
        std::env::var("ENN_REPAIR_CATCHUP")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

fn catchup_room(budget: usize, repair_rows: usize, defer_rows: u32) -> usize {
    room_at(catchup_rate(), budget, repair_rows, defer_rows)
}

pub(crate) fn room_at(
    rate: Option<usize>,
    budget: usize,
    repair_rows: usize,
    defer_rows: u32,
) -> usize {
    match rate {
        Some(rate) if defer_rows > 0 && repair_rows > 0 => {
            (budget / repair_rows).saturating_mul(rate).min(budget)
        }
        _ => budget,
    }
}

impl ForestIndex {
    pub(crate) fn tree_drift(&self, t: usize) -> f64 {
        scale_drift(&self.metrics.tree_scales[t], &self.metrics.target)
    }

    pub(crate) fn stale_trees(&self) -> usize {
        (0..self.trees.len())
            .filter(|&t| self.tree_drift(t) > self.params.stale_drift)
            .count()
    }

    fn stalest(&self) -> Option<usize> {
        (0..self.trees.len())
            .filter(|&t| self.tree_drift(t) > self.params.stale_drift)
            .max_by(|&a, &b| {
                self.tree_drift(a)
                    .total_cmp(&self.tree_drift(b))
                    .then(b.cmp(&a))
            })
    }

    pub(crate) fn retarget(&mut self, scale: &[f64]) -> Result<(), ENNError> {
        self.metrics.target = scale.to_vec();
        if let Some((t, s)) = &self.metrics.repair {
            if scale_drift(s, scale) > scale_drift(&self.metrics.tree_scales[*t], scale) {
                self.abandon_repair()?;
            }
        }
        self.save_metrics()
    }

    pub(crate) fn remove_stale_files(&self) -> Result<(), ENNError> {
        for t in 0..self.trees.len().max(self.params.num_trees) {
            let shadow = shadow_path(&self.work_dir, t);
            for path in [
                shadow.with_extension("pack"),
                shadow,
                super::forest::tree_path(&self.work_dir, t).with_extension("pack"),
            ] {
                match std::fs::remove_file(&path) {
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io_error(e)),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    pub(crate) fn abandon_repair(&mut self) -> Result<(), ENNError> {
        if let Some(r) = self.repair.take() {
            self.retire_io(&r.shadow);
            drop(r.shadow);
            std::fs::remove_file(shadow_path(&self.work_dir, r.tree)).map_err(io_error)?;
            self.repair_stats.abandoned += 1;
        }
        self.metrics.repair = None;
        Ok(())
    }

    fn start_repair(&mut self) -> Result<bool, ENNError> {
        let Some(t) = self.stalest() else {
            return Ok(false);
        };
        let dim = self.metrics.target.len();
        let mut shadow = Tree::create(
            &shadow_path(&self.work_dir, t),
            self.params.leaf_cap,
            table_seed(self.seed, t),
            self.params.rules.split,
            dim,
        )?;
        shadow.nodes.set_cache(self.cache.clone());
        let scale = self.metrics.target.clone();
        self.repair = Some(Repair {
            tree: t,
            inv: inverse(&scale),
            shadow,
            cursor: 0,
        });
        self.metrics.repair = Some((t, scale));
        self.repair_stats.started += 1;
        self.save_metrics()?;
        Ok(true)
    }

    fn finish_repair(&mut self) -> Result<(), ENNError> {
        let Some(mut r) = self.repair.take() else {
            return Ok(());
        };
        let path = self.tree_file(r.tree);
        r.shadow.compact(&shadow_path(&self.work_dir, r.tree))?;
        std::fs::rename(shadow_path(&self.work_dir, r.tree), &path).map_err(io_error)?;
        if let Some(l) = r.shadow.lrows.as_mut() {
            l.moved(&path)?;
        }
        r.shadow.path.clone_from(&path);
        let old = std::mem::replace(&mut self.trees[r.tree], r.shadow);
        self.retire_io(&old);
        r.shadow = old;
        self.trees[r.tree].stats.absorb(&r.shadow.stats);
        self.invs[r.tree] = r.inv;
        if let Some((_, scale)) = self.metrics.repair.take() {
            self.metrics.tree_scales[r.tree] = scale;
        }
        self.repair_stats.finished += 1;
        self.save_metrics()
    }

    pub(crate) fn maintain(
        &mut self,
        rows: &MmapColumnStore,
        budget: usize,
    ) -> Result<usize, ENNError> {
        let (alpha, mut done) = (self.params.alpha, 0);
        let top = self.num_rows.min(rows.nrows);
        while done < budget {
            if self.repair.is_none() && !self.start_repair()? {
                break;
            }
            let r = self
                .repair
                .as_mut()
                .ok_or_else(|| io_error("forest repair is missing"))?;
            if r.cursor == 0 && r.shadow.split.defer_rows > 0 {
                if r.shadow.job.is_none() {
                    r.shadow.start_bulk(top, &r.inv, alpha);
                }
                let n = r.shadow.bulk_rows().unwrap_or(0);
                if r.shadow.drive(
                    Rows {
                        x: rows,
                        order: self.order.as_ref(),
                        near: None,
                    },
                    (budget - done) * levels(n, self.params.leaf_cap),
                )? {
                    r.cursor = n;
                }
                done = budget;
                continue;
            }
            let room = catchup_room(budget, self.params.repair_rows, r.shadow.split.defer_rows)
                .saturating_sub(done);
            if room == 0 {
                break;
            }
            let hi = top.min(r.cursor + room);
            for id in r.cursor..hi {
                r.shadow.insert(
                    id as u32,
                    Rows {
                        x: rows,
                        order: self.order.as_ref(),
                        near: None,
                    },
                    &r.inv,
                    alpha,
                )?;
            }
            done += hi - r.cursor;
            r.cursor = hi;
            if r.cursor == top {
                self.finish_repair()?;
            }
        }
        self.repair_stats.rows += done as u64;
        Ok(done)
    }

    pub(crate) fn resume_repair(
        &mut self,
        cursor: Option<usize>,
        nrows: usize,
    ) -> Result<(), ENNError> {
        let Some((t, scale)) = self.metrics.repair.clone() else {
            return Ok(());
        };
        let shadow = Tree::open(
            &shadow_path(&self.work_dir, t),
            self.params.leaf_cap,
            table_seed(self.seed, t),
        );
        match (shadow, cursor) {
            (Ok(mut shadow), Some(cursor)) if cursor <= nrows && t < self.trees.len() => {
                shadow.nodes.set_cache(self.cache.clone());
                shadow.split = self.params.rules.split;
                self.repair = Some(Repair {
                    tree: t,
                    inv: inverse(&scale),
                    shadow,
                    cursor,
                });
                Ok(())
            }
            _ => {
                let _ = std::fs::remove_file(shadow_path(&self.work_dir, t));
                self.metrics.repair = None;
                self.save_metrics()
            }
        }
    }
}
