use std::time::Instant;

use super::forest::ForestIndex;
use super::row_order::{OrderBuild, OrderParams, Rows};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) fn io_marks() -> [u64; 2] {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let tail = stat.rsplit_once(')').map_or("", |(_, t)| t);
    let faults = tail
        .split_whitespace()
        .nth(9)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let io = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let read = io
        .lines()
        .find_map(|l| l.strip_prefix("read_bytes: "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    [faults, read]
}

pub(crate) fn diag_marks() -> [u64; 2] {
    if diag_on() {
        io_marks()
    } else {
        [0, 0]
    }
}

fn diag_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ENN_DIAG_FAULTS").is_some())
}

pub(crate) fn diag_faults() -> u64 {
    diag_marks()[0]
}

impl ForestIndex {
    pub(crate) fn set_row_order(&mut self, params: OrderParams) -> Result<(), ENNError> {
        if !(params.grow > 1.0 && params.grow.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "row order grow must be finite and above 1".into(),
            ));
        }
        self.params.order = params;
        if params.rows_per_insert == 0 {
            self.order_build = None;
        }
        Ok(())
    }

    fn due(&self) -> bool {
        let p = self.params.order;
        let covered = self.order.as_ref().map_or(0, |o| o.rows());
        self.num_rows >= p.min_rows.max(1) && self.num_rows as f64 >= p.grow * covered as f64
    }

    fn begin_order(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let firsts: Vec<&super::forest_tree::Tree> = self
            .trees
            .iter()
            .step_by(self.params.num_trees.max(1))
            .collect();
        let generation = self.order.as_ref().map_or(1, |o| o.generation() + 1);
        self.order_build = OrderBuild::start(
            &self.work_dir,
            generation,
            &firsts,
            self.num_rows,
            rows.ncols,
            self.params.order.sort_bytes,
        )?;
        Ok(())
    }

    fn step_order(&mut self, rows: &MmapColumnStore, budget: usize) -> Result<(), ENNError> {
        let src = Rows {
            x: rows,
            order: self.order.as_ref(),
            near: None,
        };
        let Some(build) = self.order_build.as_mut() else {
            return Ok(());
        };
        self.maint.order_rows += build.step(src, budget)? as u64;
        if build.finished() {
            let done = self
                .order_build
                .take()
                .map(|b| b.finish(&self.work_dir))
                .transpose()?;
            self.order = done;
            self.maint.order_builds += 1;
        }
        Ok(())
    }

    pub(crate) fn advance_order(
        &mut self,
        rows: &MmapColumnStore,
        inserted: usize,
    ) -> Result<(), ENNError> {
        let per = self.params.order.rows_per_insert;
        if per == 0 || inserted == 0 || super::forest_lrows::clustered() {
            return Ok(());
        }
        let t0 = Instant::now();
        if self.order_build.is_none() && self.due() {
            self.begin_order(rows)?;
        }
        self.step_order(rows, inserted.saturating_mul(per))?;
        self.maint.order_ns += t0.elapsed().as_nanos() as u64;
        Ok(())
    }

    pub(crate) fn order_now(&mut self, rows: &MmapColumnStore) -> Result<usize, ENNError> {
        if super::forest_lrows::clustered() {
            return Ok(0);
        }
        let t0 = Instant::now();
        self.step_order(rows, usize::MAX)?;
        if self.order.as_ref().map_or(0, |o| o.rows()) < self.num_rows {
            self.begin_order(rows)?;
            self.step_order(rows, usize::MAX)?;
        }
        self.maint.order_ns += t0.elapsed().as_nanos() as u64;
        Ok(self.order.as_ref().map_or(0, |o| o.rows()))
    }
}
