use super::disk_index::O1Occupancy;
use super::forest::{ForestIndex, MaintStats};
use super::forest_node::record_bytes;
use super::forest_tree::TreeStats;
use crate::error::ENNError;

impl ForestIndex {
    pub(crate) fn maint_stats(&self) -> MaintStats {
        let mut m = self.maint;
        m.ordered_rows = self.order.as_ref().map_or(0, |o| o.rows() as u64);
        for tree in &self.trees {
            m.rebuilt_rows += tree.stats.rebuilt_rows;
            m.splits += tree.stats.splits;
            m.gathered += tree.stats.gathered;
            m.rebuild_gathered += tree.stats.rebuild_gathered;
            m.route_rows += tree.stats.route_rows;
            m.deferred += tree.stats.deferred;
            m.skipped += tree.stats.skipped;
            m.aborted += tree.stats.aborted;
            m.late_rows += tree.stats.late_rows;
            m.scan_nodes += tree.stats.scan_nodes;
            m.max_rebuild_rows = m.max_rebuild_rows.max(tree.stats.max_rebuild_rows);
            m.job_active += u64::from(tree.job.is_some());
            m.phase_ns
                .iter_mut()
                .zip(tree.stats.phase_ns)
                .for_each(|(a, b)| *a += b);
            for k in 0..5 {
                m.phase_flt[k] += tree.stats.phase_flt[k];
                m.phase_rd[k] += tree.stats.phase_rd[k];
            }
            m.store_rows[0] += tree.stats.split_store;
            m.store_rows[1] += tree.stats.rebuild_store;
            m.gather_flt[0] += tree.stats.split_flt;
            m.gather_flt[1] += tree.stats.rebuild_flt;
            m.split_age
                .iter_mut()
                .zip(tree.stats.split_age)
                .for_each(|(a, b)| *a += b);
            m.near_hits += tree.stats.near_hits;
            for k in 0..12 {
                m.split_diag[k] += tree.stats.split_diag[k];
            }
            m.gather_ns[0] += tree.stats.split_ns;
            m.gather_ns[1] += tree.stats.rebuild_ns;
            m.gather_ns[2] += tree
                .nodes
                .counters
                .read_ns
                .load(std::sync::atomic::Ordering::Relaxed);
            m.gather_ns[3] += tree
                .nodes
                .counters
                .leaf_ns
                .load(std::sync::atomic::Ordering::Relaxed);
            m.gather_ns[4] += tree
                .nodes
                .counters
                .leaf_reads
                .load(std::sync::atomic::Ordering::Relaxed);
            m.pack_hits += tree
                .nodes
                .counters
                .pack_hits
                .load(std::sync::atomic::Ordering::Relaxed);
            m.pack_miss += tree
                .nodes
                .counters
                .pack_miss
                .load(std::sync::atomic::Ordering::Relaxed);
        }
        m
    }

    pub(crate) fn occupancy(&self) -> Result<O1Occupancy, ENNError> {
        let mut seen = vec![false; self.num_rows];
        let mut occ = O1Occupancy {
            num_rows: self.num_rows,
            bucket_cap: self.params.leaf_cap,
            ..O1Occupancy::default()
        };
        let mut stats = TreeStats::default();
        for tree in &self.trees {
            let (mut rows, mut leaves, mut chained, mut depth_sum) = (0, 0, 0, 0);
            tree.walk(|leaf| {
                rows += leaf.ids.len();
                leaves += 1;
                chained += usize::from(leaf.pages > 1);
                depth_sum += leaf.depth * leaf.ids.len();
                occ.hash_bits = occ.hash_bits.max(leaf.depth);
                let page_max = leaf.ids.len().div_ceil(leaf.pages.max(1));
                occ.max_arrivals = occ.max_arrivals.max(page_max as u64);
                for &i in leaf.ids {
                    if let Some(s) = seen.get_mut(i as usize) {
                        *s = true;
                    }
                }
            })?;
            occ.table_rows.push(rows);
            occ.keys_used.push(leaves);
            occ.overflow_keys.push(chained);
            occ.mean_depth += depth_sum as f64 / (rows.max(1) * self.trees.len()) as f64;
            occ.disk_nodes += tree.nodes.num_nodes() as usize;
            occ.free_nodes += tree.nodes.free_count()?;
            stats.splits += tree.stats.splits;
            stats.rebuilds += tree.stats.rebuilds;
            stats.rebuilt_rows += tree.stats.rebuilt_rows;
            stats.max_rebuild_rows = stats.max_rebuild_rows.max(tree.stats.max_rebuild_rows);
            stats.global_rebuilds = stats.global_rebuilds.max(tree.stats.global_rebuilds);
        }
        occ.reachable_rows = seen.iter().filter(|&&s| s).count();
        occ.node_bytes = self
            .trees
            .first()
            .map_or(record_bytes(self.params.leaf_cap, 0), |t| {
                t.nodes.record_bytes()
            });
        (occ.node_reads, occ.node_writes) = self.node_io();
        occ.splits = stats.splits;
        occ.rebuilds = stats.rebuilds;
        occ.rebuilt_rows = stats.rebuilt_rows;
        occ.max_rebuild_rows = stats.max_rebuild_rows;
        occ.global_rebuilds = stats.global_rebuilds;
        occ.stale_trees = self.stale_trees();
        occ.max_drift = (0..self.trees.len())
            .map(|t| self.tree_drift(t))
            .fold(0.0, f64::max);
        occ.repairs_started = self.repair_stats.started;
        occ.repairs_finished = self.repair_stats.finished;
        occ.repairs_abandoned = self.repair_stats.abandoned;
        occ.repair_rows = self.repair_stats.rows;
        occ.repair_cursor = self.repair.as_ref().map(|r| r.cursor);
        Ok(occ)
    }
}
