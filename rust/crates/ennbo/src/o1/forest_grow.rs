use super::forest_node::{Node, NIL};
use super::forest_split::{choose_split, Parted, Placed};
use super::forest_tree::{row, uncovered, Tree};
use super::lsh::io_error;
use super::row_order::Rows;
use crate::error::ENNError;

impl Tree {
    pub(super) fn place(
        &mut self,
        id: u32,
        leaf: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        skip_present: bool,
        split_ok: bool,
    ) -> Result<Placed, ENNError> {
        let mut pages = Vec::new();
        let mut at = leaf;
        let mut last = Vec::new();
        while at != NIL {
            let Node::Leaf { ids, next } = self.nodes.read(at)? else {
                return Err(io_error("forest leaf chain is corrupt"));
            };
            if skip_present && ids.contains(&id) {
                return Ok(Placed::Present);
            }
            pages.push(at);
            (last, at) = (ids, next);
        }
        let tail = *pages
            .last()
            .ok_or_else(|| io_error("forest leaf is missing"))?;
        self.ensure_lrows(rows.x.ncols)?;
        if last.len() < self.nodes.cap() {
            last.push(id);
            self.nodes.write(
                tail,
                &Node::Leaf {
                    ids: last.clone(),
                    next: NIL,
                },
            )?;
            return Ok(Placed::Appended);
        }
        if !split_ok || (pages.len() > 1 && !pages.len().is_power_of_two()) {
            let page = self.nodes.alloc()?;
            self.nodes.write(
                page,
                &Node::Leaf {
                    ids: vec![id],
                    next: NIL,
                },
            )?;
            self.stats.chain_pages += 1;
            self.nodes.write(
                tail,
                &Node::Leaf {
                    ids: last,
                    next: page,
                },
            )?;
            return Ok(Placed::Appended);
        }
        let mut ids = Vec::new();
        self.leaf_ids(leaf, &mut ids)?;
        ids.push(id);
        self.stats.split_store += uncovered(rows, &ids);
        for &i in &ids {
            let age = id.saturating_sub(i);
            self.stats.split_age[[1 << 14, 1 << 16, 1 << 18, u32::MAX]
                .iter()
                .position(|&b| age < b)
                .unwrap_or(3)] += 1;
        }
        let (f0, t0) = (
            super::forest_order::diag_faults(),
            std::time::Instant::now(),
        );
        let near = self.split_rows(&pages, &ids, id, rows)?;
        let made = self.build(
            leaf,
            ids,
            Rows {
                near: near.as_ref(),
                ..rows
            },
            inv_scale,
        )?;
        self.stats.split_ns += t0.elapsed().as_nanos() as u64;
        self.stats.split_flt += super::forest_order::diag_faults().saturating_sub(f0);
        self.nodes.free(&pages[1..])?;
        Ok(Placed::Split(made))
    }

    pub(super) fn part(
        &mut self,
        slot: u32,
        mut ids: Vec<u32>,
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<Option<Parted>, ENNError> {
        if ids.len() <= self.nodes.cap() {
            if let Some(l) = &mut self.lrows {
                let owned: Vec<_> = ids
                    .iter()
                    .filter_map(|&i| row(rows, i).ok().map(|r| (i, r)))
                    .collect();
                l.put_all(slot, owned.iter().map(|(i, r)| (*i, &r[..])))?;
            }
            self.nodes.write(slot, &Node::Leaf { ids, next: NIL })?;
            return Ok(None);
        }
        self.stats.gathered += ids.len() as u64;
        let held = ids
            .iter()
            .map(|&i| row(rows, i))
            .collect::<Result<Vec<_>, _>>()?;
        let xs: Vec<&[f64]> = held.iter().map(|r| &r[..]).collect();
        let split = self.effective_split(rows.x.ncols);
        let Some((cut, thr, left)) = choose_split(
            &self.projector(inv_scale),
            &xs,
            &ids,
            slot,
            self.seed,
            split,
        ) else {
            self.write_chain(slot, ids)?;
            self.seed_from_rows(slot, rows)?;
            return Ok(None);
        };
        drop(xs);
        drop(held);
        let (mut l_ids, mut r_ids) = (Vec::new(), Vec::new());
        for (id, is_left) in ids.drain(..).zip(left) {
            if is_left {
                l_ids.push(id)
            } else {
                r_ids.push(id)
            }
        }
        let (l, r) = (self.nodes.alloc()?, self.nodes.alloc()?);
        Ok(Some(Parted {
            node: Node::Split {
                cut,
                thr,
                left: l,
                right: r,
            },
            left: (l, l_ids),
            right: (r, r_ids),
        }))
    }

    pub(crate) fn build(
        &mut self,
        slot: u32,
        ids: Vec<u32>,
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<usize, ENNError> {
        self.ensure_lrows(rows.x.ncols)?;
        let Some(p) = self.part(slot, ids, rows, inv_scale)? else {
            return Ok(0);
        };
        let dl = self.build(p.left.0, p.left.1, rows, inv_scale)?;
        let dr = self.build(p.right.0, p.right.1, rows, inv_scale)?;
        self.nodes.write(slot, &p.node)?;
        self.stats.splits += 1;
        Ok(1 + dl.max(dr))
    }

    pub(super) fn write_chain(&mut self, slot: u32, ids: Vec<u32>) -> Result<usize, ENNError> {
        let chunks: Vec<Vec<u32>> = ids.chunks(self.nodes.cap()).map(<[u32]>::to_vec).collect();
        let mut next = NIL;
        for chunk in chunks[1..].iter().rev() {
            let page = self.nodes.alloc()?;
            self.nodes.write(
                page,
                &Node::Leaf {
                    ids: chunk.clone(),
                    next,
                },
            )?;
            next = page;
            self.stats.chain_pages += 1;
        }
        self.nodes.write(
            slot,
            &Node::Leaf {
                ids: chunks[0].clone(),
                next,
            },
        )?;
        Ok(0)
    }

    pub(crate) fn seed_leaves(
        &mut self,
        slot: u32,
        ids: &[u32],
        xs: &[f64],
        dim: usize,
    ) -> Result<(), ENNError> {
        if dim == 0 || xs.len() != ids.len() * dim {
            return Ok(());
        }
        self.ensure_lrows(dim)?;
        if self.lrows.is_none() {
            return Ok(());
        }
        let mut at = std::collections::HashMap::<u32, usize>::with_capacity(ids.len());
        ids.iter().enumerate().for_each(|(i, &id)| {
            at.insert(id, i);
        });
        for (page, page_ids) in self.leaf_pages(slot)? {
            let found = page_ids
                .iter()
                .filter_map(|&id| at.get(&id).map(|&i| (id, &xs[i * dim..(i + 1) * dim])));
            if let Some(l) = &mut self.lrows {
                l.put_all(page, found)?;
            }
        }
        Ok(())
    }

    fn seed_from_rows(&mut self, slot: u32, rows: Rows<'_>) -> Result<(), ENNError> {
        let dim = rows.x.ncols;
        self.ensure_lrows(dim)?;
        if self.lrows.is_none() {
            return Ok(());
        }
        for (page, page_ids) in self.leaf_pages(slot)? {
            let owned = page_ids
                .iter()
                .map(|&id| Ok((id, row(rows, id)?.to_vec())))
                .collect::<Result<Vec<_>, ENNError>>()?;
            if let Some(l) = &mut self.lrows {
                l.put_all(page, owned.iter().map(|(i, r)| (*i, r.as_slice())))?;
            }
        }
        Ok(())
    }

    fn leaf_pages(&self, slot: u32) -> Result<Vec<(u32, Vec<u32>)>, ENNError> {
        let mut stack = vec![slot];
        let mut leaves = Vec::new();
        while let Some(n) = stack.pop() {
            match self.nodes.read(n)? {
                Node::Leaf { ids, next } => {
                    leaves.push((n, ids));
                    if next != NIL {
                        stack.push(next);
                    }
                }
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(leaves)
    }

    pub(super) fn gather(
        &self,
        node: u32,
        ids: &mut Vec<u32>,
        nodes: &mut Vec<u32>,
    ) -> Result<(), ENNError> {
        let mut stack = vec![node];
        while let Some(at) = stack.pop() {
            nodes.push(at);
            match self.nodes.read(at)? {
                Node::Leaf { ids: page, next } => {
                    ids.extend_from_slice(&page);
                    stack.extend((next != NIL).then_some(next));
                }
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(())
    }

    fn count(&self, node: u32) -> Result<usize, ENNError> {
        let (mut stack, mut n) = (vec![node], 0);
        while let Some(at) = stack.pop() {
            match self.nodes.read(at)? {
                Node::Leaf { ids, next } => {
                    n += ids.len();
                    stack.extend((next != NIL).then_some(next));
                }
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(n)
    }

    pub(super) fn rebalance(
        &mut self,
        path: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
    ) -> Result<(), ENNError> {
        let mut below = self.count(path[path.len() - 1])?;
        for (i, w) in path.windows(2).enumerate().rev() {
            let Node::Split { left, right, .. } = self.nodes.read(w[0])? else {
                return Err(io_error("forest path is corrupt"));
            };
            let sibling = if left == w[1] { right } else { left };
            let size = below + self.count(sibling)?;
            if below.max(size - below) as f64 > alpha * size as f64 {
                return self.rebuild(path[i], rows, inv_scale);
            }
            below = size;
        }
        Ok(())
    }

    pub(crate) fn rebuild(
        &mut self,
        slot: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<(), ENNError> {
        let (mut ids, mut old) = (Vec::new(), Vec::new());
        self.gather(slot, &mut ids, &mut old)?;
        let near = self.subtree_near(slot)?;
        let rows = Rows {
            near: near.as_ref().or(rows.near),
            ..rows
        };
        let n = ids.len() as u64;
        let before = self.stats.gathered;
        self.stats.rebuild_store += uncovered(rows, &ids);
        let (f0, t0) = (
            super::forest_order::diag_faults(),
            std::time::Instant::now(),
        );
        if self.split.par_above > 0 && ids.len() >= self.split.par_above as usize {
            let plan = self.plan_subtree(slot, ids, rows, inv_scale)?;
            self.write_plan(slot, plan)?;
            self.seed_from_rows(slot, rows)?;
        } else {
            self.build(slot, ids, rows, inv_scale)?;
        }
        self.stats.rebuild_ns += t0.elapsed().as_nanos() as u64;
        self.stats.rebuild_flt += super::forest_order::diag_faults().saturating_sub(f0);
        self.nodes.free(&old[1..])?;
        self.stats.rebuild_gathered += self.stats.gathered - before;
        self.stats.rebuilds += 1;
        self.stats.rebuilt_rows += n;
        self.stats.max_rebuild_rows = self.stats.max_rebuild_rows.max(n);
        Ok(())
    }
}
