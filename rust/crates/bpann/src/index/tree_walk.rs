use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::index::tree_codes_search::CodedScan;
use crate::index::tree_search::FrontierItem;
use crate::index::tree_store::TreeStore;

pub(super) use crate::index::tree_cold::ColdProbe;

pub(super) struct Lookahead {
    pending: VecDeque<Reverse<FrontierItem>>,
    depth: usize,
    from_pending: bool,
}

impl Lookahead {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(depth),
            depth,
            from_pending: false,
        }
    }

    pub(super) fn pop(
        &mut self,
        frontier: &mut BinaryHeap<Reverse<FrontierItem>>,
    ) -> Option<Reverse<FrontierItem>> {
        self.from_pending = !self.pending.is_empty();
        self.pending.pop_front().or_else(|| frontier.pop())
    }

    pub(super) fn before_leaf(
        &mut self,
        s: &TreeStore,
        frontier: &mut BinaryHeap<Reverse<FrontierItem>>,
        coded: Option<&mut CodedScan>,
    ) {
        if self.depth == 0 {
            return;
        }
        while self.pending.len() < self.depth {
            match frontier.peek() {
                Some(Reverse(item)) if s.is_leaf(item.3) => {}
                _ => break,
            }
            if let Some(item) = frontier.pop() {
                s.advise_codes(item.0 .3);
                self.pending.push_back(item);
            }
        }
        if let Some(scan) = coded {
            scan.advised = self.from_pending;
        }
    }
}

pub(super) fn pad_query(query: &[f32], dim: usize) -> std::borrow::Cow<'_, [f32]> {
    if dim <= query.len() {
        return std::borrow::Cow::Borrowed(query);
    }
    let mut padded = query.to_vec();
    padded.resize(dim, 0.0);
    std::borrow::Cow::Owned(padded)
}
