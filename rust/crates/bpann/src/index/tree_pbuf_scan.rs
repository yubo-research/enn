use std::sync::atomic::Ordering;

use crate::error::BpannError;
use crate::index::tree_codes::coded_sq_l2;
use crate::index::tree_pbuf_tag::{note_scan, TagPick};
use crate::index::tree_store::TreeStore;

impl TreeStore {
    pub(super) fn pbuf_tagged_records(
        &self,
        id: u32,
    ) -> impl Iterator<Item = (u32, &[u8], u8)> + '_ {
        let n = self.pbuf_len(id);
        let (rec, cl, tagged) = (self.pbuf.rec, self.pbuf.code_len(), self.pbuf.tagged);
        let bytes: &[u8] = if n == 0 { &[] } else { self.pbuf.region(id, n) };
        bytes.chunks_exact(rec).map(move |r| {
            let tag = if tagged { r[rec - 1] } else { 0 };
            (
                u32::from_le_bytes([r[0], r[1], r[2], r[3]]),
                &r[4..4 + cl],
                tag,
            )
        })
    }

    pub(super) fn pbuf_scan(
        &self,
        id: u32,
        (query, qn2): (&[f32], f32),
        pick: TagPick,
    ) -> Vec<(u32, f32, f32)> {
        let score = |(row, code): (u32, &[u8])| {
            let (d2, e) = coded_sq_l2(query, qn2, code);
            (row, d2, e)
        };
        if let TagPick::Rows(ix) = pick {
            return ix.iter().map(|&i| score(self.pbuf_record(id, i))).collect();
        }
        self.pbuf_tagged_records(id)
            .filter(|r| pick.keeps(r.2))
            .map(|(row, code, _)| score((row, code)))
            .collect()
    }

    pub(super) fn pbuf_record(&self, id: u32, i: u32) -> (u32, &[u8]) {
        let (rec, cl) = (self.pbuf.rec, self.pbuf.code_len());
        let r = &self.pbuf.region(id, i as usize + 1)[i as usize * rec..];
        (u32::from_le_bytes([r[0], r[1], r[2], r[3]]), &r[4..4 + cl])
    }
}

pub(super) fn score_pbuf(
    s: &TreeStore,
    page_id: u32,
    (query, qn2): (&[f32], f32),
    coded: Option<&mut crate::index::tree_codes_search::CodedScan>,
    top: &mut crate::index::tree_topk::TopK,
    score: &mut impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
    pick: TagPick,
) -> Result<usize, BpannError> {
    use crate::index::tree_search::XP_LEAF_ROWS;
    let len = s.pbuf_len(page_id);
    if len == 0 {
        return Ok(0);
    }
    let visited = if matches!(pick, TagPick::Rows(_)) {
        0
    } else {
        s.advise_pbuf(page_id, 0, len);
        len
    };
    if let Some(scan) = coded {
        let n = scan.scan_pbuf(s, page_id, (query, qn2), pick, &mut |i, d2| {
            top.offer(i, d2)
        });
        XP_LEAF_ROWS.fetch_add(n as u64, Ordering::Relaxed);
        note_scan(visited, n);
        return Ok(n);
    }
    let ids: Vec<u32> = match pick {
        TagPick::Rows(ix) => ix.iter().map(|&i| s.pbuf_record(page_id, i).0).collect(),
        _ => s
            .pbuf_tagged_records(page_id)
            .filter(|r| pick.keeps(r.2))
            .map(|r| r.0)
            .collect(),
    };
    let n = ids.len();
    note_scan(visited, n);
    XP_LEAF_ROWS.fetch_add(n as u64, Ordering::Relaxed);
    for row in ids {
        if let Some(distance) = score(row, None, top.limit())? {
            top.offer(row, distance);
        }
    }
    Ok(n)
}
