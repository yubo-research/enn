use super::*;
use crate::index::search::MmapSearchStore;
use crate::index::sync::IncrementalIndex;
use crate::index::tree_search::search_tree;
use crate::mmap_store::MmapColumnStore;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn row(r: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
        .collect()
}

#[test]
fn eight_bit_codes_reencode_losslessly_from_their_decoding() {
    let d = 37;
    for r in 0..200 {
        let x = row(r, d);
        let mut a = vec![0u8; code_row_bytes(d)];
        encode_row(&x, &mut a);
        let (scale, _, c) = decode_header(&a);
        let y: Vec<f32> = c.iter().map(|&q| scale * f32::from(q)).collect();
        let mut b = vec![0u8; code_row_bytes(d)];
        encode_row(&y, &mut b);
        assert_eq!(a[8..], b[8..], "row {r}");
        assert!((scale - decode_header(&b).0).abs() <= scale * 1e-6);
    }
}

fn assert_buffers_covered(t: &Tree) {
    let s = &t.store;
    let half = 0.5 * (s.num_dim() as f32).sqrt();
    for p in 0..t.num_pages() as u32 {
        let Some(g) = s.parent(p).filter(|_| !s.is_free(p) && s.pbuf_len(p) > 0) else {
            continue;
        };
        let slot = s.ids(g).iter().position(|&k| k == p).unwrap();
        let dd = s.num_dim();
        let c = &s.block(g)[slot * dd..(slot + 1) * dd];
        let mut v = Vec::new();
        for (_, code) in s.pbuf_records(p) {
            s.decode_code(code, &mut v);
            let slack = half * decode_header(code).0;
            assert!(
                dist(&v, c) - slack <= s.radius(p) * 1.0001 + 1e-5,
                "page {p}"
            );
        }
    }
}

#[test]
fn buffered_tree_holds_every_row_and_radii_cover_buffers() {
    set_pbuf_rows(50);
    let (n, d) = (6000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let mut t = Tree::new_tree(0, &row(0, d), d, dir.path().join("i")).unwrap();
    for r in 1..n {
        t.insert_row(r as u32, &row(r, d)).unwrap();
    }
    set_pbuf_rows(0);
    assert!(t.store.pbuf_on());
    assert_eq!(t.header.indexed_rows, n);
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    let buffered = t.pbuf_row_ids().len();
    assert!(buffered > 0 && buffered < n, "{buffered} buffered");
    assert_buffers_covered(&t);
    let mut c = t.try_clone().unwrap();
    assert_eq!(c.pbuf_row_ids().len(), buffered);
    c.pbuf_flush_all().unwrap();
    assert!(c.pbuf_row_ids().is_empty());
    assert_eq!(c.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
}

#[test]
fn buffered_search_finds_true_neighbours() {
    let (n, d) = (4000usize, 3usize);
    let dir = tempfile::TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let x = ndarray::Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
    let mut store =
        MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), d, None).unwrap();
    store.mmap_append(&x.view()).unwrap();
    set_pbuf_rows(200);
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    for end in [1, 700, n] {
        idx.ensure_sync_for_backend(&store, d, false, &[1.0; 3], end)
            .unwrap();
    }
    set_pbuf_rows(0);
    let t = idx.tree.as_ref().unwrap();
    assert!(!t.pbuf_row_ids().is_empty());
    let s = MmapSearchStore {
        train_x: &store,
        scale_x: false,
        x_scale: &[1.0; 3],
    };
    let mut missed = 0usize;
    for _ in 0..20 {
        let q: Vec<f32> = (0..d).map(|_| rng.gen::<f32>()).collect();
        let got = search_tree(t, &q, 7, usize::MAX, &s).unwrap();
        let mut all: Vec<(u32, f32)> = (0..n as u32)
            .map(|r| {
                let xr: Vec<f32> = x.row(r as usize).iter().map(|&v| v as f32).collect();
                (r, crate::distance::l2_sq_f32(&q, &xr))
            })
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        missed += all[..7]
            .iter()
            .filter(|w| !got.iter().any(|g| g.0 == w.0))
            .count();
    }
    assert!(missed <= 2, "{missed} of 140 true neighbours missed");
}
