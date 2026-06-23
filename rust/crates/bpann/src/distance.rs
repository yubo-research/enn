use ndarray::ArrayView1;

pub fn row_sq_l2(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    scale_x: bool,
    x_scale: ArrayView1<f64>,
) -> f64 {
    let acc = if scale_x {
        x.iter()
            .zip(y.iter())
            .zip(x_scale.iter())
            .fold(0.0, |acc, ((&xi, &yi), &sc)| {
                let d = xi / sc - yi / sc;
                acc + d * d
            })
    } else {
        x.iter()
            .zip(y.iter())
            .fold(0.0, |acc, (&xi, &yi)| {
                let d = xi - yi;
                acc + d * d
            })
    };
    acc.max(0.0)
}

pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub fn bpann_row_to_f32(row: &[f64], scale_x: bool, x_scale: &[f64], out: &mut Vec<f32>) {
    out.clear();
    if scale_x {
        out.extend(
            row.iter()
                .zip(x_scale.iter())
                .map(|(&v, &s)| (v / s) as f32),
        );
    } else {
        out.extend(row.iter().map(|&v| v as f32));
    }
}

pub fn batched_sq_l2_f32(query: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().map(|r| l2_sq_f32(query, r)).collect()
}

pub fn batched_sq_l2_f64_rows(
    query: &[f64],
    train_x: &crate::mmap_store::MmapColumnStore,
    row_ids: &[u32],
    scale_x: bool,
    x_scale: &[f64],
) -> Result<Vec<f64>, crate::error::BpannError> {
    let mut out = Vec::with_capacity(row_ids.len());
    for &id in row_ids {
        let row = train_x.mmap_row_slice(id as usize)?;
        out.push(row_sq_l2(
            ndarray::ArrayView1::from(query),
            ndarray::ArrayView1::from(row),
            scale_x,
            ndarray::ArrayView1::from(x_scale),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod kiss_coverage_tests {
    use crate::mmap_store::MmapColumnStore;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn distance_units_are_linked() {
        let mut buf = Vec::new();
        crate::distance::bpann_row_to_f32(&[1.0, 2.0], false, &[1.0, 1.0], &mut buf);
        assert_eq!(buf, vec![1.0, 2.0]);
        assert!((crate::distance::l2_sq_f32(&[0.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        let a = array![0.0, 0.0];
        let b = array![1.0, 0.0];
        let s = array![1.0, 1.0];
        let _ = crate::distance::row_sq_l2(a.view(), b.view(), false, s.view());
        let dir = TempDir::new().unwrap();
        let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store
            .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        let _ = crate::distance::batched_sq_l2_f32(&[0.0, 0.0], &[vec![1.0, 0.0]]);
        let _ = crate::distance::batched_sq_l2_f64_rows(
            &[0.0, 0.0],
            &store,
            &[0, 1],
            false,
            &[1.0, 1.0],
        )
        .unwrap();
    }
}
