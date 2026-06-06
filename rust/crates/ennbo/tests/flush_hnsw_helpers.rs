use ennbo::backend::DiskEnnBackend;
use ennbo::disk_hnsw::flush::{try_schedule_background_flush, BackgroundFlushState};
use ennbo::disk_hnsw::DiskHnswEnnBackend;
use ndarray::array;
use std::sync::{Arc, Mutex};

pub fn hnsw_arc(backend: DiskHnswEnnBackend) -> Arc<Mutex<DiskEnnBackend>> {
    Arc::new(Mutex::new(DiskEnnBackend::Hnsw(backend)))
}

pub fn flush_arc(arc: &Arc<Mutex<DiskEnnBackend>>) -> Arc<Mutex<BackgroundFlushState>> {
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(b) = &*guard;
    b.flush_arc()
}

pub fn schedule_background_flush(arc: &Arc<Mutex<DiskEnnBackend>>) {
    let (flush, should) = {
        let guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(b) = &*guard;
        (
            b.flush_arc(),
            !b.is_index_stale() && b.pending_rows() >= b.pending_flush_threshold(),
        )
    };
    if should {
        try_schedule_background_flush(&flush, Arc::clone(arc)).unwrap();
    }
}

pub fn append_row(backend: &mut DiskHnswEnnBackend, i: f64) {
    backend
        .append_rows(
            &array![[i, i]].view(),
            &array![[i]].view(),
            None,
        )
        .unwrap();
}
