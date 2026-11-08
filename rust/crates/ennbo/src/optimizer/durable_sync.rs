use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::error::ENNError;

const PASS_SPACING: Duration = Duration::from_millis(100);

#[derive(Default)]
struct Progress {
    requested: u64,
    completed: u64,
    error: Option<String>,
    urgent: bool,
    stop: bool,
}

impl Progress {
    fn check(&self) -> Result<(), ENNError> {
        match &self.error {
            Some(message) => Err(ENNError::InvalidParameter(format!(
                "fsync of work_dir failed: {message}"
            ))),
            None => Ok(()),
        }
    }
}

#[derive(Default)]
struct Shared {
    progress: Mutex<Progress>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Progress> {
        self.progress.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, Progress>) -> MutexGuard<'a, Progress> {
        self.changed
            .wait(guard)
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn wait_for<'a>(
        &self,
        guard: MutexGuard<'a, Progress>,
        timeout: Duration,
    ) -> MutexGuard<'a, Progress> {
        self.changed
            .wait_timeout(guard, timeout)
            .unwrap_or_else(PoisonError::into_inner)
            .0
    }
}

fn run_passes(shared: &Shared, spacing: Duration, mut sync: impl FnMut() -> io::Result<()>) {
    let mut progress = shared.lock();
    let mut next_pass = Instant::now();
    loop {
        while progress.requested == progress.completed && !progress.stop {
            progress = shared.wait(progress);
        }
        if progress.requested == progress.completed {
            return;
        }
        while !progress.urgent && !progress.stop {
            let left = next_pass.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            progress = shared.wait_for(progress, left);
        }
        let target = progress.requested;
        progress.urgent = false;
        drop(progress);
        let result = sync();
        next_pass = Instant::now() + spacing;
        progress = shared.lock();
        match result {
            Ok(()) => progress.completed = target,
            Err(e) => progress.error = Some(e.to_string()),
        }
        shared.changed.notify_all();
        if progress.error.is_some() {
            return;
        }
    }
}

pub(super) struct BackgroundSync {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl BackgroundSync {
    fn spawn(spacing: Duration, sync: impl FnMut() -> io::Result<()> + Send + 'static) -> Self {
        let shared = Arc::new(Shared::default());
        let for_worker = Arc::clone(&shared);
        let worker = std::thread::spawn(move || run_passes(&for_worker, spacing, sync));
        Self {
            shared,
            worker: Some(worker),
        }
    }

    pub(super) fn for_work_dir(work_dir: PathBuf, last_file: &'static str) -> Self {
        Self::spawn(PASS_SPACING, move || sync_work_dir(&work_dir, last_file))
    }

    pub(super) fn request(&self) -> Result<(), ENNError> {
        let mut progress = self.shared.lock();
        progress.check()?;
        progress.requested += 1;
        self.shared.changed.notify_all();
        Ok(())
    }

    pub(super) fn wait_until_on_disk(&self) -> Result<(), ENNError> {
        let mut progress = self.shared.lock();
        let target = progress.requested;
        if progress.completed < target {
            progress.urgent = true;
            self.shared.changed.notify_all();
        }
        while progress.completed < target && progress.error.is_none() {
            progress = self.shared.wait(progress);
        }
        progress.check()
    }
}

impl Drop for BackgroundSync {
    fn drop(&mut self) {
        self.shared.lock().stop = true;
        self.shared.changed.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn is_staged(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "tmp")
}

fn tolerate_missing(result: io::Result<()>) -> io::Result<()> {
    match result {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

fn sync_order(work_dir: &Path, last_file: &str) -> io::Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut files = Vec::new();
    let mut dirs = vec![work_dir.to_path_buf()];
    let mut next = 0;
    while next < dirs.len() {
        let dir = dirs[next].clone();
        next += 1;
        let entries = match std::fs::read_dir(&dir) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            entries => entries?,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                dirs.push(path);
            } else if !is_staged(&path) {
                files.push(path);
            }
        }
    }
    let last = work_dir.join(last_file);
    files.sort_by_key(|path| *path == last);
    dirs.reverse();
    Ok((files, dirs))
}

fn sync_file(path: &Path) -> io::Result<()> {
    tolerate_missing(File::open(path).and_then(|file| file.sync_all()))
}

pub(super) fn sync_dir(path: &Path) -> io::Result<()> {
    if cfg!(not(unix)) {
        return Ok(());
    }
    match sync_file(path) {
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => Ok(()),
        other => other,
    }
}

fn sync_work_dir(work_dir: &Path, last_file: &str) -> io::Result<()> {
    let (files, dirs) = sync_order(work_dir, last_file)?;
    files.iter().try_for_each(|path| sync_file(path))?;
    dirs.iter().try_for_each(|path| sync_dir(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    const HOUR: Duration = Duration::from_secs(3600);

    #[test]
    fn requests_never_wait_for_a_running_pass_and_collapse_into_one() {
        let (release, released) = mpsc::channel::<()>();
        let passes = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&passes);
        let sync = BackgroundSync::spawn(Duration::ZERO, move || {
            counted.fetch_add(1, Ordering::SeqCst);
            released.recv().ok();
            Ok(())
        });
        sync.request().unwrap();
        while passes.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        for _ in 0..100 {
            sync.request().unwrap();
        }
        release.send(()).unwrap();
        release.send(()).unwrap();
        sync.wait_until_on_disk().unwrap();
        assert_eq!(passes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn dropping_finishes_requested_work() {
        let passes = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&passes);
        let sync = BackgroundSync::spawn(HOUR, move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        sync.request().unwrap();
        sync.wait_until_on_disk().unwrap();
        sync.request().unwrap();
        drop(sync);
        assert_eq!(passes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_later_pass_is_spaced_out_unless_someone_is_waiting() {
        let passes = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&passes);
        let sync = BackgroundSync::spawn(HOUR, move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        sync.request().unwrap();
        while passes.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        sync.wait_until_on_disk().unwrap();
        sync.request().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(passes.load(Ordering::SeqCst), 1);
        sync.wait_until_on_disk().unwrap();
        assert_eq!(passes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_failure_is_reported_to_every_later_caller() {
        let sync = BackgroundSync::spawn(Duration::ZERO, || Err(io::Error::other("disk gone")));
        sync.request().unwrap();
        for _ in 0..2 {
            let waited = sync.wait_until_on_disk().err().unwrap();
            assert!(waited.to_string().contains("disk gone"), "{waited}");
        }
        assert!(sync.request().is_err());
    }

    #[test]
    fn the_snapshot_follows_every_row_file_and_staged_files_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("index")).unwrap();
        for name in [
            "a_state.json",
            "train_x.bin",
            "num_obs.bin",
            "a_state.json.tmp",
            "index/pages.bin",
            "index/pages.bin.tmp",
        ] {
            std::fs::write(root.join(name), b"x").unwrap();
        }
        let (files, dirs) = sync_order(root, "a_state.json").unwrap();
        assert_eq!(files.len(), 4);
        assert_eq!(files.last().unwrap(), &root.join("a_state.json"));
        assert!(files.contains(&root.join("index/pages.bin")));
        assert_eq!(dirs, vec![root.join("index"), root.to_path_buf()]);
    }

    #[test]
    fn syncing_a_real_tree_succeeds_and_a_missing_one_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("state.json"), b"{}").unwrap();
        sync_work_dir(dir.path(), "state.json").unwrap();
        sync_work_dir(&dir.path().join("gone"), "state.json").unwrap();
    }
}
