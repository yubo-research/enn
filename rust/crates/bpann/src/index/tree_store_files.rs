use std::fs::{self, File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::MmapMut;

use crate::error::BpannError;

static SCRATCH_SEQ: AtomicUsize = AtomicUsize::new(0);

pub(super) fn io_err(e: std::io::Error) -> BpannError {
    BpannError::InvalidParameter(e.to_string())
}

pub(super) fn scratch_file(dir: &Path, tag: &str) -> Result<File, BpannError> {
    fs::create_dir_all(dir).map_err(io_err)?;
    let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("{tag}.{}.{seq}.scratch", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(io_err)?;
    let _ = fs::remove_file(&path);
    Ok(file)
}

pub(super) fn map(file: &File, bytes: usize) -> Result<MmapMut, BpannError> {
    file.set_len(bytes as u64).map_err(io_err)?;
    let m = unsafe { MmapMut::map_mut(file) }.map_err(io_err)?;
    let _ = m.advise(memmap2::Advice::Random);
    Ok(m)
}

pub const META_FILE: &str = "tree_meta.bin";
pub const SLOTS_FILE: &str = "tree_slots.bin";
pub const CODES_FILE: &str = "tree_codes.bin";
pub const STAMP_FILE: &str = "tree_state.json";

pub(super) fn named_file(dir: &Path, name: &str) -> Result<File, BpannError> {
    fs::create_dir_all(dir).map_err(io_err)?;
    let path = dir.join(name);
    let _ = fs::remove_file(&path);
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_err)
}

#[cfg(unix)]
pub(super) fn same_file(file: &File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (file.metadata(), fs::metadata(path)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
pub(super) fn same_file(_file: &File, _path: &Path) -> bool {
    false
}

pub(super) fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}
