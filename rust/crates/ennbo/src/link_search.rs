use std::path::{Path, PathBuf};

pub fn emit_link_search(p: &str) {
    println!("cargo:rustc-link-search=native={p}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
}

pub fn has_faiss_c(dir: &Path) -> bool {
    dir.join("libfaiss_c.dylib").exists() || dir.join("libfaiss_c.so").exists()
}

pub fn has_blas_for_link(dir: &Path) -> bool {
    dir.join("libblas.so").exists()
        || dir.join("libopenblas.so").exists()
        || dir.join("libopenblas.so.0").exists()
}

pub fn openblas_for_link(dir: &Path) -> Option<PathBuf> {
    ["libopenblas.so", "libopenblas.so.0"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
}

pub fn emit_openblas_link(dir: &Path) {
    if let Some(openblas) = openblas_for_link(dir) {
        println!("cargo:rustc-link-arg={}", openblas.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

pub fn emit_blas_lapack_link_search_linux() {
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if has_blas_for_link(&lib) {
            println!(
                "cargo:rustc-link-search=native={}",
                lib.to_str().expect("utf-8 conda lib path")
            );
            emit_openblas_link(&lib);
        }
    }
    for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if has_blas_for_link(Path::new(p)) {
            println!("cargo:rustc-link-search=native={p}");
            emit_openblas_link(Path::new(p));
        }
    }
}

pub fn emit_faiss_link_search() {
    println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if let Ok(dir) = std::env::var("FAISS_LIB_DIR") {
        emit_link_search(&dir);
        return;
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if has_faiss_c(&lib) {
            emit_link_search(lib.to_str().expect("utf-8 conda lib path"));
            return;
        }
    }
    if cfg!(target_os = "macos") {
        for p in ["/opt/homebrew/opt/faiss/lib", "/usr/local/opt/faiss/lib"] {
            if has_faiss_c(Path::new(p)) {
                emit_link_search(p);
                return;
            }
        }
    }
    if cfg!(target_os = "linux") {
        for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
            if has_faiss_c(Path::new(p)) {
                emit_link_search(p);
                return;
            }
        }
        emit_blas_lapack_link_search_linux();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn has_faiss_c_empty_dir() {
        let dir = std::env::temp_dir().join(format!("enn_faiss_check_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!has_faiss_c(Path::new(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn has_blas_for_link_empty_dir() {
        let dir = std::env::temp_dir().join(format!("enn_blas_link_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!has_blas_for_link(Path::new(&dir)));
        assert!(openblas_for_link(Path::new(&dir)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_faiss_link_search_smoke() {
        emit_faiss_link_search();
    }
}
