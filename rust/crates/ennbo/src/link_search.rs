use std::path::{Path, PathBuf};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn has_blas_for_link_empty_dir() {
        let dir = std::env::temp_dir().join(format!("enn_blas_link_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!has_blas_for_link(Path::new(&dir)));
        assert!(openblas_for_link(Path::new(&dir)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
