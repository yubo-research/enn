use std::path::{Path, PathBuf};

fn emit_link_search(p: &str) {
    println!("cargo:rustc-link-search=native={p}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
}

fn has_faiss_c(dir: &Path) -> bool {
    dir.join("libfaiss_c.dylib").exists() || dir.join("libfaiss_c.so").exists()
}

fn has_blas_for_link(dir: &Path) -> bool {
    dir.join("libblas.so").exists()
        || dir.join("libopenblas.so").exists()
        || dir.join("libopenblas.so.0").exists()
}

fn openblas_for_link(dir: &Path) -> Option<PathBuf> {
    ["libopenblas.so", "libopenblas.so.0"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
}

fn emit_openblas_link(dir: &Path) {
    if let Some(openblas) = openblas_for_link(dir) {
        println!("cargo:rustc-link-arg={}", openblas.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

/// `faiss-sys` (static Linux) links `-lblas -llapack`; ensure the linker sees them
/// (conda `OpenBLAS` provides `libblas.so`, Debian needs `libblas-dev` / `libopenblas0`).
fn emit_blas_lapack_link_search_linux() {
    // Only link-search here: `rustc-link-arg` rpath from a dependency build script
    // does not apply to the final cdylib (`enn-py` adds rpath via its own `build.rs`).
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

fn main() {
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
