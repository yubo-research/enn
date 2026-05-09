use std::path::{Path, PathBuf};

fn has_blas_for_link(dir: &Path) -> bool {
    dir.join("libblas.so").exists()
        || dir.join("libopenblas.so").exists()
        || dir.join("libopenblas.so.0").exists()
}

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if !cfg!(target_os = "linux") {
        return;
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if has_blas_for_link(&lib) {
            let p = lib.to_string_lossy();
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{}", p);
        }
    }
    for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if has_blas_for_link(Path::new(p)) {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{p}");
        }
    }
}
