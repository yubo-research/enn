use std::path::{Path, PathBuf};
use std::process::Command;

fn blas_libs_present(dir: &Path) -> bool {
    ["libblas.so", "libopenblas.so", "libopenblas.so.0"]
        .iter()
        .any(|name| dir.join(name).exists())
}

fn install_patchelf_if_needed() {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../ennbo/cmake/install_patchelf_root.sh");
    let status = Command::new("bash")
        .arg(script)
        .status()
        .unwrap_or_else(|e| panic!("install_patchelf_root.sh: {e}"));
    assert!(status.success(), "install_patchelf_root.sh failed: {status}");
}

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if !cfg!(target_os = "linux") {
        return;
    }
    install_patchelf_if_needed();
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if blas_libs_present(&lib) {
            let p = lib.to_string_lossy();
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{p}");
        }
    }
    for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if blas_libs_present(Path::new(p)) {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{p}");
        }
    }
}
