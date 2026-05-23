use std::path::{Path, PathBuf};
use std::process::Command;

pub fn blas_libs_present(dir: &Path) -> bool {
    ["libblas.so", "libopenblas.so", "libopenblas.so.0"]
        .iter()
        .any(|name| dir.join(name).exists())
}

pub fn install_patchelf_if_needed() {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../ennbo/cmake/install_patchelf_root.sh");
    let status = Command::new("bash")
        .arg(script)
        .status()
        .unwrap_or_else(|e| panic!("install_patchelf_root.sh: {e}"));
    assert!(
        status.success(),
        "install_patchelf_root.sh failed: {status}"
    );
}

pub fn emit_linux_rpath_link_args() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn blas_libs_present_empty_dir() {
        let dir = std::env::temp_dir().join(format!("enn_blas_check_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!blas_libs_present(Path::new(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_linux_rpath_link_args_smoke() {
        emit_linux_rpath_link_args();
    }
}
