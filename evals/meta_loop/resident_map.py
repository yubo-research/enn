import ctypes
import mmap
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = 4096


def mapped_pages(fd, size):
    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap")
    vec = (ctypes.c_ubyte * ((size + PAGE - 1) // PAGE))()
    rc = libc.mincore(addr, size, vec)
    libc.munmap(addr, size)
    if rc != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    return bytes(v & 1 for v in vec)


def resident(path):
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    out = mapped_pages(fd, size) if size else b""
    os.close(fd)
    return out


def main():
    path = sys.argv[1]
    bins = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    r = resident(path)
    n, mib = len(r), PAGE / 2**20
    print(f"{path}: {n * mib:.1f} MiB, resident {sum(r) * mib:.1f} MiB")
    for k in (16, 64, 256):
        p = (k << 20) // PAGE
        print(f"  last {k} MiB: {sum(r[max(0, n - p) :]) * mib:.1f} MiB resident")
    step = max(1, n // bins)
    print(
        "  slices:", [round(sum(r[i : i + step]) * mib, 1) for i in range(0, n, step)]
    )


if __name__ == "__main__":
    main()
