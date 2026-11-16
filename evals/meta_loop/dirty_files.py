import ctypes
import json
import os
import sys
import time

NR_CACHESTAT = 451
MIB = 4096 / 2**20
libc = ctypes.CDLL(None, use_errno=True)


class Range(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint64), ("len", ctypes.c_uint64)]


class Stat(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_uint64)
        for n in ("cache", "dirty", "writeback", "evicted", "recently_evicted")
    ]


def cachestat(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    r, s = Range(0, 0), Stat()
    rc = libc.syscall(NR_CACHESTAT, fd, ctypes.byref(r), ctypes.byref(s), 0)
    os.close(fd)
    return s if rc == 0 else None


def read(path, mode="r"):
    try:
        with open(path, mode) as f:
            return f.read()
    except OSError:
        return None


def scope_cgroup(pid, pattern):
    cmd = read(f"/proc/{pid}/cmdline", "rb")
    if not cmd or pattern.encode() not in cmd or b"dirty_files" in cmd:
        return None
    cg = (read(f"/proc/{pid}/cgroup") or "::").split(":")[2].strip()
    return "/sys/fs/cgroup" + cg if cg.endswith(".scope") and "/run-" in cg else None


def cgroup_of(pattern):
    for p in filter(str.isdigit, os.listdir("/proc")):
        cg = scope_cgroup(p, pattern)
        if cg:
            return cg
    return None


def files_state(work):
    out = {}
    for root, _, names in os.walk(work):
        for n in names:
            p = os.path.join(root, n)
            s = cachestat(p)
            if s is not None:
                out[os.path.relpath(p, work)] = [
                    round(s.cache * MIB, 2),
                    round(s.dirty * MIB, 2),
                    round(s.writeback * MIB, 2),
                ]
    return out


def cg_state(cg):
    txt = read(cg + "/memory.stat") if cg else None
    if not txt:
        return {}
    st = dict(l.split() for l in txt.splitlines())
    return {
        "cg_dirty": int(st["file_dirty"]) / 2**20,
        "cg_wb": int(st["file_writeback"]) / 2**20,
        "cg_file": int(st["file"]) / 2**20,
    }


def main():
    work, out = sys.argv[1], sys.argv[2]
    hz = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    pat = sys.argv[4] if len(sys.argv) > 4 else "stall_probe.py"
    seen, cg = False, None
    with open(out, "w") as f:
        while os.path.isdir(work) or not seen:
            if os.path.isdir(work):
                seen, cg = True, cg or cgroup_of(pat)
                rec = {"t": time.time(), "files": files_state(work), **cg_state(cg)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
            time.sleep(1 / hz)


if __name__ == "__main__":
    main()
