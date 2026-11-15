import json
import os
import pathlib
import sys
import time

import numpy as np

MODE, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAXN = int(float(sys.argv[4]))
CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
F = int(sys.argv[6]) if len(sys.argv) > 6 else 16
DEV = sys.argv[7] if len(sys.argv) > 7 else "nvme0n1p1"
MODE2 = sys.argv[8] if len(sys.argv) > 8 else ""
BOUND = MODE2 in ("sub", "seg")
SEG = MODE2 == "seg"
IDX = MODE2 == "idx"
MRG = MODE2 == "mrg"
FS = 64
BOUND = BOUND or IDX or MRG
PG = 4096
RB, LEAF_ROWS, LEAF_BYTES, HDR = 112, 48, 8192, 64
BATCH = 4096
LMAX = MAXN // LEAF_ROWS + 1
os.makedirs(WORK, exist_ok=True)
pat = np.random.default_rng(1).integers(0, 256, RB, dtype=np.uint8)


def disk():
    for line in pathlib.Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if f[2] == DEV:
            return int(f[5]) * 512, int(f[9]) * 512
    raise SystemExit("device not found")


def mkrows(keys):
    r = np.empty((len(keys), RB), np.uint8)
    r[:] = pat
    r[:, :4] = keys.astype("<u4").view(np.uint8).reshape(-1, 4)
    return r


leaf_fd = os.open(f"{WORK}/leaf.bin", os.O_RDWR | os.O_CREAT | os.O_TRUNC)
leaf_cnt = np.zeros(LMAX, np.int32)


def slot(j):
    return max(8, -(-3 * capj(j) // F))


def capj(j):
    return min(CAP, F**j * LEAF_ROWS) if BOUND else CAP


if MODE == "buffered":
    levels = 1
    while F**levels < LMAX:
        levels += 1
    top = levels
    nnodes = {j: -(-LMAX // F**j) for j in range(1, top)}
    buf_fd = os.open(f"{WORK}/buf.bin", os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    base, off = {}, 0
    for j in range(1, top):
        base[j] = off
        off += nnodes[j] * (F * slot(j) if SEG else 2 * capj(j)) * RB
        off += nnodes[j] * F * PG if IDX else 0
    cnt = {j: np.zeros(nnodes[j], np.int32) for j in range(1, top)}
    boff = {j: np.zeros(nnodes[j], np.int64) for j in range(1, top)}
    runs = {j: {} for j in range(1, top)}
    segcnt = (
        {j: np.zeros((nnodes[j], F), np.int32) for j in range(1, top)} if SEG else {}
    )
    root = np.empty((0, RB), np.uint8)


def keys_of(rows):
    return rows[:, :4].copy().view("<u4").ravel()


def to_leaves(rows):
    keys = keys_of(rows)
    order = np.argsort(keys, kind="stable")
    rows, keys = rows[order], keys[order]
    cuts = np.flatnonzero(np.diff(keys)) + 1
    for g in np.split(np.arange(len(keys)), cuts):
        l = int(keys[g[0]])
        data = rows[g]
        raw = os.pread(leaf_fd, LEAF_BYTES, l * LEAF_BYTES)
        n = min(len(g), LEAF_ROWS)
        c = int(leaf_cnt[l]) % LEAF_ROWS
        blob = bytearray(raw.ljust(LEAF_BYTES, b"\0"))
        for r in data[-n:]:
            blob[HDR + c * RB : HDR + (c + 1) * RB] = r.tobytes()
            c = (c + 1) % LEAF_ROWS
        os.pwrite(leaf_fd, bytes(blob), l * LEAF_BYTES)
        leaf_cnt[l] += len(g)


def flush(j, rows):
    if j == 1:
        to_leaves(rows)
        return
    keys = keys_of(rows)
    child = (keys // F ** (j - 1)).astype(np.int64)
    order = np.argsort(child, kind="stable")
    rows, child = rows[order], child[order]
    cuts = np.flatnonzero(np.diff(child)) + 1
    for g in np.split(np.arange(len(child)), cuts):
        i = int(child[g[0]])
        k = j - 1
        if SEG:
            append_seg(k, i, rows[g])
            continue
        if MRG:
            merge_into(k, i, rows[g])
            continue
        write_run(k, i, rows[g])
        cnt[k][i] += len(g)
        if cnt[k][i] >= capj(k):
            n = int(cnt[k][i])
            raw = read_back(k, i, n)
            cnt[k][i] = 0
            flush(k, np.frombuffer(raw, np.uint8).reshape(n, RB))


fences = {}


def merge_into(k, i, part):
    o = base[k] + i * 2 * capj(k) * RB
    c = int(cnt[k][i])
    old = (
        np.frombuffer(os.pread(buf_fd, c * RB, o), np.uint8).reshape(c, RB)
        if c
        else part[:0]
    )
    allr = np.concatenate([old, part])
    allr = allr[np.argsort(keys_of(allr), kind="stable")]
    n = len(allr)
    if n >= capj(k):
        cnt[k][i] = 0
        fences.pop((k, i), None)
        flush(k, allr)
        return
    os.pwrite(buf_fd, allr.tobytes(), o)
    cnt[k][i] = n
    fences[(k, i)] = keys_of(allr)[::FS].copy()


def write_run(k, i, part):
    if not IDX:
        o = base[k] + (i * 2 * capj(k) + int(cnt[k][i])) * RB
        os.pwrite(buf_fd, part.tobytes(), o)
        return
    o = base[k] + i * (2 * capj(k) * RB + F * PG) + int(boff[k][i])
    os.pwrite(buf_fd, bytes(PG) + part.tobytes(), o)
    runs[k].setdefault(i, []).append((o, len(part)))
    boff[k][i] += PG + len(part) * RB


def read_back(k, i, n):
    if not IDX:
        return os.pread(buf_fd, n * RB, base[k] + i * 2 * capj(k) * RB)
    parts = [
        np.frombuffer(os.pread(buf_fd, m * RB, o0 + PG), np.uint8).reshape(m, RB)
        for o0, m in runs[k].pop(i)
    ]
    boff[k][i] = 0
    return np.concatenate(parts).tobytes()


def flush_node(k, i):
    parts = []
    for sg in range(F):
        c = int(segcnt[k][i, sg])
        if c:
            o = base[k] + (i * F + sg) * slot(k) * RB
            parts.append(
                np.frombuffer(os.pread(buf_fd, c * RB, o), np.uint8).reshape(c, RB)
            )
            segcnt[k][i, sg] = 0
    cnt[k][i] = 0
    flush(k, np.concatenate(parts))


def append_seg(k, i, rows):
    sg_of = (keys_of(rows) // F ** (k - 1) % F).astype(np.int64)
    order = np.argsort(sg_of, kind="stable")
    rows, sg_of = rows[order], sg_of[order]
    cuts = np.flatnonzero(np.diff(sg_of)) + 1
    for g in np.split(np.arange(len(sg_of)), cuts):
        sg = int(sg_of[g[0]])
        for a in range(0, len(g), slot(k)):
            part = rows[g[a : a + slot(k)]]
            if segcnt[k][i, sg] + len(part) > slot(k):
                flush_node(k, i)
            o = base[k] + ((i * F + sg) * slot(k) + int(segcnt[k][i, sg])) * RB
            os.pwrite(buf_fd, part.tobytes(), o)
            segcnt[k][i, sg] += len(part)
            cnt[k][i] += len(part)
    if cnt[k][i] >= capj(k):
        flush_node(k, i)


def scan_node(l, k):
    i = l // F**k
    if MRG:
        fe = fences.get((k, i))
        if fe is None:
            return 0
        a = max(int(np.searchsorted(fe, l, "left")) - 1, 0) * FS
        b = min(int(np.searchsorted(fe, l, "right")) * FS, int(cnt[k][i]))
        if b > a:
            os.pread(buf_fd, (b - a) * RB, base[k] + (i * 2 * capj(k) + a) * RB)
        return 0
    if IDX:
        for o0, m in runs[k].get(i, ()):
            os.pread(buf_fd, PG, o0)
            os.pread(buf_fd, PG, o0 + PG + int(rng_q.integers(0, m)) * RB)
        return 0
    c = int(cnt[k][i])
    if not c or SEG:
        return 0
    raw = os.pread(buf_fd, c * RB, base[k] + i * 2 * capj(k) * RB)
    ks = np.frombuffer(raw, np.uint8).reshape(c, RB)[:, :4].copy().view("<u4")
    return int((ks == l).sum())


NPR = [0]
_pread = os.pread


def cpread(fd, n, o):
    NPR[0] += 1
    return _pread(fd, n, o)


def query(l):
    n = 0
    if MODE == "buffered":
        n += len(root)
        for k in range(1, top):
            n += scan_node(l, k)
    os.pread(leaf_fd, LEAF_BYTES, l * LEAF_BYTES)
    return n


NQ = 300
rng_q = np.random.default_rng(3)
rng = np.random.default_rng(0)
qrng = np.random.default_rng(7)
N, t0, last = 0, time.time(), (time.time(), 0, disk())
pathlib.Path(OUT).write_text("")
while N < MAXN:
    lcur = max(1, N // LEAF_ROWS)
    keys = rng.integers(0, lcur, BATCH)
    rows = mkrows(keys)
    if MODE == "inplace":
        for l, r in zip(keys.tolist(), rows):
            base_off = l * LEAF_BYTES
            os.pread(leaf_fd, 4, base_off)
            c = int(leaf_cnt[l]) % LEAF_ROWS
            os.pwrite(leaf_fd, r.tobytes(), base_off + HDR + c * RB)
            leaf_cnt[l] += 1
    else:
        root = np.concatenate([root, rows])
        if len(root) >= CAP:
            flush(top, root)
            root = root[:0]
    N += BATCH
    now = time.time()
    if N - last[1] >= 250000 or N >= MAXN:
        r, w = disk()
        dn = N - last[1]
        add_us = (now - last[0]) / dn * 1e6
        os.pread = cpread
        NPR[0] = 0
        tq = time.time()
        for l in qrng.integers(0, max(1, N // LEAF_ROWS), NQ).tolist():
            query(l)
        q_us = (time.time() - tq) / NQ * 1e6
        preads_q = NPR[0] / NQ
        os.pread = _pread
        tq = time.time()
        for l in qrng.integers(0, max(1, N // LEAF_ROWS), NQ).tolist():
            os.pread(leaf_fd, LEAF_BYTES, l * LEAF_BYTES)
        leaf_us = (time.time() - tq) / NQ * 1e6
        now = time.time()
        rec = {
            "N": N,
            "us_per_row": add_us,
            "query_us": q_us,
            "leaf_only_us": leaf_us,
            "preads_per_query": preads_q,
            "read_B_row": (r - last[2][0]) / dn,
            "write_B_row": (w - last[2][1]) / dn,
            "t": now - t0,
        }
        with pathlib.Path(OUT).open("a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        last = (now, N, (r, w))
