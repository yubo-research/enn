import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def find(pattern):
    me = str(Path("/proc/self").resolve().name)
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or p.name == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if (
            pattern in cmd
            and "wchan_sample" not in cmd
            and "python" in cmd.split(" ")[0]
        ):
            return p
    return None


def lines(path):
    try:
        return sum(1 for _ in open(path))
    except OSError:
        return 0


def main():
    pattern, prog, out = sys.argv[1:4]
    hz = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
    p = None
    while p is None:
        p = find(pattern)
        time.sleep(0.2)
    cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    print("sampling", p.name, cmd, flush=True)
    bins, b, last = defaultdict(Counter), 0, 0.0
    while True:
        try:
            w = (p / "wchan").read_text() or "0"
            st = (p / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            break
        now = time.monotonic()
        if now - last > 1.0:
            b, last = lines(prog), now
        bins[b][f"{st}:{w}"] += 1
        time.sleep(1.0 / hz)
    Path(out).write_text(json.dumps({k: dict(v) for k, v in bins.items()}, indent=1))


if __name__ == "__main__":
    main()
