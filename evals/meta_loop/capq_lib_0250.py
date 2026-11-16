import subprocess
from pathlib import Path


def fincore_mib(f):
    r = subprocess.run(
        ["fincore", "-b", "-n", "-o", "RES", str(f)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(r.stdout.split()[0]) / 2**20
    except (ValueError, IndexError):
        return 0.0


def resident(p):
    out = {}
    for f in Path(p).rglob("*"):
        if f.is_file():
            out[f.name] = out.get(f.name, 0) + fincore_mib(f)
    return out
