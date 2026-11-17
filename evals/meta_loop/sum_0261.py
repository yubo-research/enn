import json
import sys


def resident(o):
    r = o.get("res_mib", 0)
    return sum(r.values()) if isinstance(r, dict) else r


def main():
    for path in sys.argv[1:]:
        print(path.split("/")[-1])
        print("      n   add_us  wr_B/row  rd_B/row   q_med_us   q_rd_ops  res_MiB")
        with open(path) as fh:
            lines = [x for x in fh if x.startswith("{")]
        for line in lines:
            o = json.loads(line)
            print(
                f"{o['n']:>8} {o['add_us_row']:8.1f} {o['wr_b_row']:9.0f} {o['rd_b_row']:9.0f}"
                f" {o['q_us_med']:10.0f} {o['q_rd_ops']:10.1f} {resident(o):8.0f}"
            )


if __name__ == "__main__":
    main()
