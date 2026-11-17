import os

import probe_0261 as base

import enn._rust as r

base_per_query = base.per_query


def per_query_tagged(model, qx, p, f):
    r.take_pbuf_tag_stats()
    out = base_per_query(model, qx, p, f)
    tg = r.take_pbuf_tag_stats()
    out.update(
        {
            "tag": os.environ.get("ENN_PBUF_TAG", "0"),
            "buf_scanned_q": tg[0] / len(qx),
            "buf_visited_q": tg[1] / len(qx),
            "retags": tg[2],
            "orphans_q": tg[3] / len(qx),
        }
    )
    return out


def main():
    base.per_query = per_query_tagged
    base.main()


if __name__ == "__main__":
    main()
