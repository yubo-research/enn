import json
import math
import pathlib

R = pathlib.Path(__file__).resolve().parent.parent
D60 = R / "reports/report-0260-data/cap256"
D63 = R / "reports/report-0263-data/cap256"
ARMS = [
    ("main (brute force)", D60 / "capq_D10_256M_main.jsonl", "#555555", ""),
    ("default tree", D60 / "capq_D10_256M_def.jsonl", "#0072B2", ""),
    ("graph arm (0260)", D60 / "capq_D10_256M_g10.jsonl", "#E69F00", ""),
    ("tagged buffers (0262)", D63 / "capq_D10_256M_bg10t.jsonl", "#009E73", ""),
    ("+ re-homing R=300 (0263)", D63 / "capq_D10_256M_h300.jsonl", "#D55E00", "5 3"),
]


def load(p):
    out = []
    for l in pathlib.Path(p).read_text().splitlines():
        d = json.loads(l)
        if d["rows"] >= 7000:
            out.append((d["rows"], d["add_us_row"], d["q_us_med"] / 1000))
    return out


def logscale(v, lo, hi, a, b):
    return a + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (
        b - a
    )


def line_chart(idx, title, ylab, ylo, yhi, yticks):
    W, H, L, Rr, T, B = 470, 300, 52, 8, 30, 40
    s = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">',
        f'<text x="{L}" y="18" class="ct">{title}</text>',
    ]
    for t in yticks:
        y = logscale(t, ylo, yhi, H - B, T)
        s.append(
            f'<line x1="{L}" x2="{W - Rr}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{L - 6}" y="{y + 4:.1f}" class="ax" text-anchor="end">{t:g}</text>'
        )
    for x in (1e4, 1e5, 1e6):
        px = logscale(x, 5e3, 2.6e6, L, W - Rr)
        s.append(
            f'<text x="{px:.1f}" y="{H - B + 16}" class="ax" text-anchor="middle">{x:.0e}'.replace(
                "e+0", "e"
            ).replace("e+", "e")
            + "</text>"
        )
    s.append(
        f'<text x="{(L + W) / 2:.0f}" y="{H - 6}" class="ax" text-anchor="middle">rows in index (log scale)</text>'
    )
    s.append(
        f'<text x="12" y="{(T + H - B) / 2:.0f}" class="ax" text-anchor="middle" transform="rotate(-90 12 {(T + H - B) / 2:.0f})">{ylab}</text>'
    )
    for lab, f, c, dash in ARMS:
        pts = [
            (
                logscale(v[0], 5e3, 2.6e6, L, W - Rr),
                logscale(v[idx], ylo, yhi, H - B, T),
            )
            for v in load(f)
        ]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        s.append(
            f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="{dash}"><title>{lab}</title></polyline>'
        )
        for x, y in pts:
            s.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}" stroke="#fff" stroke-width="1.5"/>'
            )
    s.append("</svg>")
    return "\n".join(s)


def bars(title, items, xmax, unit):
    W, rowh, L, T = 470, 30, 190, 30
    H = T + rowh * len(items) + 10
    s = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}"><text x="0" y="18" class="ct">{title}</text>'
    ]
    for i, (lab, v, c) in enumerate(items):
        y = T + i * rowh
        w = v / xmax * (W - L - 60)
        s.append(
            f'<text x="{L - 8}" y="{y + 17}" class="ax" text-anchor="end">{lab}</text><rect x="{L}" y="{y + 4}" width="{w:.1f}" height="18" rx="3" fill="{c}"/><text x="{L + w + 6:.1f}" y="{y + 18}" class="val">{v:g}{unit}</text>'
        )
    s.append("</svg>")
    return "\n".join(s)


def recall_chart():
    W, H, L, T, B = 470, 260, 52, 30, 44
    pts = [(300, 0.981, "tags"), (1000, 0.954, "tags"), (3000, 0.911, "tags")]
    rh = [(300, 0.991), (3000, 0.983)]

    def X(r):
        return logscale(r, 200, 5000, L, W - 20)

    def Y(v):
        return H - B - (v - 0.90) / (1.0 - 0.90) * (H - B - T)

    s = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Recall vs buffer size"><text x="{L}" y="18" class="ct">Recall@10 at 10⁶ rows vs buffer size R</text>'
    ]
    for t in (0.90, 0.92, 0.94, 0.96, 0.98, 1.0):
        s.append(
            f'<line x1="{L}" x2="{W - 20}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" class="ax" text-anchor="end">{t:.2f}</text>'
        )
    for r in (300, 1000, 3000):
        s.append(
            f'<text x="{X(r):.1f}" y="{H - B + 16}" class="ax" text-anchor="middle">{r}</text>'
        )
    s.append(
        f'<text x="{(L + W) / 2:.0f}" y="{H - 6}" class="ax" text-anchor="middle">rows per buffer R (log scale)</text>'
    )
    s.append(
        f'<line x1="{L}" x2="{W - 20}" y1="{Y(0.994):.1f}" y2="{Y(0.994):.1f}" stroke="#555" stroke-dasharray="2 3"/><text x="{W - 22}" y="{Y(0.994) - 5:.1f}" class="ax" text-anchor="end">unbuffered .994</text>'
    )
    for data, c, lab in (
        (pts, "#009E73", "child tags (0262)"),
        (rh, "#D55E00", "tags + re-homing (0263)"),
    ):
        d = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in data)
        s.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        for p in data:
            s.append(
                f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="4" fill="{c}" stroke="#fff" stroke-width="1.5"><title>{lab} R={p[0]} recall {p[1]}</title></circle><text x="{X(p[0]) + 7:.1f}" y="{Y(p[1]) + 14:.1f}" class="val">{p[1]:.3f}</text>'
            )
    s.append(
        f'<text x="{X(1000) + 12:.1f}" y="{Y(0.954) - 12:.1f}" class="ax" fill="#009E73">child tags</text><text x="{X(1000) - 50:.1f}" y="{Y(0.987) + 22:.1f}" class="ax">tags + re-homing</text></svg>'
    )
    return "\n".join(s)


leaves = bars(
    "Leaves visited to reach the last true neighbour (10⁶ rows, D=10)",
    [
        ("default tree walk", 191, "#0072B2"),
        ("Lloyd sweep + tree walk", 114, "#56B4E9"),
        ("Lloyd sweep + graph walk", 35, "#009E73"),
        ("offline Lloyd, ideal order", 28, "#555555"),
    ],
    200,
    "",
)
cap_add = line_chart(
    1, "Add time per row, 256 MB cap", "µs per row", 0.3, 1000, (1, 10, 100, 1000)
)
cap_q = line_chart(
    2, "Query time (median), 256 MB cap", "ms per query", 0.1, 400, (0.1, 1, 10, 100)
)
recall = recall_chart()
legend = "".join(
    f'<span class="lg"><i style="background:{c}"></i>{l}</span>' for l, _, c, _ in ARMS
)

rows = [
    (
        "0254",
        "11-16 14:41",
        "Leaf capacity set at build time (<code>ENN_LEAF_CAP</code>); default 64 unchanged.",
        "committed",
    ),
    (
        "0255–0256",
        "11-16 20:32",
        "lnN audit; leaves per query vs N to 10⁷. Query-cost growth comes from the tree’s partition, not the problem: ideal-order tree needs 114 leaves at 10⁶, offline Lloyd 29.",
        "committed",
    ),
    (
        "0257",
        "11-16 21:33",
        "Paced Lloyd sweep works only with neighbourhoods from a <em>leaf graph</em> (NN-descent): 28 leaves at 10⁶ vs 114. Adds 14–18× slower in RAM.",
        "committed",
    ),
    (
        "0258",
        "11-16 23:10*",
        "Graph-ordered walk follows ideal order (N<sup>0.17</sup> vs N<sup>0.39</sup>); the walk’s length is still set by its stop rule.",
        "committed",
    ),
    (
        "0259",
        "*",
        "Neighbour lists stored on disk with 8-bit centroid codes; order kept (35 vs 34 leaves). Still 1.2× the tree walk in RAM.",
        "committed",
    ),
    (
        "0260",
        "*",
        "10 lnN leaf budget at parity with <code>main</code>; under 256 MB the graph arm steps at 3×10⁶ rows (13.8 KB written/row).",
        "committed",
    ),
    (
        "0261",
        "*",
        "Lloyd at buffer flush: capped adds 24.7 µs vs 797 µs, but queries become near-linear scans (28 ms at 10⁶).",
        "committed",
    ),
    (
        "0262",
        "11-17 04:39",
        "Child-tagged buffers: 26 ms/query and 24 µs/add at 3×10⁶ under the cap, but recall at 10⁶ falls to 0.914.",
        "committed",
    ),
    (
        "0263",
        "—",
        "Misses are rows in leaf-parent buffers under the wrong subtree; graph re-homing at R=300 restores recall (0.991), but capped add/query still step at 3×10⁶.",
        "<b>uncommitted</b>",
    ),
]
trs = "".join(
    f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows
)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ENN progress, 2026-11-16 afternoon to 2026-11-17</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1a1a1a}}
h1{{font-size:24px;margin-bottom:2px}} h2{{font-size:18px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.sub{{color:#666}} .tldr{{background:#f4f7fa;border-left:4px solid #0072B2;padding:8px 16px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:800px){{.grid2{{grid-template-columns:1fr}}}}
svg{{width:100%;height:auto}} .ct{{font-size:13px;font-weight:600;fill:#1a1a1a}} .ax{{font-size:11px;fill:#555}} .val{{font-size:11px;fill:#1a1a1a}}
.grid{{stroke:#e3e3e3}} table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid #e3e3e3;padding:5px 8px;text-align:left;vertical-align:top}}
.lg{{margin-right:14px;font-size:13px}} .lg i{{display:inline-block;width:14px;height:4px;margin-right:5px;vertical-align:middle}}
.fail{{color:#b00020;font-weight:600}} .pass{{color:#0a7a3d;font-weight:600}} .part{{color:#9a6700;font-weight:600}} small,.note{{color:#666;font-size:13px}}
</style></head><body>
<h1>ENN progress since yesterday afternoon</h1>
<div class="sub">2026-11-16 ~12:00 → 2026-11-17 · branch <code>dsweet/agg_obs</code> · reports 0254–0263 · D=10 unless noted</div>
<div class="tldr"><b>Main points.</b>
<ul>
<li><b>Query cost:</b> the partition, not the problem, made queries grow with N (0256). A Lloyd sweep over a leaf graph plus a graph-ordered walk cuts leaves visited at 10⁶ rows from 191 to ~35 and matches <code>main</code>’s prediction quality (0257–0260).</li>
<li><b>Add cost under the 256 MB cap:</b> buffering plus Lloyd at flush made capped adds ~30× cheaper (0261). Child tags made buffered queries 5–8× cheaper (0262).</li>
<li><b>Quality break found and explained:</b> tagged buffers lost recall at 10⁶ (0.914). 98.7% of misses were rows in leaf-parent buffers under the wrong subtree (0263). Graph re-homing at R=300 restores recall to 0.991.</li>
<li><b>VISION test still not passed:</b> every arm either steps in add or query time at 3×10⁶ rows under the cap. The latest arm (0263) steps in both.</li>
<li><b>Next:</b> re-parent leaves (Lloyd at the internal level), because Lloyd never moves leaves between parents, so internal nodes are not spatial clusters.</li>
</ul></div>

<h2>1. Timeline</h2>
<table><tr><th>Report</th><th>Commit time</th><th>Result</th><th>State</th></tr>{trs}</table>
<p class="note">* Reports 0258–0262 sit in four commits titled “ok” (11-16 23:10, 11-17 00:54, 02:19, 02:55) plus 04:39; I did not map each report to a commit. Source: <code>git log</code>, report abstracts in <code>reports/report-02xx.tex</code>, and <code>STATE.md</code>. Every new flag defaults off.</p>

<h2>2. Query order: the graph walk visits leaves in near-ideal order</h2>
{leaves}
<p class="note">Sources: 191→114 (0257), 35 (0259; 32 in 0258), offline Lloyd 28–29 (0256–0257). Counts from the 0256 harness, 300 queries, seed 0. The graph walk’s order follows the sweep’s NN-descent leaf graph [Dong, Moses, Li, WWW 2011], and the default budget still makes it visit ~314 leaves per query (0258).</p>

<h2>3. The VISION test under a 256 MB cap</h2>
<p>Add and query time should be a + b·lnN with no step as rows pass the cache. Each line is one arm; all are single capped runs, D=10. Ratios under 1.3× between sessions are not evidence (EBS burst credits; 0248–0249).</p>
<div>{legend}</div>
<div class="grid2">{cap_add}{cap_q}</div>
<p class="note">Data: <code>reports/report-0260-data/cap256</code> and <code>report-0263-data/cap256</code> (<code>capq_D10_256M_*.jsonl</code>). The log labels the last point 2×10⁶ rows; the reports call that step “3×10⁶”. I plotted the log’s labels and have not resolved the difference. Points are medians; <code>main</code> is brute force in anonymous memory, so it cannot go far past the cap.</p>
<table><tr><th>Arm (last point)</th><th>add µs/row</th><th>query ms</th><th>Verdict</th></tr>
<tr><td>main (brute force)</td><td>1.5</td><td>29.7</td><td class="part">smooth, but memory grows with N</td></tr>
<tr><td>default tree</td><td>198</td><td>255</td><td class="fail">steps in both</td></tr>
<tr><td>graph arm (0260)</td><td>797</td><td>58.5</td><td class="fail">add steps</td></tr>
<tr><td>tagged buffers (0262)</td><td>25.7</td><td>28.2</td><td class="part">passes cost; recall fails at 10⁶</td></tr>
<tr><td>+ re-homing R=300 (0263)</td><td>245</td><td>55.6</td><td class="fail">steps in both (4.3 KB written/row)</td></tr></table>

<h2>4. Why tagged buffers lost recall, and the partial fix</h2>
<div class="grid2">{recall}
<div><p>Recall falls steadily as buffers grow (0262). The diagnostic added in 0263 locates each true neighbour: <b>224 of 227 misses</b> (R=3000, seed 0) were in leaf-parent buffers whose owner leaf the walk never popped. Tree descent alone puts a row under its nearest leaf only <b>4%</b> of the time at late N.</p>
<p>Re-homing finds a row’s nearest leaf by greedy search over the on-disk leaf graph (≤8 hops). At R=300 it passes the pre-registered targets (recall ≥0.99, ≤5 ms in RAM: 1.9 ms). At R=3000 it reaches only 0.983, because rows go stale while waiting in a large buffer.</p>
<p>Quality against <code>main</code> (10 seeds): 10⁵ parity (Δloglik −0.0021, p=0.30). At 10⁶, Δloglik −0.0078 (p=0.071) and ΔNRMSE +0.00055 (p=0.025); before re-homing, −0.066.</p></div></div>
<p class="note">Sources: 0262 recall points (2 seeds) in <code>STATE.md</code>; 0263 re-homing numbers in <code>report-0263.tex</code>. Recall points at R=1000 come from the tag-only arm, with no re-homing measurement.</p>

<h2>5. Next steps and open questions</h2>
<ol>
<li><b>Re-parent leaves.</b> A paced pass moves each leaf to the leaf-parent whose centroid is nearest. Screen offline first: owner rank must fall from median 30 to ≤3. Pre-registered: R=3000 tags without re-homing reach recall ≥0.99 at 10⁶.</li>
<li><b>Attribute the capped add step</b> of the 0263 arm (4.3 KB written per row vs 569 for 0262) with per-file write counters.</li>
<li>If (1) fails: duplicate buffered rows into their two nearest leaves’ tags (SPANN-style).</li>
<li>When an arm passes in RAM and in quality: run the capped test to 3×10⁷ rows (10× the cache) for the VISION test.</li>
</ol>
<p><b>Questions for you</b> (<code>STATE.md</code>):</p>
<ul><li>May leaves move between parents on the housekeeping path? This changes parent pages and route words and moves no row data.</li>
<li>Is the 10⁶ ΔNRMSE of +0.17% (p=0.025 vs <code>main</code>) tolerable as “statistically insignificant degradation”? It is not distinguishable from the unbuffered graph arm (p=0.14).</li>
<li>Is the one-byte tag per buffered row acceptable as a creation-time store-format flag, like <code>ENN_LEAF_CAP</code>?</li></ul>
<p class="note">Caveats: report 0263, its flags (<code>ENN_PBUF_REHOME</code>, <code>ENN_PBUF_LEAF_ROWS</code>) and the diagnostic (<code>tree_pbuf_diag.rs</code>) are uncommitted. The figures above are generated by <code>scripts/make_progress_html.py</code>.</p>
</body></html>"""
(R / "progress.html").write_text(html)
print("ok", len(html))
