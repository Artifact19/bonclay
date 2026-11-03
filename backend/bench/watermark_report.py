#!/usr/bin/env python3
"""Aggregate detect_rows.json from watermark_attacks.py into a markdown table. Usage: watermark_report.py rows.json"""
import json, sys, collections
import numpy as np
rows = json.load(open(sys.argv[1]))
by = collections.defaultdict(lambda: {"marked": [], "clean": []})
for r in rows:
    by[r["attack"]][r["kind"]].append(r)
print("| attack | detect rate (marked) | mean prob (marked) | exact 16-bit msg match | mean bit acc (raw frames) | EN / HI detect | false-pos rate (clean) | mean prob (clean) |")
print("|---|---|---|---|---|---|---|---|")
for a, d in by.items():
    m, c = d["marked"], d["clean"]
    n = len(m)
    det = sum(r["detected"] for r in m)
    ex = sum(r["msg_exact"] for r in m)
    ba = [r["raw_msg_bitacc"] for r in m if r["raw_msg_bitacc"] is not None]
    en = sum(r["detected"] for r in m if r["lang"] == "en"); hi = det - en
    fp = sum(r["detected"] for r in c)
    print(f"| {a} | {det}/{n} ({100*det/n:.0f}%) | {np.mean([r['prob'] for r in m]):.3f} | {ex}/{n} | "
          f"{(np.mean(ba) if ba else float('nan')):.3f} (n={len(ba)}) | {en}/10, {hi}/10 | {fp}/{len(c)} | {np.mean([r['prob'] for r in c]):.3f} |")
