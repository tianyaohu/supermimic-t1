"""Summarise a train_log.csv: selected columns at the start, at evenly spaced points, and at the end.

Run: python supermimic_t1/tools/summarize.py logs/supermimic/<run>/train_log.csv [--cols a,b,c] [--points 8]
Values at each point are the mean over a window of --window log rows ending at that row.
"""

import argparse
import csv

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--cols", default="ho_e8_norm,ho_pos,ho_rot,ho_linvel,ho_angvel,ho_q,ho_qd,sim_rew,ep_len,ep_ret_norm,imag_rew,it_per_s")
p.add_argument("--points", type=int, default=8)
p.add_argument("--window", type=int, default=3)
a = p.parse_args()
rows = list(csv.DictReader(open(a.csv)))
cols = a.cols.split(",")
n = len(rows)
idx = sorted({min(n - 1, round(i * (n - 1) / (a.points - 1))) for i in range(a.points)})
print("iteration,wall_h," + ",".join(cols))
for i in idx:
    win = rows[max(0, i - a.window + 1) : i + 1]
    vals = [sum(float(r[c]) for r in win) / len(win) for c in cols]
    print(f"{rows[i]['iteration']},{float(rows[i]['wall_h']):.3f}," + ",".join(f"{v:.5g}" for v in vals))
print(f"peak vram_torch_peak_gib {max(float(r['vram_torch_peak_gib']) for r in rows):.2f}, "
      f"vram_smi_mib {max(float(r['vram_smi_mib']) for r in rows):.0f}")
