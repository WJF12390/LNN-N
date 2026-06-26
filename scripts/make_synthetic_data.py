from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Make synthetic medication packaging questionnaire data.")
    p.add_argument("--out", default="data/demo_medication_packaging.csv")
    p.add_argument("--n", type=int, default=448)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args(); rng = np.random.default_rng(args.seed); n = args.n
    latent = rng.normal(0, 1, size=n)
    visual = 3.5 + 0.45 * latent + rng.normal(0, 0.45, n)
    function = 3.7 + 0.50 * latent + rng.normal(0, 0.45, n)
    experience = 3.45 + 0.48 * latent + rng.normal(0, 0.50, n)
    perceived = 3.58 + 0.35 * visual + 0.20 * experience + rng.normal(0, 0.45, n) - 1.75
    daily = 3.75 + 0.32 * function + 0.25 * experience + rng.normal(0, 0.43, n) - 1.80
    adherence = 3.35 + 0.10 * visual + 0.18 * function + 0.08 * experience + 0.03 * perceived + 0.28 * daily + rng.normal(0, 0.42, n) - 1.80
    df = pd.DataFrame({"视觉设计": visual, "功能性设计": function, "体验设计": experience, "感知体验": perceived, "日常管理体验": daily, "用药依从性": adherence})
    for c in df.columns: df[c] = df[c].clip(1, 5).round(3)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved synthetic dataset to: {out}")

if __name__ == "__main__": main()
