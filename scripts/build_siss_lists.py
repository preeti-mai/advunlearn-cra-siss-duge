#!/usr/bin/env python3
import csv, math, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("--scores_csv", required=True)   # CSV with columns: prompt,score
ap.add_argument("--in_file", required=True)      # original prompt list (one per line)
ap.add_argument("--out_file", required=True)     # overwritten with duplicated lines
ap.add_argument("--temperature", type=float, default=0.5)
ap.add_argument("--max_copies", type=int, default=8)
args = ap.parse_args()

scores = {}
with open(args.scores_csv) as f:
    for row in csv.DictReader(f):
        scores[row["prompt"]] = float(row["score"])

with open(args.in_file) as f:
    prompts = [x.strip() for x in f if x.strip()]

vals = [scores.get(p, 0.0) for p in prompts]
# softmax with temperature
exps = [math.exp(v / max(args.temperature,1e-6)) for v in vals]
Z = sum(exps) if exps else 1.0
probs = [e / Z for e in exps]

aug = []
for p, pr in zip(prompts, probs):
    k = max(1, min(args.max_copies, int(round(pr * len(prompts) * 2))))
    aug.extend([p]*k)

random.shuffle(aug)
with open(args.out_file, "w") as f:
    f.write("\n".join(aug) + "\n")
print(f"[siss] wrote {len(aug)} lines to {args.out_file}")
