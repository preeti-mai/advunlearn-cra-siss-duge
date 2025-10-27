# eval_asr.py (skeleton)
#!/usr/bin/env python3
import json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--attacks", required=True)
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

# TODO: call the repo’s attack evaluator on a small set; parse its summary
# For now, write a dummy schema so outer_loop works end-to-end.
res = {"asr": 0.22}
json.dump(res, open(args.out_json, "w"))
print(res)
