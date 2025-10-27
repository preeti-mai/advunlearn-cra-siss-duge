#!/usr/bin/env python3
import os, sys, yaml, subprocess, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADV = ROOT / "vendor" / "AdvUnlearn"

def run(cmd, env=None):
    print("[run]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["ckpt_dir"]); out.mkdir(parents=True, exist_ok=True)
    logs = Path(cfg["log_dir"]); logs.mkdir(parents=True, exist_ok=True)

    # base args to AdvUnlearn trainer (adjust to the repo’s CLI if needed)
    base_cli = [
        sys.executable, str(ADV / "train-scripts" / "AdvUnlearn.py"),
        "--prompt", cfg["concept"],
        "--sd_ckpt", cfg["sd_checkpoint_path"],
        "--retain_dataset_root", cfg["retain_dataset_root"],
        "--seed", str(cfg["seed"]),
        "--retain_train", "reg",
        "--attack_type", "prefix_k",
        "--attack_init", "random",
        "--num_gpus", str(cfg.get("num_gpus", 1)),
    ]

    last_ckpt = None
    for i, ph in enumerate(cfg["phases"]):
        print(f"\n=== Phase {i+1}/{len(cfg['phases'])}: {ph['name']} ===")

        # SISS: rebuild training prompt list with duplication by difficulty (if any)
        train_list = ROOT / ph["train_attacks_file"]
        if cfg.get("siss", {}).get("enabled", False):
            run([sys.executable, str(HERE / "build_siss_lists.py"),
                 "--temperature", str(cfg["siss"]["temperature"]),
                 "--scores_csv", cfg["siss"]["difficulty_scores_csv"],
                 "--in_file", str(train_list),
                 "--out_file", str(train_list),
                 "--max_copies", str(cfg["siss"]["max_copies_per_prompt"])])

        # assemble phase CLI (no trainer edits)
        phase_cli = base_cli + [
            "--steps", str(ph["steps"]),
            "--attack_step", str(ph["attack_step"]),
            "--retain_loss_w", str(ph["retain_loss_w"]),
            "--train_attacks_file", str(train_list),
            "--save_dir", str(out),
        ]
        if last_ckpt:
            phase_cli += ["--resume", str(last_ckpt)]

        # run training
        run(phase_cli)

        # locate newest checkpoint
        ckpts = sorted(out.glob("*.ckpt"), key=os.path.getmtime)
        last_ckpt = ckpts[-1] if ckpts else last_ckpt

        # CRA alignment micro-steps (external; doesn’t touch trainer)
        if cfg.get("cra", {}).get("enabled", False):
            run([sys.executable, str(HERE / "cra_align.py"),
                 "--ckpt", str(last_ckpt),
                 "--anchors", cfg["cra"]["anchors_file"],
                 "--steps", str(cfg["cra"]["align_steps"]),
                 "--batch_size", str(cfg["cra"]["batch_size"]),
                 "--lr", str(cfg["cra"]["lr"]),
                 "--save_out", str(out / f"{Path(last_ckpt).stem}_cra.ckpt")])
            # update checkpoint pointer
            last_ckpt = sorted(out.glob("*_cra.ckpt"))[-1]

        # Gate evaluation
        gate_cfg = cfg["gate_eval"]
        gate_res = {}
        run([sys.executable, str(HERE / "eval_asr.py"),
             "--ckpt", str(last_ckpt),
             "--attacks", gate_cfg["val_attacks_file"],
             "--out_json", str(logs / f"{ph['name']}_asr.json")])
        with open(logs / f"{ph['name']}_asr.json") as f:
            gate_res["asr"] = json.load(f)["asr"]

        run([sys.executable, str(HERE / "eval_fid_clip.py"),
             "--ckpt", str(last_ckpt),
             "--num_images", str(gate_cfg["images_for_fid"]),
             "--ref_dir", gate_cfg["fid_ref_dir"],
             "--out_json", str(logs / f"{ph['name']}_fid.json")])
        with open(logs / f"{ph['name']}_fid.json") as f:
            g2 = json.load(f); gate_res["fid_delta"] = g2["fid_delta"]

        print(f"[gate] {gate_res}")
        if (gate_res["asr"] > ph["gate"]["asr_max"]) or (gate_res["fid_delta"] > ph["gate"]["fid_delta_max"]):
            print("[gate] Not satisfied. Re-running this phase with +500 steps.")
            ph["steps"] += 500
            # loop will naturally re-enter next iteration with modified steps

    print("\n[done] All phases completed.")
    print(f"[final_ckpt] {last_ckpt}")

if __name__ == "__main__":
    main()
