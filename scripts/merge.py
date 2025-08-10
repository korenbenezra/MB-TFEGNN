#!/usr/bin/env python3
# scripts/merge.py
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Any, List, Set


def main():
    ap = argparse.ArgumentParser("Merge results/runs/**/metrics.json into one CSV")
    ap.add_argument("--runs", type=str, default="results/runs", help="root folder with run subdirs")
    ap.add_argument("--out", type=str, default="results/summary/all_runs.csv", help="merged CSV path")
    ap.add_argument("--summary", type=str, default="", help="optional summary CSV (grouped by dataset,model)")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(runs_root):
        files = set(files)
        if "metrics.json" in files and "config.json" in files:
            run_dir = Path(root)
            try:
                with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                with (run_dir / "config.json").open("r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"[merge] skip {run_dir} (bad json): {e}")
                continue

            row: Dict[str, Any] = {}
            # core identifiers
            row["run_dir"] = str(run_dir)
            row["dataset"] = config.get("dataset", "")
            row["model"] = config.get("model", "")
            row["seed"] = config.get("seed", "")
            row["split_id"] = config.get("split_id", "")
            row["run_name"] = config.get("run_name", "")

            # common hparams (present in our main.py)
            for k in ["K", "bands", "tau", "hidden", "layers", "dropout",
                      "lr", "wd", "epochs", "patience", "fusion", "lambda_div"]:
                if k in config:
                    row[k] = config[k]

            # metrics we care about (copy flat ones; JSON-stringify complex)
            for k, v in metrics.items():
                if isinstance(v, (str, int, float)) or v is None:
                    row[k] = v
                else:
                    # nested dict/list -> keep as compact JSON string for now
                    row[k] = json.dumps(v, separators=(",", ":"))

            rows.append(row)

    if not rows:
        print(f"[merge] no runs found under {runs_root}")
        return

    # union of keys -> header
    header: List[str] = sorted(_collect_keys(rows))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"[merge] wrote {len(rows)} rows -> {out_csv}")

    # optional summary grouped by (dataset, model)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in rows:
            key = (r.get("dataset", ""), r.get("model", ""))
            groups.setdefault(key, []).append(r)

        # metrics to summarize — keep it simple and robust:
        metric_keys = [k for k in header if k.startswith("acc_") or k.startswith("macro_f1_")]

        sum_rows: List[Dict[str, Any]] = []
        for (dataset, model), grp in groups.items():
            sr: Dict[str, Any] = {"dataset": dataset, "model": model, "n": len(grp)}
            for mk in metric_keys:
                vals = _safe_floats([g.get(mk, None) for g in grp])
                if vals:
                    sr[f"{mk}_mean"] = f"{mean(vals):.4f}"
                    sr[f"{mk}_std"]  = f"{pstdev(vals):.4f}" if len(vals) > 1 else "0.0000"
            sum_rows.append(sr)

        sum_header = sorted(_collect_keys(sum_rows))
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sum_header)
            w.writeheader()
            for r in sum_rows:
                w.writerow({k: r.get(k, "") for k in sum_header})
        print(f"[merge] wrote summary ({len(sum_rows)} groups) -> {summary_path}")


def _collect_keys(dicts: List[Dict[str, Any]]) -> Set[str]:
    keys: Set[str] = set()
    for d in dicts:
        keys.update(d.keys())
    return keys


def _safe_floats(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        try:
            if x is None or x == "":
                continue
            out.append(float(x))
        except Exception:
            pass
    return out


if __name__ == "__main__":
    main()
