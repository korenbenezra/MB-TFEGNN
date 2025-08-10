# scripts/make_jobs.py
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List, Tuple

HEADER = [
    "job_id", "dataset", "model", "seed", "split_id",
    "K", "bands", "tau",
    "hidden", "layers", "dropout",
    "lr", "wd",
    "epochs", "patience",
    "fusion", "lambda_div",
    "run_name",
]

def parse_list(s: str, cast=float) -> List:
    s = (s or "").strip()
    if not s:
        return []
    return [cast(x) for x in s.split(",") if x.strip()]

def parse_tau_sets(s: str) -> List[str]:
    """
    Accepts one or multiple tau sets.
    Example:
      "0.5,1.5,4.0;0.3,1.0,3.0"  ->  ["0.5,1.5,4.0", "0.3,1.0,3.0"]
    If empty, caller should fall back to --tau (single set).
    """
    s = (s or "").strip()
    if not s:
        return []
    return [chunk.strip() for chunk in s.split(";") if chunk.strip()]

def tau_len(tau_str: str) -> int:
    return len([x for x in tau_str.split(",") if x.strip()])

def write_tsv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        # header
        f.write("\t".join(HEADER) + "\n")
        # rows
        for r in rows:
            vals = [str(r.get(k, "")) for k in HEADER]
            f.write("\t".join(vals) + "\n")

def main():
    ap = argparse.ArgumentParser("Generate jobs.tsv for MB-TFE/TFE runs (Cartesian grid)")
    ap.add_argument("--datasets", type=str, required=True,
                    help="comma-separated list, e.g., 'chameleon,squirrel'")
    ap.add_argument("--model", type=str, default="mbtfe", choices=["mbtfe", "tfe"])
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    ap.add_argument("--split-id", type=int, default=-1,
                    help="-1 means 'use public if exists, else random split 0'")
    # spectral / bands
    ap.add_argument("--K", type=str, default="8", help="chebyshev orders, e.g. '5,8,12'")
    ap.add_argument("--bands", type=str, default="3", help="#bands values, e.g. '2,3'")
    ap.add_argument("--tau", type=str, default="0.5,1.5,4.0",
                    help="single tau set if --tau-sets not given (comma list)")
    ap.add_argument("--tau-sets", type=str, default="",
                    help="multiple tau sets separated by ';' (each is a comma-list)")
    # model size / training
    ap.add_argument("--hidden", type=str, default="128")
    ap.add_argument("--layers", type=str, default="1")
    ap.add_argument("--dropout", type=str, default="0.5")
    ap.add_argument("--lr", type=str, default="1e-3")
    ap.add_argument("--wd", type=str, default="5e-5")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100)
    # extras
    ap.add_argument("--fusion", type=str, default="concat", choices=["concat", "sum"],
                    help="used by TFE (safe to keep for mbtfe)")
    ap.add_argument("--lambda-div", type=str, default="0.0")
    ap.add_argument("--run-name-prefix", type=str, default="", help="optional prefix for run_name")
    ap.add_argument("--out", type=str, default="scripts/jobs.tsv")

    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds    = [int(x) for x in parse_list(args.seeds, int)]
    Ks       = [int(x) for x in parse_list(args.K, int)]
    bandsL   = [int(x) for x in parse_list(args.bands, int)]
    hiddenL  = [int(x) for x in parse_list(args.hidden, int)]
    layersL  = [int(x) for x in parse_list(args.layers, int)]
    dropouts = [float(x) for x in parse_list(args.dropout, float)]
    lrs      = [float(x) for x in parse_list(args.lr, float)]
    wds      = [float(x) for x in parse_list(args.wd, float)]
    lambdas  = [float(x) for x in parse_list(args.lambda_div, float)]

    tau_sets = parse_tau_sets(args.tau_sets)
    if not tau_sets:
        tau_sets = [args.tau.strip()]  # single set

    rows = []
    jid = 0
    for dataset, seed, K, bands, tau_str, hidden, layers, dropout, lr, wd, lam in itertools.product(
        datasets, seeds, Ks, bandsL, tau_sets, hiddenL, layersL, dropouts, lrs, wds, lambdas
    ):
        # Enforce valid (bands == len(tau))
        if tau_len(tau_str) != bands:
            # skip invalid combination
            continue

        run_name = f"{args.run_name_prefix}{args.model}_{dataset}"
        row = {
            "job_id": jid,
            "dataset": dataset,
            "model": args.model,
            "seed": seed,
            "split_id": args.split_id,
            "K": K,
            "bands": bands,
            "tau": tau_str,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
            "lr": lr,
            "wd": wd,
            "epochs": args.epochs,
            "patience": args.patience,
            "fusion": args.fusion,
            "lambda_div": lam,
            "run_name": run_name,
        }
        rows.append(row)
        jid += 1

    out_path = Path(args.out)
    write_tsv(out_path, rows)
    print(f"[make_jobs] wrote {len(rows)} jobs to {out_path}")

if __name__ == "__main__":
    main()
