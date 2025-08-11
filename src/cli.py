# src/cli.py
from __future__ import annotations

import argparse
import csv

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MB-TFE: single-run trainer/evaluator")

    # dataset / split
    p.add_argument("--dataset", type=str, required=False, default="chameleon")
    p.add_argument("--split-id", type=int, default=None, help="random split index (None=public if exists)")
    p.add_argument("--seed", type=int, default=1)

    # model choice and sizes
    p.add_argument("--model", type=str, default="mbtfe", choices=["mbtfe", "tfe"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)

    # spectral / MB-TFE params
    p.add_argument("--K", type=int, default=8, help="Chebyshev order")
    p.add_argument("--bands", type=int, default=3, help="number of diff bands (len(tau))")
    p.add_argument("--tau", type=str, default="0.5,1.5,4.0", help="comma-separated list of tau values")
    p.add_argument("--lambda-div", type=float, default=0.0, help="diversity regularizer weight (MB-TFE)")

    # (for TFE baseline only; kept for completeness)
    p.add_argument("--fusion", type=str, default="concat", choices=["concat", "sum"])

    # training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--clip", type=float, default=0.0, help="grad clip norm (0=off)")
    p.add_argument("--log-interval", type=int, default=10)

    # robustness (optional; comma lists)
    p.add_argument("--robust-feature-noise", type=str, default="",
                   help="e.g., '0.0,0.05,0.1' for sigma values")
    p.add_argument("--robust-edge-drop", type=str, default="",
                   help="e.g., '0.0,0.1,0.2' for edge drop probabilities")

    # I/O / device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--outdir", type=str, default="results/runs")
    p.add_argument("--run-name", type=str, default=None)

    # grid / SLURM array support
    p.add_argument("--from-tsv", type=str, default=None, help="path to jobs.tsv")
    p.add_argument("--index", type=int, default=None, help="row index in jobs.tsv (0-based)")

    args = p.parse_args()

    # If pulling from TSV, merge those values first (TSV overrides defaults)
    if args.from_tsv is not None:
        assert args.index is not None, "Provide --index with --from-tsv"
        row = read_row_from_tsv(args.from_tsv, args.index)
        args = merge_args_from_row(args, row)

    # Basic sanity for tau / bands
    taus = parse_tau(args.tau)
    if len(taus) != args.bands:
        raise ValueError(f"--bands={args.bands} but got {len(taus)} tau values: {taus}")
    return args


def read_row_from_tsv(tsv_path: str, index: int) -> dict[str, str]:
    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter='\\t')
        rows = list(reader)
    if not (0 <= index < len(rows)):
        raise IndexError(f"Index {index} out of range for {tsv_path} (n={len(rows)})")
    return rows[index]


def merge_args_from_row(args: argparse.Namespace, row: dict[str, str]) -> argparse.Namespace:
    # Allowed keys map directly to CLI names where possible
    fields = {
        "dataset": str,
        "model": str,
        "seed": int,
        "split_id": int,
        "hidden": int,
        "layers": int,
        "dropout": float,
        "K": int,
        "bands": int,
        "tau": str,
        "lambda_div": float,
        "fusion": str,
        "epochs": int,
        "patience": int,
        "lr": float,
        "wd": float,
        "clip": float,
        "run_name": str,
    }
    for k, caster in fields.items():
        if k in row and row[k] != "" and row[k] is not None:
            # argparse uses dash names; our attr names are matching underscores
            attr = k.replace("-", "_")
            setattr(args, attr, caster(row[k]))
    return args


def parse_tau(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip() != ""]
