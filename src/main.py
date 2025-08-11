# src/main.py
from __future__ import annotations

from .cli import parse_args
from .train import run

if __name__ == "__main__":
    args = parse_args()
    run(args)
