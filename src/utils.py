# src/utils.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# ---------------------------
# Reproducibility & devices
# ---------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if present), and PyTorch (CPU/GPU)."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # make determinism best-effort without killing perf:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(pref: str = "auto") -> torch.device:
    """Pick device from {'auto','cpu','cuda'}."""
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            print("[utils] CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def param_count(model: torch.nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------
# JSON / logging utilities
# ---------------------------

def save_json(path: os.PathLike | str, obj: Any, indent: int = 2) -> None:
    """Write JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=indent)
    tmp.replace(path)


def load_json(path: os.PathLike | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class JSONLLogger:
    """Append JSON lines (one dict per line)."""
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(record)) + "\n")


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion for JSON dumping."""
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # Tensors / numpy
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, torch.Tensor):
        if x.numel() <= 1:
            return x.item()
        return x.detach().cpu().tolist()
    # Fallback to string
    return str(x)


# ---------------------------
# Run directories & checkpoints
# ---------------------------

def mk_run_dir(base_outdir: str | os.PathLike, dataset: str, model: str, run_name: Optional[str] = None) -> Path:
    """
    Create a run directory like:
      results/runs/<dataset>/<model>/<run_name_or_timestamp>/
    """
    base = Path(base_outdir)
    if run_name is None or len(str(run_name).strip()) == 0:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        short = str(uuid.uuid4())[:8]
        run_name = f"{stamp}-{short}"
    run_dir = base / sanitize(dataset) / sanitize(model) / sanitize(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize(s: str) -> str:
    """Make a safe folder name."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(s))


class CheckpointSaver:
    """
    Minimal checkpoint manager. Usage:
        saver = CheckpointSaver(run_dir)
        saver.save_if_best(model)  # when you detect improvement
        saver.save_last(model)     # optional each epoch
        saver.load_best(model)
    """
    def __init__(self, run_dir: os.PathLike | str):
        self.run_dir = Path(run_dir)
        self.best_path = self.run_dir / "ckpt_best.pt"
        self.last_path = self.run_dir / "ckpt_last.pt"

    def save_if_best(self, model: torch.nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
        self._atomic_save(self.best_path, _pack_state(model, extra))

    def save_last(self, model: torch.nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
        self._atomic_save(self.last_path, _pack_state(model, extra))

    def load_best(self, model: torch.nn.Module, map_location: Optional[str | torch.device] = None) -> None:
        if not self.best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {self.best_path}")
        state = torch.load(self.best_path, map_location=map_location or "cpu")
        model.load_state_dict(state["model"])

    def _atomic_save(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)


def _pack_state(model: torch.nn.Module, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = {"model": model.state_dict()}
    if extra:
        state["extra"] = extra
    return state


# ---------------------------
# Arg formatting
# ---------------------------

def format_args(args) -> Dict[str, Any]:
    """Turn argparse.Namespace (or dict-like) into a plain dict for saving."""
    if hasattr(args, "__dict__"):
        d = {k: v for k, v in vars(args).items()}
    elif isinstance(args, dict):
        d = dict(args)
    else:
        # last resort: reflect
        d = {k: getattr(args, k) for k in dir(args) if not k.startswith("_")}
    # Convert lists/tuples of numbers to plain lists; leave strings as-is
    for k, v in list(d.items()):
        if isinstance(v, (tuple, set)):
            d[k] = list(v)
    return d
