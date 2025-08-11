# src/train.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F

# local modules
from . import data as data_mod
from . import utils
from . import metrics

# models (mbtfe is our default; tfe is optional baseline)
from .mbtfe_model import MBTFEModel
from .model_tfe import TFEModel

# helpers from CLI (to avoid duplication)
from .cli import parse_tau, parse_float_list


def run(args: argparse.Namespace) -> dict[str, object]:
    """Train and evaluate a single run based on CLI args."""
    device = utils.get_device(args.device)
    utils.set_seed(args.seed)

    # ---------- data ----------
    ds = data_mod.load_dataset(
        name=args.dataset,
        split_id=args.split_id,
        seed=args.seed,
        cache=True,
    )
    X: torch.Tensor = ds["X"].to(device)
    y: torch.Tensor = ds["y"].to(device)
    train_mask: torch.Tensor = ds["train_mask"].to(device)
    val_mask: torch.Tensor = ds["val_mask"].to(device)
    test_mask: torch.Tensor = ds["test_mask"].to(device)
    L_hat: torch.Tensor = ds["L_hat"].to(device)
    num_nodes, in_dim = X.shape
    num_classes = int(ds["meta"]["num_classes"])

    # ---------- model ----------
    taus = parse_tau(args.tau)
    if args.model == "mbtfe":
        model = MBTFEModel(
            in_dim=in_dim,
            hidden=args.hidden,
            out_dim=num_classes,
            K=args.K,
            taus=taus,
            n_layers=args.layers,
            dropout=args.dropout,
            lambda_div=args.lambda_div,
        ).to(device)
    elif args.model == "tfe":
        model = TFEModel(
            in_dim=in_dim,
            hidden=args.hidden,
            out_dim=num_classes,
            K=args.K,
            tau_lp=taus[0] if len(taus) > 0 else 0.5,
            tau_hp=taus[-1] if len(taus) > 0 else 1.5,
            n_layers=args.layers,
            dropout=args.dropout,
            fusion=args.fusion,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ---------- run dir & logging ----------
    run_dir = utils.mk_run_dir(args.outdir, args.dataset, args.model, args.run_name)
    logger = utils.JSONLLogger(Path(run_dir) / "train.log.jsonl")
    config_to_save = utils.format_args(args)
    utils.save_json(Path(run_dir) / "config.json", config_to_save)

    # ---------- optimizer ----------
    optim_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = None  # keep simple; you can add cosine anneal later

    # ---------- training loop ----------
    best_val_acc = -1.0
    best_epoch = -1
    saver = utils.CheckpointSaver(run_dir)

    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optim_.zero_grad()

        logits = model(X, L_hat, need_aux=False)  # forward
        loss = F.cross_entropy(logits[train_mask], y[train_mask])

        div_term = getattr(model, "diversity_loss_total", None)
        if callable(div_term):
            loss = loss + div_term()

        loss.backward()
        if args.clip and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim_.step()
        if scheduler is not None:
            scheduler.step()

        # eval on train/val
        model.eval()
        with torch.no_grad():
            train_acc = metrics.accuracy(logits, y, train_mask)
            val_dict = metrics.evaluate(model, X, L_hat, y,
                                        masks={"train": train_mask, "val": val_mask, "test": test_mask},
                                        device=device, need_aux=False)
            val_acc = float(val_dict["acc_val"])

        # early stopping
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            saver.save_if_best(model)

        # logging
        rec = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "acc_train": float(train_acc),
            "acc_val": float(val_acc),
            "lr": float(optim_.param_groups[0]["lr"]),
            "time": float(time.time() - start_time),
        }
        logger.log(rec)
        history.append(rec)
        if (epoch % args.log_interval) == 0 or epoch == 1:
            print(f"[{epoch:04d}] loss={rec['loss']:.4f} | acc_tr={rec['acc_train']:.3f} acc_val={rec['acc_val']:.3f}")

        # patience
        if (epoch - best_epoch) >= args.patience:
            print(f"Early stopping at epoch {epoch} (best @ {best_epoch}, val_acc={best_val_acc:.4f})")
            break

    # ---------- final eval (best checkpoint) ----------
    saver.load_best(model)
    model.eval()
    with torch.no_grad():
        out = metrics.evaluate(model, X, L_hat, y,
                               masks={"train": train_mask, "val": val_mask, "test": test_mask},
                               device=device, need_aux=True)
        final = {
            "acc_train": float(out["acc_train"]),
            "acc_val": float(out["acc_val"]),
            "acc_test": float(out["acc_test"]),
            "macro_f1_train": float(out.get("macro_f1_train", 0.0)),
            "macro_f1_val": float(out.get("macro_f1_val", 0.0)),
            "macro_f1_test": float(out.get("macro_f1_test", 0.0)),
        }

        # optional robustness
        sigmas = parse_float_list(args.robust_feature_noise)
        if sigmas:
            final["robust_feature_noise"] = metrics.feature_noise_eval(model, X, L_hat, y,
                                                                       masks={"test": test_mask},
                                                                       sigmas=sigmas)
        ps = parse_float_list(args.robust_edge_drop)
        if ps:
            # need L_sym to rebuild when dropping edges; pull from loader again (CPU ok)
            L_sym = ds["L_sym"].to(device)
            final["robust_edge_drop"] = metrics.edge_dropout_eval(model, X, L_sym, y,
                                                                  masks={"test": test_mask},
                                                                  ps=ps)

    # ---------- save metrics ----------
    utils.save_json(Path(run_dir) / "metrics.json", final)
    # also keep per-epoch history
    utils.save_json(Path(run_dir) / "history.json", history)

    # console summary
    print(json.dumps({"BEST_VAL_ACC": best_val_acc, "RUN_DIR": str(run_dir)}, indent=2))
    return final
