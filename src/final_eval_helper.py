from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import json
import torch

from .metrics import evaluate, feature_noise_eval, edge_dropout_eval
from .cli import parse_float_list


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)

def _plot_training_curves(history, out_png: Path):
    import math
    fig = plt.figure(figsize=(7, 4))
    xs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    accv = [h["acc_val"] for h in history]
    ax1 = plt.gca()
    ax1.plot(xs, loss, label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(xs, accv, label="acc_val", linestyle="--")
    ax2.set_ylabel("val acc")
    ax1.set_title("Training curves")
    _savefig(fig, out_png)

def _confusion_matrix_plot(y_true, y_pred, class_names, out_png: Path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(5.5, 5))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, str(val),
                        ha="center", va="center",
                        color="white" if val > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    _savefig(fig, out_png)

def _roc_pr_plots_ovr(y_true, proba, class_names, out_dir: Path):
    """
    One-vs-rest ROC & PR curves (requires scikit-learn). Gracefully no-op if unavailable.
    """
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    except Exception:
        return  # skip if sklearn not present

    C = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(C)))
    # ROC (OVR)
    fig = plt.figure(figsize=(6, 5))
    aucs = []
    for c in range(C):
        fpr, tpr, _ = roc_curve(y_bin[:, c], proba[:, c])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (One-vs-Rest)")
    plt.legend(fontsize=8)
    _savefig(fig, out_dir / "roc_ovr.png")

    # Precision–Recall (OVR)
    fig = plt.figure(figsize=(6, 5))
    aps = []
    for c in range(C):
        prec, rec, _ = precision_recall_curve(y_bin[:, c], proba[:, c])
        ap = average_precision_score(y_bin[:, c], proba[:, c])
        aps.append(ap)
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (One-vs-Rest)")
    plt.legend(fontsize=8)
    _savefig(fig, out_dir / "pr_ovr.png")

    # Save summary
    np.save(out_dir / "roc_ovr_aucs.npy", np.array(aucs))
    np.save(out_dir / "pr_ovr_aps.npy", np.array(aps))

def _reliability_diagram_confidence(y_true, proba, n_bins: int, out_png: Path):
    """
    Multiclass-friendly reliability diagram using max-confidence.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs, counts = [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == 0:
            sel = (conf >= lo) & (conf <= hi)
        else:
            sel = (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        accs.append(correct[sel].mean())
        confs.append(conf[sel].mean())
        counts.append(sel.sum())

    ece = np.sum(np.array(counts) / max(1, len(conf)) * np.abs(np.array(accs) - np.array(confs)))

    fig = plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "--", label="perfect")
    plt.scatter(confs, accs)
    plt.plot(confs, accs, alpha=0.7)
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.title(f"Reliability (confidence)  —  ECE≈{ece:.3f}")
    _savefig(fig, out_png)

def _confidence_hist(proba, out_png: Path):
    conf = proba.max(axis=1)
    fig = plt.figure(figsize=(5.5, 3.2))
    plt.hist(conf, bins=20, edgecolor="k")
    plt.xlabel("max probability (confidence)")
    plt.ylabel("count")
    plt.title("Confidence histogram")
    _savefig(fig, out_png)

def _robustness_plots(final_dict, out_dir: Path):
    if "robust_feature_noise" in final_dict:
        xs = [d["sigma"] for d in final_dict["robust_feature_noise"]]
        ys = [d["acc_test"] for d in final_dict["robust_feature_noise"]]
        fig = plt.figure(figsize=(5.5, 3.2))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("σ (feature noise)")
        plt.ylabel("test acc")
        plt.title("Robustness to feature noise")
        _savefig(fig, out_dir / "robust_feature_noise.png")
    if "robust_edge_drop" in final_dict:
        xs = [d["p"] for d in final_dict["robust_edge_drop"]]
        ys = [d["acc_test"] for d in final_dict["robust_edge_drop"]]
        fig = plt.figure(figsize=(5.5, 3.2))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("p (edge dropout)")
        plt.ylabel("test acc")
        plt.title("Robustness to edge dropout")
        _savefig(fig, out_dir / "robust_edge_dropout.png")

def _mbtfe_diagnostics(aux_layers, out_dir: Path):
    if not aux_layers:
        return
    # band energy heatmap (layers x bands)
    band_mat = np.array([np.array(l["band_energy"], dtype=np.float32) for l in aux_layers])
    fig = plt.figure(figsize=(6.2, 3.8))
    im = plt.imshow(band_mat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("band index")
    plt.ylabel("layer")
    plt.title("MB-TFE band energy (per layer)")
    _savefig(fig, out_dir / "mbtfe_band_energy.png")

    # telescoping residual and diversity per layer
    tel = [float(l.get("telescoping_residual", 0.0)) for l in aux_layers]
    div = [float(l.get("div_loss", 0.0)) for l in aux_layers]
    fig = plt.figure(figsize=(6.2, 3.2))
    plt.plot(range(1, len(tel) + 1), tel, marker="o", label="telescoping residual")
    plt.plot(range(1, len(div) + 1), div, marker="s", label="diversity loss")
    plt.xlabel("layer")
    plt.title("MB-TFE diagnostics")
    plt.legend()
    _savefig(fig, out_dir / "mbtfe_layer_diagnostics.png")

def _tsne_logits_plot(logits, y_true, out_png: Path):
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("Skipping t-SNE plot: scikit-learn not available.")
        return  # skip if sklearn not present
    Z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(logits)
    fig = plt.figure(figsize=(5.8, 5.2))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y_true, s=10)
    plt.title("t-SNE of logits")
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    _savefig(fig, out_png)

# ---------- final eval + plots ----------
def do_final_eval_and_plots(
    model, saver, X, L_hat, y, masks, device, run_dir: Path, ds_meta: dict, args
):
    print(f"Starting final evaluation and plots in {run_dir}")
    eval_dir = _ensure_dir(Path(run_dir) / "final_eval")
    plots_dir = _ensure_dir(eval_dir / "plots")
    tables_dir = _ensure_dir(eval_dir / "tables")
    np_dir = _ensure_dir(eval_dir / "np")

    # Load best and compute full forward with aux
    try:
        saver.load_best(model)
        model.eval()
        with torch.no_grad():
            logits_out = model(X, L_hat, need_aux=True)
            if isinstance(logits_out, tuple) and len(logits_out) == 2:
                logits, aux = logits_out
            else:
                # Model doesn't return aux
                logits, aux = logits_out, {}
                print("Warning: Model does not return aux information")
            probs = torch.softmax(logits, dim=1)
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        # Return basic metrics if we have them
        return {
            "acc_train": 0.0,
            "acc_val": 0.0,
            "acc_test": 0.0,
            "error": str(e)
        }

    # Metrics (reusing your evaluate)
    try:
        out = evaluate(model, X, L_hat, y, masks=masks, device=device, need_aux=True)
        final = {
            "acc_train": float(out.get("acc_train", 0.0)),
            "acc_val": float(out.get("acc_val", 0.0)),
            "acc_test": float(out.get("acc_test", 0.0)),
            "macro_f1_train": float(out.get("macro_f1_train", 0.0)),
            "macro_f1_val": float(out.get("macro_f1_val", 0.0)),
            "macro_f1_test": float(out.get("macro_f1_test", 0.0)),
        }
    except Exception as e:
        print(f"Error during metrics evaluation: {e}")
        final = {
            "acc_train": 0.0,
            "acc_val": 0.0, 
            "acc_test": 0.0,
            "error": str(e)
        }

    # Optional robustness you already support
    sigmas = parse_float_list(args.robust_feature_noise)
    if sigmas:
        final["robust_feature_noise"] = feature_noise_eval(
            model, X, L_hat, y, masks={"test": masks["test"]}, sigmas=sigmas, device=device
        )
    ps = parse_float_list(args.robust_edge_drop)
    if ps:
        # L_sym should be provided directly as an argument, not from ds_meta
        # Assuming it's passed in via the masks dict for simplicity
        L_sym = masks.get("L_sym", None)
        if L_sym is not None:
            final["robust_edge_drop"] = edge_dropout_eval(
                model, X, L_sym, y, masks={"test": masks["test"]}, ps=ps, device=device
            )

    # Save arrays for downstream analysis
    np.save(np_dir / "logits.npy", logits.detach().cpu().numpy())
    np.save(np_dir / "probs.npy", probs.detach().cpu().numpy())
    np.save(np_dir / "y.npy", y.detach().cpu().numpy())
    for k, v in final.items():
        if isinstance(v, (list, tuple)):
            # robustness lists
            np.save(np_dir / f"{k}.npy", np.array(v, dtype=object))

    # Export predictions on test set
    try:
        test_mask = masks["test"].bool()
        y_t = y[test_mask].cpu().numpy()
        p_t = probs[test_mask].cpu().numpy()
        yhat_t = p_t.argmax(axis=1)
        
        # tables CSV
        import csv
        # Ensure we have class names, default to numerical class IDs if not provided
        num_classes = int(y.max().item() + 1)
        class_names = ds_meta.get("class_names", [str(i) for i in range(num_classes)])
        
        with open(tables_dir / "predictions_test.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_id", "y_true", "y_pred", "confidence"])
            # Make sure test_mask and torch.arange are on the same device
            idxs = torch.arange(y.size(0), device=test_mask.device)[test_mask].cpu().numpy()
            for nid, yt, yp, conf in zip(idxs, y_t, yhat_t, p_t.max(axis=1)):
                w.writerow([int(nid), int(yt), int(yp), float(conf)])
    except Exception as e:
        print(f"Error during predictions export: {e}")

    # Plots
    try:
        _plot_training_curves(json.load(open(Path(run_dir)/"history.json")), plots_dir / "training_curves.png")
    except Exception as e:
        print(f"Error generating training curves: {e}")
    
    try:
        _confusion_matrix_plot(y_t, yhat_t, class_names, plots_dir / "confusion_matrix.png")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    try:
        _roc_pr_plots_ovr(y_t, p_t, class_names, plots_dir)
    except Exception as e:
        print(f"Error generating ROC/PR plots: {e}")
    
    try:
        _reliability_diagram_confidence(y_t, p_t, n_bins=15, out_png=plots_dir / "reliability_confidence.png")
    except Exception as e:
        print(f"Error generating reliability diagram: {e}")
    
    try:
        _confidence_hist(p_t, plots_dir / "confidence_hist.png")
    except Exception as e:
        print(f"Error generating confidence histogram: {e}")
    
    try:
        _robustness_plots(final, plots_dir)
    except Exception as e:
        print(f"Error generating robustness plots: {e}")

    # MB-TFE extras if aux present
    try:
        # Get layers information from aux
        if isinstance(aux, dict) and "layers" in aux:
            aux_layers = aux["layers"]
            _mbtfe_diagnostics(aux_layers, plots_dir)
        # Try alternative location in case format is different
        elif "aux" in out and isinstance(out["aux"], dict) and "layers" in out["aux"]:
            aux_layers = out["aux"]["layers"]
            _mbtfe_diagnostics(aux_layers, plots_dir)
    except Exception as e:
        print(f"Warning: Could not generate MB-TFE diagnostics: {e}")

    # Optional: t-SNE of logits on test nodes
    try:
        _tsne_logits_plot(logits[test_mask].cpu().numpy(), y_t, plots_dir / "tsne_logits_test.png")
    except Exception as e:
        print(f"Error generating t-SNE plot: {e}")

    # Also dump a mini README of what we saved
    try:
        with open(eval_dir / "README.txt", "w") as f:
            f.write(
"""This folder contains the final-evaluation artifacts for the best checkpoint:
- plots/: confusion matrix, ROC/PR (OVR), calibration (confidence), training curves,
          robustness curves, MB-TFE diagnostics (if available), t-SNE of logits.
- tables/: predictions_test.csv (node_id, y_true, y_pred, confidence)
- np/: logits.npy, probs.npy, y.npy, and robustness arrays if configured.
"""
            )
    except Exception as e:
        print(f"Error writing README: {e}")
        
    print(f"Final evaluation complete. Results saved to {eval_dir}")
    return final
