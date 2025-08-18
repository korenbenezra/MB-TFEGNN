This folder contains the final-evaluation artifacts for the best checkpoint:
- plots/: confusion matrix, ROC/PR (OVR), calibration (confidence), training curves,
          robustness curves, MB-TFE diagnostics (if available), t-SNE of logits.
- tables/: predictions_test.csv (node_id, y_true, y_pred, confidence)
- np/: logits.npy, probs.npy, y.npy, and robustness arrays if configured.
