from __future__ import annotations

import argparse, json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from data_utils import load_csv_dataset, make_folds, standardize_train_valid_test, train_valid_split
from metrics import regression_metrics
from models.baselines import build_baseline_models
from train_utils import predict_lnn, set_seed, train_lnn


def parse_args():
    p = argparse.ArgumentParser(description="Run Improved LNN and baseline experiments.")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=40)
    return p.parse_args()


def plot_training_curve(train_losses, valid_losses, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
    ax.plot(train_losses, label="Train loss", linewidth=1.2)
    ax.plot(valid_losses, label="Validation loss", linewidth=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color("black")
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


def plot_prediction_scatter(y_true, y_pred, out_path):
    y_true = np.asarray(y_true).reshape(-1); y_pred = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=300)
    ax.scatter(y_true, y_pred, s=28, alpha=0.75, edgecolors="black", linewidths=0.3)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.tick_params(direction="in")
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color("black")
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


def permutation_importance_lnn(model, X_test, y_test, feature_names, n_repeats=30, seed=42):
    rng = np.random.default_rng(seed)
    base_pred = predict_lnn(model, X_test)
    base_rmse = regression_metrics(y_test, base_pred)["rmse"]
    rows = []
    for j, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            rng.shuffle(Xp[:, j])
            rmse = regression_metrics(y_test, predict_lnn(model, Xp))["rmse"]
            scores.append(rmse - base_rmse)
        rows.append({"feature": name, "importance_mean_rmse_increase": float(np.mean(scores)), "importance_std": float(np.std(scores))})
    return pd.DataFrame(rows).sort_values("importance_mean_rmse_increase", ascending=False)


def run_lnn_cv(dataset, args, out_dir):
    rows, preds, imps = [], [], []
    for fold_id, (train_idx, test_idx) in enumerate(make_folds(len(dataset.X), args.folds, args.seed), 1):
        inner_train_idx, valid_idx = train_valid_split(train_idx, 0.15, args.seed + fold_id)
        X_train_raw, y_train = dataset.X[inner_train_idx], dataset.y[inner_train_idx]
        X_valid_raw, y_valid = dataset.X[valid_idx], dataset.y[valid_idx]
        X_test_raw, y_test = dataset.X[test_idx], dataset.y[test_idx]
        X_train, X_valid, X_test, scaler = standardize_train_valid_test(X_train_raw, X_valid_raw, X_test_raw)
        result = train_lnn(X_train, y_train, X_valid, y_valid, hidden_size=args.hidden_size, dropout=args.dropout,
                           dt=args.dt, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay, patience=args.patience, seed=args.seed + fold_id)
        y_pred = predict_lnn(result.model, X_test)
        m = regression_metrics(y_test, y_pred); m.update({"model":"ImprovedLNN", "fold":fold_id, "best_epoch":result.best_epoch}); rows.append(m)
        preds.append(pd.DataFrame({"fold":fold_id, "y_true":y_test.reshape(-1), "y_pred":y_pred.reshape(-1)}))
        imp = permutation_importance_lnn(result.model, X_test, y_test, dataset.feature_names, seed=args.seed+fold_id); imp["fold"] = fold_id; imps.append(imp)
        plot_training_curve(result.train_losses, result.valid_losses, out_dir / f"training_curve_fold_{fold_id}.png")
        torch.save(result.model.state_dict(), out_dir / f"lnn_fold_{fold_id}.pt")
        joblib.dump(scaler, out_dir / f"scaler_fold_{fold_id}.joblib")
    pred_df = pd.concat(preds, ignore_index=True)
    pred_df.to_csv(out_dir / "predictions_lnn.csv", index=False, encoding="utf-8-sig")
    plot_prediction_scatter(pred_df["y_true"], pred_df["y_pred"], out_dir / "prediction_scatter_lnn.png")
    imp_df = pd.concat(imps, ignore_index=True)
    imp_sum = imp_df.groupby("feature", as_index=False).agg(importance_mean=("importance_mean_rmse_increase","mean"), importance_std=("importance_mean_rmse_increase","std")).sort_values("importance_mean", ascending=False)
    imp_sum.to_csv(out_dir / "permutation_importance_lnn.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def run_baseline_cv(dataset, args):
    rows = []
    for name, model in build_baseline_models(args.seed).items():
        for fold_id, (train_idx, test_idx) in enumerate(make_folds(len(dataset.X), args.folds, args.seed), 1):
            X_train, y_train = dataset.X[train_idx], dataset.y[train_idx].reshape(-1)
            X_test, y_test = dataset.X[test_idx], dataset.y[test_idx].reshape(-1)
            model.fit(X_train, y_train)
            m = regression_metrics(y_test, model.predict(X_test)); m.update({"model": name, "fold": fold_id}); rows.append(m)
    return pd.DataFrame(rows)


def main():
    args = parse_args(); set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f: json.dump(vars(args), f, ensure_ascii=False, indent=2)
    dataset = load_csv_dataset(args.data, args.features, args.target, clip_likert=True)
    metrics_all = pd.concat([run_lnn_cv(dataset, args, out_dir), run_baseline_cv(dataset, args)], ignore_index=True)
    metrics_all.to_csv(out_dir / "metrics_all_models.csv", index=False, encoding="utf-8-sig")
    summary = metrics_all.groupby("model", as_index=False).agg(r2_mean=("r2","mean"), r2_std=("r2","std"), mae_mean=("mae","mean"), mae_std=("mae","std"), rmse_mean=("rmse","mean"), rmse_std=("rmse","std")).sort_values("r2_mean", ascending=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False)); print(f"
Saved outputs to: {out_dir}")

if __name__ == "__main__": main()
