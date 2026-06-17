import argparse
from pathlib import Path

import numpy as np


def fit_ridge(x_train, y_train, alpha):
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    regularizer = alpha * np.eye(x_aug.shape[1], dtype=x_train.dtype)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y_train)


def predict(x, weights):
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def standardize(train, values):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (values - mean) / std, mean, std


def main():
    parser = argparse.ArgumentParser(description="Train a baseline intent-to-Unitree-target model.")
    parser.add_argument("--dataset", default="training_data/unitree_intent_dataset.npz")
    parser.add_argument("--output", default="training_data/intent_ridge_model.npz")
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--holdout-sequence", type=int, default=-1, help="Sequence id to hold out. Default: last sequence.")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    observations = data["observations"].astype(np.float64)
    targets = np.concatenate([data["g1_wrist_targets"], data["dex3_targets"]], axis=1).astype(np.float64)
    sequence_ids = data["sequence_ids"]
    holdout = args.holdout_sequence if args.holdout_sequence >= 0 else int(sequence_ids.max())

    train_mask = sequence_ids != holdout
    test_mask = sequence_ids == holdout
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise SystemExit("Need at least one train sequence and one holdout sequence.")

    x_train, x_mean, x_std = standardize(observations[train_mask], observations[train_mask])
    x_test = (observations[test_mask] - x_mean) / x_std
    y_train = targets[train_mask]
    y_test = targets[test_mask]

    weights = fit_ridge(x_train, y_train, args.alpha)
    pred = predict(x_test, weights)
    mae = np.mean(np.abs(pred - y_test), axis=0)

    output_names = np.concatenate([data["g1_wrist_names"], data["dex3_names"]])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=weights,
        x_mean=x_mean,
        x_std=x_std,
        observation_names=data["observation_names"],
        output_names=output_names,
        holdout_sequence=holdout,
        mae=mae,
    )

    print(f"wrote {output}")
    print(f"train rows: {train_mask.sum()}  holdout rows: {test_mask.sum()}  holdout sequence: {holdout}")
    for name, value in zip(output_names, mae):
        print(f"{name}: MAE {value:.4f} rad")


if __name__ == "__main__":
    main()
