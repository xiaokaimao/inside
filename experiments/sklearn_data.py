"""Reproducible sklearn train/test splits for data-valuation games."""

from __future__ import annotations

from typing import Any

import numpy as np


def load_sklearn_train_test_split(
    dataset: str,
    *,
    test_size: float = 0.2,
    dataset_seed: int = 2024,
    samples_per_class: int | None = None,
) -> tuple[dict[str, np.ndarray | int | float], dict[str, Any]]:
    """Load, stratify, and standardize a small sklearn dataset.

    Every row in the training partition is a Shapley player.  The scaler is
    fitted on that partition only, so no test-set information enters either
    the valued features or the classifier fits performed by the utility.
    """
    from sklearn.datasets import (
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    aliases = {
        "breast_cancer": "cancer",
    }
    canonical_dataset = aliases.get(dataset, dataset)
    loaders = {
        "cancer": load_breast_cancer,
        "digits": load_digits,
        "iris": load_iris,
        "wine": load_wine,
    }
    try:
        bunch = loaders[canonical_dataset]()
    except KeyError as error:
        supported = ", ".join(sorted(loaders | aliases))
        raise ValueError(
            f"dataset must be one of: {supported}"
        ) from error

    data = np.asarray(bunch.data, dtype=np.float64)
    target = np.asarray(bunch.target, dtype=np.int64)
    source_num_observations = len(target)
    if canonical_dataset == "digits":
        if samples_per_class is None:
            samples_per_class = 100
        if samples_per_class < 1:
            raise ValueError("samples_per_class must be positive")
        subset_rng = np.random.default_rng(dataset_seed)
        subset_parts = []
        for class_label in np.unique(target):
            class_indices = np.flatnonzero(target == class_label)
            if len(class_indices) < samples_per_class:
                raise ValueError(
                    f"class {class_label} has fewer than "
                    f"{samples_per_class} observations"
                )
            subset_parts.append(
                np.sort(
                    subset_rng.choice(
                        class_indices,
                        size=samples_per_class,
                        replace=False,
                    )
                )
            )
        original_indices = np.concatenate(subset_parts)
    elif samples_per_class is not None:
        raise ValueError("samples_per_class is only supported for digits")
    else:
        samples_per_class = None
        original_indices = np.arange(len(target), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        original_indices,
        test_size=test_size,
        random_state=dataset_seed,
        stratify=target[original_indices],
    )
    label_train = target[train_indices]
    label_test = target[test_indices]

    raw_data_train = data[train_indices]
    raw_train_variances = np.var(raw_data_train, axis=0)
    kept_feature_indices = np.flatnonzero(raw_train_variances > 0)
    dropped_feature_indices = np.flatnonzero(raw_train_variances == 0)
    if len(kept_feature_indices) == 0:
        raise ValueError("the training partition has no nonconstant features")
    if len(kept_feature_indices) == data.shape[1]:
        # Keep the pre-Digits numerical path byte-for-byte stable when no
        # feature needs filtering.
        filtered_data_train = raw_data_train
        filtered_data_test = data[test_indices]
    else:
        filtered_data_train = raw_data_train[:, kept_feature_indices]
        filtered_data_test = data[test_indices][:, kept_feature_indices]
    scaler = StandardScaler()
    data_train = scaler.fit_transform(filtered_data_train)
    data_test = scaler.transform(filtered_data_test)

    num_classes = len(np.unique(target))
    game_args: dict[str, np.ndarray | int | float] = {
        "X_valued": np.asarray(data_train, dtype=np.float64),
        "y_valued": label_train,
        "X_performance": np.asarray(data_test, dtype=np.float64),
        "y_performance": label_test,
        "num_classes": num_classes,
        "learning_rate": 1.0,
        "game_seed": dataset_seed,
    }
    metadata: dict[str, Any] = {
        "dataset": canonical_dataset,
        "split": "stratified_train_test",
        "dataset_seed": dataset_seed,
        "test_size": float(test_size),
        "source_num_observations": source_num_observations,
        "subset_original_indices": original_indices.tolist(),
        "subset_num_observations": len(original_indices),
        "subset_strategy": (
            "seeded_uniform_without_replacement_per_class"
            if samples_per_class is not None
            else "all_observations"
        ),
        "subset_seed": dataset_seed if samples_per_class is not None else None,
        "subset_samples_per_class": samples_per_class,
        "subset_class_counts": np.bincount(
            target[original_indices], minlength=num_classes
        ).tolist(),
        "train_original_indices": train_indices.tolist(),
        "test_original_indices": test_indices.tolist(),
        "train_labels": label_train.tolist(),
        "test_labels": label_test.tolist(),
        "train_class_counts": np.bincount(
            label_train, minlength=num_classes
        ).tolist(),
        "test_class_counts": np.bincount(
            label_test, minlength=num_classes
        ).tolist(),
        "standardization_fitted_on": "training partition only",
        "standardization_mean": np.asarray(scaler.mean_).tolist(),
        "standardization_std": np.asarray(scaler.scale_).tolist(),
        "zero_variance_feature_indices": dropped_feature_indices.tolist(),
        "kept_feature_indices": kept_feature_indices.tolist(),
        "dropped_feature_indices": dropped_feature_indices.tolist(),
        "original_feature_names": [
            str(name) for name in bunch.feature_names
        ],
        "feature_names": [
            str(bunch.feature_names[index])
            for index in kept_feature_indices
        ],
        "dropped_feature_names": [
            str(bunch.feature_names[index])
            for index in dropped_feature_indices
        ],
        "target_names": [str(name) for name in bunch.target_names],
    }
    return game_args, metadata


def load_wine_train_test_split(
    *,
    test_size: float = 0.2,
    dataset_seed: int = 2024,
) -> tuple[dict[str, np.ndarray | int | float], dict[str, Any]]:
    """Compatibility convenience wrapper for the Wine dataset."""
    return load_sklearn_train_test_split(
        "wine",
        test_size=test_size,
        dataset_seed=dataset_seed,
    )


def load_cancer_train_test_split(
    *,
    test_size: float = 0.2,
    dataset_seed: int = 2024,
) -> tuple[dict[str, np.ndarray | int | float], dict[str, Any]]:
    """Compatibility convenience wrapper for Breast Cancer Wisconsin."""
    return load_sklearn_train_test_split(
        "cancer",
        test_size=test_size,
        dataset_seed=dataset_seed,
    )


# Descriptive alias for callers that prefer sklearn's full dataset name.
load_breast_cancer_train_test_split = load_cancer_train_test_split


def load_digits_train_test_split(
    *,
    test_size: float = 0.2,
    dataset_seed: int = 2024,
    samples_per_class: int = 100,
) -> tuple[dict[str, np.ndarray | int | float], dict[str, Any]]:
    """Load a seeded balanced 1,000-observation Digits subset."""
    return load_sklearn_train_test_split(
        "digits",
        test_size=test_size,
        dataset_seed=dataset_seed,
        samples_per_class=samples_per_class,
    )
