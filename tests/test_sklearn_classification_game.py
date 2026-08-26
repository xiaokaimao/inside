from __future__ import annotations

import unittest

import numpy as np

from experiments.iris_sklearn_game import (
    IrisSklearnGame,
    SklearnClassificationGame,
)
from experiments.sklearn_data import (
    load_digits_train_test_split,
    load_sklearn_train_test_split,
)


class SklearnDatasetTests(unittest.TestCase):
    def test_wine_split_uses_every_training_row_and_is_stratified(self) -> None:
        game_args, metadata = load_sklearn_train_test_split(
            "wine",
            test_size=0.2,
            dataset_seed=2024,
        )
        self.assertEqual(game_args["X_valued"].shape, (142, 13))
        self.assertEqual(game_args["X_performance"].shape, (36, 13))
        self.assertEqual(metadata["train_class_counts"], [47, 57, 38])
        self.assertEqual(metadata["test_class_counts"], [12, 14, 10])
        self.assertEqual(
            len(metadata["train_original_indices"]),
            len(set(metadata["train_original_indices"])),
        )
        self.assertEqual(
            set(metadata["train_original_indices"]).intersection(
                metadata["test_original_indices"]
            ),
            set(),
        )

    def test_wine_scaler_is_fitted_only_on_training_partition(self) -> None:
        from sklearn.datasets import load_wine

        game_args, metadata = load_sklearn_train_test_split(
            "wine",
            test_size=0.2,
            dataset_seed=2024,
        )
        raw = np.asarray(load_wine().data, dtype=np.float64)
        train_indices = np.asarray(metadata["train_original_indices"])
        test_indices = np.asarray(metadata["test_original_indices"])
        mean = raw[train_indices].mean(axis=0)
        scale = raw[train_indices].std(axis=0)

        np.testing.assert_allclose(metadata["standardization_mean"], mean)
        np.testing.assert_allclose(metadata["standardization_std"], scale)
        np.testing.assert_allclose(
            game_args["X_valued"].mean(axis=0), 0.0, atol=5e-15
        )
        np.testing.assert_allclose(
            game_args["X_valued"].std(axis=0), 1.0, atol=2e-15
        )
        np.testing.assert_allclose(
            game_args["X_performance"],
            (raw[test_indices] - mean) / scale,
            rtol=0.0,
            atol=2e-15,
        )

    def test_cancer_split_uses_every_training_row_and_is_stratified(
        self,
    ) -> None:
        game_args, metadata = load_sklearn_train_test_split(
            "cancer",
            test_size=0.2,
            dataset_seed=2024,
        )
        self.assertEqual(game_args["X_valued"].shape, (455, 30))
        self.assertEqual(game_args["X_performance"].shape, (114, 30))
        self.assertEqual(metadata["train_class_counts"], [170, 285])
        self.assertEqual(metadata["test_class_counts"], [42, 72])
        self.assertEqual(
            len(metadata["train_original_indices"]),
            len(set(metadata["train_original_indices"])),
        )
        self.assertEqual(
            set(metadata["train_original_indices"]).intersection(
                metadata["test_original_indices"]
            ),
            set(),
        )

    def test_cancer_scaler_is_fitted_only_on_training_partition(self) -> None:
        from sklearn.datasets import load_breast_cancer

        game_args, metadata = load_sklearn_train_test_split(
            "cancer",
            test_size=0.2,
            dataset_seed=2024,
        )
        raw = np.asarray(load_breast_cancer().data, dtype=np.float64)
        train_indices = np.asarray(metadata["train_original_indices"])
        test_indices = np.asarray(metadata["test_original_indices"])
        mean = raw[train_indices].mean(axis=0)
        scale = raw[train_indices].std(axis=0)

        np.testing.assert_allclose(metadata["standardization_mean"], mean)
        np.testing.assert_allclose(metadata["standardization_std"], scale)
        np.testing.assert_allclose(
            game_args["X_valued"].mean(axis=0), 0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            game_args["X_valued"].std(axis=0), 1.0, atol=2e-14
        )
        np.testing.assert_allclose(
            game_args["X_performance"],
            (raw[test_indices] - mean) / scale,
            rtol=0.0,
            atol=2e-14,
        )

    def test_breast_cancer_alias_uses_canonical_metadata(self) -> None:
        cancer_args, cancer_metadata = load_sklearn_train_test_split(
            "cancer"
        )
        alias_args, alias_metadata = load_sklearn_train_test_split(
            "breast_cancer"
        )
        self.assertEqual(cancer_metadata, alias_metadata)
        self.assertEqual(alias_metadata["dataset"], "cancer")
        for key in (
            "X_valued",
            "y_valued",
            "X_performance",
            "y_performance",
        ):
            np.testing.assert_array_equal(cancer_args[key], alias_args[key])

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "breast_cancer, cancer, digits, iris, wine"
        ):
            load_sklearn_train_test_split("unknown")


class DigitsDatasetTests(unittest.TestCase):
    def test_balanced_subset_precedes_stratified_split(self) -> None:
        from sklearn.datasets import load_digits

        game_args, metadata = load_digits_train_test_split(
            test_size=0.2,
            dataset_seed=2024,
        )
        raw_target = np.asarray(load_digits().target, dtype=np.int64)
        subset_indices = np.asarray(metadata["subset_original_indices"])
        train_indices = np.asarray(metadata["train_original_indices"])
        test_indices = np.asarray(metadata["test_original_indices"])

        self.assertEqual(metadata["source_num_observations"], 1797)
        self.assertEqual(metadata["subset_num_observations"], 1000)
        self.assertEqual(metadata["subset_samples_per_class"], 100)
        self.assertEqual(metadata["subset_seed"], 2024)
        self.assertEqual(metadata["subset_class_counts"], [100] * 10)
        self.assertEqual(metadata["train_class_counts"], [80] * 10)
        self.assertEqual(metadata["test_class_counts"], [20] * 10)
        self.assertEqual(game_args["X_valued"].shape, (800, 60))
        self.assertEqual(game_args["X_performance"].shape, (200, 60))
        self.assertEqual(len(np.unique(subset_indices)), 1000)
        self.assertEqual(
            set(train_indices) | set(test_indices), set(subset_indices)
        )
        self.assertFalse(set(train_indices) & set(test_indices))
        np.testing.assert_array_equal(
            np.bincount(raw_target[subset_indices], minlength=10),
            np.full(10, 100),
        )

    def test_seeded_subset_and_split_indices_are_deterministic(self) -> None:
        _, first = load_sklearn_train_test_split(
            "digits", dataset_seed=2024
        )
        _, second = load_sklearn_train_test_split(
            "digits", dataset_seed=2024
        )
        _, other_seed = load_sklearn_train_test_split(
            "digits", dataset_seed=2025
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["subset_original_indices"][:10],
            [0, 36, 49, 55, 72, 78, 79, 126, 150, 160],
        )
        self.assertNotEqual(
            first["subset_original_indices"],
            other_seed["subset_original_indices"],
        )

    def test_digits_scaler_is_fitted_only_on_training_partition(self) -> None:
        from sklearn.datasets import load_digits

        game_args, metadata = load_sklearn_train_test_split(
            "digits", dataset_seed=2024
        )
        raw = np.asarray(load_digits().data, dtype=np.float64)
        train_indices = np.asarray(metadata["train_original_indices"])
        test_indices = np.asarray(metadata["test_original_indices"])
        raw_scale = raw[train_indices].std(axis=0)
        kept = np.flatnonzero(raw_scale > 0)
        dropped = np.flatnonzero(raw_scale == 0)
        mean = raw[train_indices][:, kept].mean(axis=0)
        scale = raw[train_indices][:, kept].std(axis=0)

        np.testing.assert_allclose(metadata["standardization_mean"], mean)
        np.testing.assert_allclose(metadata["standardization_std"], scale)
        np.testing.assert_array_equal(metadata["kept_feature_indices"], kept)
        np.testing.assert_array_equal(
            metadata["dropped_feature_indices"], dropped
        )
        self.assertEqual(dropped.tolist(), [0, 32, 39, 56])
        self.assertEqual(len(metadata["feature_names"]), 60)
        self.assertEqual(len(metadata["dropped_feature_names"]), 4)
        self.assertTrue(np.isfinite(game_args["X_valued"]).all())
        self.assertTrue(np.isfinite(game_args["X_performance"]).all())
        np.testing.assert_allclose(
            game_args["X_performance"],
            (raw[test_indices][:, kept] - mean) / scale,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            game_args["X_valued"].mean(axis=0),
            0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            game_args["X_valued"].std(axis=0),
            1.0,
            atol=2e-14,
        )


class SklearnClassificationGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        game_args, _ = load_sklearn_train_test_split("wine")
        cls.game_args = game_args | {
            "model": "rbf_svm",
            "regularization": 1.0,
        }
        cls.game = SklearnClassificationGame(**cls.game_args)

    def test_legacy_iris_name_is_a_compatibility_alias(self) -> None:
        self.assertIs(IrisSklearnGame, SklearnClassificationGame)

    def test_empty_coalition_uses_best_test_constant(self) -> None:
        empty = np.zeros(self.game.num_players, dtype=bool)
        expected = 14 / 36
        self.assertEqual(self.game.evaluate(empty), expected)

    def test_single_class_coalition_predicts_that_class(self) -> None:
        coalition = np.asarray(self.game_args["y_valued"]) == 2
        expected = 10 / 36
        self.assertEqual(self.game.evaluate(coalition), expected)

    def test_full_coalition_matches_direct_rbf_svc_accuracy(self) -> None:
        from sklearn.svm import SVC

        full = np.ones(self.game.num_players, dtype=bool)
        direct = SVC(C=1.0, kernel="rbf", gamma="scale")
        direct.fit(
            self.game_args["X_valued"], self.game_args["y_valued"]
        )
        expected = float(
            np.mean(
                direct.predict(self.game_args["X_performance"])
                == self.game_args["y_performance"]
            )
        )
        self.assertEqual(self.game.evaluate(full), expected)


class CancerClassificationGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        game_args, _ = load_sklearn_train_test_split("cancer")
        cls.game_args = game_args | {
            "model": "rbf_svm",
            "regularization": 1.0,
        }
        cls.game = SklearnClassificationGame(**cls.game_args)

    def test_empty_coalition_uses_best_test_constant(self) -> None:
        empty = np.zeros(self.game.num_players, dtype=bool)
        self.assertEqual(self.game.evaluate(empty), 72 / 114)

    def test_single_class_coalition_predicts_that_class(self) -> None:
        coalition = np.asarray(self.game_args["y_valued"]) == 0
        self.assertEqual(self.game.evaluate(coalition), 42 / 114)

    def test_full_coalition_matches_direct_rbf_svc_accuracy(self) -> None:
        from sklearn.svm import SVC

        full = np.ones(self.game.num_players, dtype=bool)
        direct = SVC(C=1.0, kernel="rbf", gamma="scale")
        direct.fit(
            self.game_args["X_valued"], self.game_args["y_valued"]
        )
        expected = float(
            np.mean(
                direct.predict(self.game_args["X_performance"])
                == self.game_args["y_performance"]
            )
        )
        self.assertEqual(self.game.evaluate(full), expected)


class DigitsClassificationGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        game_args, _ = load_sklearn_train_test_split("digits")
        cls.game_args = game_args | {
            "model": "rbf_svm",
            "regularization": 1.0,
        }
        cls.game = SklearnClassificationGame(**cls.game_args)

    def test_empty_coalition_uses_best_test_constant(self) -> None:
        empty = np.zeros(self.game.num_players, dtype=bool)
        self.assertEqual(self.game.evaluate(empty), 20 / 200)

    def test_single_class_coalition_predicts_that_class(self) -> None:
        coalition = np.asarray(self.game_args["y_valued"]) == 7
        self.assertEqual(self.game.evaluate(coalition), 20 / 200)

    def test_full_coalition_matches_direct_rbf_svc_accuracy(self) -> None:
        from sklearn.svm import SVC

        full = np.ones(self.game.num_players, dtype=bool)
        direct = SVC(C=1.0, kernel="rbf", gamma="scale")
        direct.fit(
            self.game_args["X_valued"], self.game_args["y_valued"]
        )
        expected = float(
            np.mean(
                direct.predict(self.game_args["X_performance"])
                == self.game_args["y_performance"]
            )
        )
        self.assertEqual(self.game.evaluate(full), expected)


if __name__ == "__main__":
    unittest.main()
