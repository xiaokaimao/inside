from pathlib import Path
import unittest

from experiments.plot_iris_full_train_results import (
    _default_output,
    _has_method,
    _plot_context,
    _relative_change_label,
)


class FullTrainPlotMetadataTests(unittest.TestCase):
    def test_wine_context_uses_report_metadata(self) -> None:
        report = {
            "configuration": {
                "num_players": 142,
                "n_test": 36,
                "model": (
                    "sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='scale')"
                ),
            },
            "dataset": {"dataset": "wine"},
            "boundary": {"utility_calls": 286},
        }

        self.assertEqual(
            _plot_context(report),
            {
                "dataset": "Wine",
                "num_players": 142,
                "n_test": 36,
                "model": "RBF-SVM",
                "boundary_calls": 286,
            },
        )

    def test_boundary_calls_fall_back_to_two_n_plus_two(self) -> None:
        report = {
            "configuration": {
                "num_players": 120,
                "model": "lr",
            },
            "dataset": {
                "dataset": "iris",
                "test_labels": [0] * 30,
            },
        }

        context = _plot_context(report)
        self.assertEqual(context["dataset"], "Iris")
        self.assertEqual(context["n_test"], 30)
        self.assertEqual(context["model"], "logistic regression")
        self.assertEqual(context["boundary_calls"], 242)

    def test_output_name_is_derived_without_overwriting_other_dataset(self) -> None:
        input_path = Path(
            "results/json/wine_full_train_rbf_svm_frame_ofa_71k_1p42m.json"
        )
        self.assertEqual(
            _default_output(input_path),
            Path(
                "results/png/wine_full_train_rbf_svm_rmse_71k_1p42m.png"
            ).resolve(),
        )

    def test_relative_change_label_handles_regression(self) -> None:
        self.assertEqual(_relative_change_label(0.135), "13.5% lower")
        self.assertEqual(_relative_change_label(-0.021), "2.1% higher")

    def test_optional_cc_requires_every_budget(self) -> None:
        report = {
            "results_by_inner_budget": {
                "10": {"methods": {"official_cc_basic": {}}},
                "20": {"methods": {"official_cc_basic": {}}},
            }
        }
        self.assertTrue(_has_method(report, "official_cc_basic"))
        del report["results_by_inner_budget"]["20"]["methods"][
            "official_cc_basic"
        ]
        self.assertFalse(_has_method(report, "official_cc_basic"))


if __name__ == "__main__":
    unittest.main()
