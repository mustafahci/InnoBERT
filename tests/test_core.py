import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from innobert.classifier import (
    InnoBERT,
    SUMMARY_COLUMNS,
    _select_highest_probability_assignment,
    _summarize_assignments,
    assign_labels,
)
from innobert.constants import DEFAULT_THRESHOLDS, LABELS, resolve_thresholds
from innobert.inputs import normalize_documents
from innobert.preprocessing import split_paragraphs, split_sentences


class ThresholdTests(unittest.TestCase):
    def test_defaults_preserve_final_notebook_order(self):
        self.assertEqual(tuple(DEFAULT_THRESHOLDS), LABELS)
        self.assertEqual(list(DEFAULT_THRESHOLDS.values()), [0.65, 0.45, 0.55, 0.55, 0.45, 0.50, 0.50, 0.25])

    def test_partial_alias_override(self):
        values = resolve_thresholds({"AI": 0.4, "business_model": 0.6})
        self.assertEqual(values["inno_AI"], 0.4)
        self.assertEqual(values["inno_businessmodel"], 0.6)
        self.assertEqual(values["inno_product"], 0.65)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            resolve_thresholds({"product": 1.1})


class DecisionRuleTests(unittest.TestCase):
    def test_gatekeeper_overrides_innovation_labels(self):
        probs = [0.9, 0.8, 0, 0, 0, 0, 0, 0.3]
        self.assertEqual(assign_labels(probs, DEFAULT_THRESHOLDS, "gatekeeper"), ["inno_uncategorized"])

    def test_highest_probability_label_respects_gatekeeper_assignment(self):
        probs = [0.1, 0.7, 0, 0, 0, 0, 0, 0.3]
        predicted = assign_labels(probs, DEFAULT_THRESHOLDS, "gatekeeper")
        label, probability = _select_highest_probability_assignment(probs, predicted)
        self.assertEqual(predicted, ["inno_uncategorized"])
        self.assertEqual(label, "inno_uncategorized")
        self.assertEqual(probability, 0.3)

    def test_fallback_keeps_multiple_innovation_labels(self):
        probs = [0.9, 0.8, 0, 0, 0, 0, 0, 0.9]
        self.assertEqual(
            assign_labels(probs, DEFAULT_THRESHOLDS, "fallback"),
            ["inno_product", "inno_process"],
        )

    def test_fallback_assigns_uncategorized_when_none_cross(self):
        self.assertEqual(assign_labels([0] * 8, DEFAULT_THRESHOLDS, "fallback"), ["inno_uncategorized"])


class OutputTests(unittest.TestCase):
    def test_summary_columns_are_compact_and_traceable(self):
        self.assertEqual(
            SUMMARY_COLUMNS,
            (
                "unit_id",
                "processed_text",
                "predicted_subcat_labels",
                "predicted_subcat_probs",
                "main_categories",
                "top_subcat_label",
                "top_subcat_prob",
            ),
        )

    def test_granular_and_main_categories_are_distinct(self):
        probabilities = [0.90, 0.10, 0.10, 0.10, 0.95, 0.10, 0.10, 0.10]
        summary = _summarize_assignments(
            probabilities, ["inno_product", "inno_businessmodel"]
        )
        self.assertEqual(summary[0], ["product", "business_model"])
        self.assertEqual(summary[1], [0.90, 0.95])
        self.assertEqual(summary[2], ["product", "business_process"])
        self.assertEqual(summary[3], "business_model")
        self.assertEqual(summary[4], 0.95)

    def test_predict_returns_public_category_hierarchy(self):
        classifier = InnoBERT(_FakeTokenizer(), _FakeModel(), _FakeDevice())
        fake_torch = SimpleNamespace(
            inference_mode=lambda: nullcontext(),
            sigmoid=lambda value: value,
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = classifier.predict(
                ["new platform"], industry="Software", year=2024,
                unit="term", progress=False,
            )
        self.assertEqual(
            result.loc[0, "predicted_subcat_labels"], ["product", "business_model"]
        )
        self.assertEqual(
            result.loc[0, "predicted_subcat_probs"], [0.912, 0.701]
        )
        self.assertEqual(result.loc[0, "main_categories"], ["product", "business_process"])
        self.assertEqual(result.loc[0, "top_subcat_label"], "product")
        self.assertEqual(result.loc[0, "top_subcat_prob"], 0.912)
        self.assertEqual(classifier.last_run_summary["classified_units"], 1)

    def test_full_output_preserves_probability_precision(self):
        classifier = InnoBERT(_FakeTokenizer(), _FakeModel(), _FakeDevice())
        fake_torch = SimpleNamespace(
            inference_mode=lambda: nullcontext(),
            sigmoid=lambda value: value,
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = classifier.predict(
                ["new platform"], industry="Software", year=2024,
                unit="term", output="full", progress=False,
            )
        self.assertEqual(result.loc[0, "predicted_subcat_probs"], [0.912345, 0.701234])
        self.assertEqual(result.loc[0, "top_subcat_prob"], 0.912345)
        self.assertEqual(result.loc[0, "prob_inno_product"], 0.912345)

    def test_long_term_errors_instead_of_silent_truncation(self):
        classifier = InnoBERT(_FakeTokenizer(), _FakeModel(), _FakeDevice())
        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "exceed max_length=64"):
                classifier.predict(
                    " ".join(["word"] * 100), industry="Software", year=2024,
                    unit="term", progress=False,
                )


class InputTests(unittest.TestCase):
    def test_scalar_context_broadcasts(self):
        docs = normalize_documents(["term one", "term two"], industry="Software", year=2024, require_context=True)
        self.assertEqual([d.industry for d in docs], ["Software", "Software"])
        self.assertEqual([d.year for d in docs], [2024, 2024])

    def test_aligned_context_is_preserved(self):
        docs = normalize_documents(
            ["term one", "term two"],
            industry=["Software", "Retail"],
            year=[2024, 2023],
            require_context=True,
        )
        self.assertEqual([(d.industry, d.year) for d in docs], [("Software", 2024), ("Retail", 2023)])

    def test_misaligned_context_fails(self):
        with self.assertRaisesRegex(ValueError, "1 value.*2 text"):
            normalize_documents(["a", "b"], industry=["Software"], year=2024, require_context=True)

    def test_missing_context_fails(self):
        with self.assertRaisesRegex(ValueError, "industry and year are required"):
            normalize_documents(["term"], require_context=True)

    def test_empty_text_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_documents(["   "])

    def test_dataframe_composite_source_id_and_metadata(self):
        import pandas as pd

        frame = pd.DataFrame({
            "gvkey": ["001234", "001234"],
            "fyear": [2023, 2024],
            "Industry": ["Services", "Services"],
            "long_passage": ["First filing text.", "Second filing text."],
        })
        docs = normalize_documents(
            frame,
            text_col="long_passage",
            industry_col="Industry",
            year_col="fyear",
            source_id_cols=["gvkey", "fyear"],
            metadata_cols=["gvkey", "fyear"],
        )
        self.assertEqual([doc.source_id for doc in docs], ["001234_2023", "001234_2024"])
        self.assertEqual(dict(docs[0].metadata), {"gvkey": "001234", "fyear": 2023})

    def test_duplicate_source_ids_fail(self):
        import pandas as pd

        frame = pd.DataFrame({"id": ["A", "A"], "text": ["One", "Two"]})
        with self.assertRaisesRegex(ValueError, "must be unique"):
            normalize_documents(frame, source_id_col="id")


class SegmentationTests(unittest.TestCase):
    def test_sentence_split_matches_notebook_rule(self):
        self.assertEqual(split_sentences("First sentence. Second sentence!"), ["First sentence.", "Second sentence!"])

    def test_paragraph_split_precedes_whitespace_cleanup(self):
        self.assertEqual(split_paragraphs("First paragraph.\n\nSecond   paragraph."), ["First paragraph.", "Second paragraph."])


class _FakeDevice:
    type = "cpu"

    def __str__(self):
        return "cpu"


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": list(range(len(texts.split()) + 2))}
        return {"input_ids": _FakeTensor(np.ones((len(texts), 4)))}


class _FakeModel:
    def __call__(self, **encoded):
        rows = encoded["input_ids"].values.shape[0]
        probabilities = np.tile(
            np.array([[0.912345, 0.10, 0.10, 0.10, 0.701234, 0.10, 0.10, 0.10]]),
            (rows, 1),
        )
        return SimpleNamespace(logits=_FakeTensor(probabilities))


if __name__ == "__main__":
    unittest.main()
