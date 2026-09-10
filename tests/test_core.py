import unittest

from innobert.classifier import _select_dominant_assignment, assign_labels
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

    def test_dominant_label_respects_gatekeeper_assignment(self):
        probs = [0.1, 0.7, 0, 0, 0, 0, 0, 0.3]
        predicted = assign_labels(probs, DEFAULT_THRESHOLDS, "gatekeeper")
        label, probability = _select_dominant_assignment(probs, predicted)
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


class SegmentationTests(unittest.TestCase):
    def test_sentence_split_matches_notebook_rule(self):
        self.assertEqual(split_sentences("First sentence. Second sentence!"), ["First sentence.", "Second sentence!"])

    def test_paragraph_split_precedes_whitespace_cleanup(self):
        self.assertEqual(split_paragraphs("First paragraph.\n\nSecond   paragraph."), ["First paragraph.", "Second paragraph."])


if __name__ == "__main__":
    unittest.main()
