"""Stable inference interface for the eight-label InnoBERT classifier."""

from pathlib import Path
from math import ceil
import re
from time import perf_counter
import warnings

from .constants import (
    LABELS,
    DISPLAY_LABELS,
    MAIN_CATEGORIES,
    MAIN_CATEGORY_ORDER,
    SUPPORTED_CONTEXT_MODES,
    SUPPORTED_UNCATEGORIZED_RULES,
    SUPPORTED_UNITS,
    resolve_thresholds,
)
from .inputs import normalize_documents
from .preprocessing import expand_documents
from .progress import progress_iter, resolve_progress


SUMMARY_COLUMNS = (
    "unit_id",
    "processed_text",
    "predicted_subcat_labels",
    "predicted_subcat_probs",
    "main_categories",
    "top_subcat_label",
    "top_subcat_prob",
)


class InnoBERT:
    """Load InnoBERT and classify terms, noun chunks, sentences, or paragraphs."""

    def __init__(self, tokenizer, model, device, *, spacy_model="en_core_web_lg"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.spacy_model = spacy_model
        self.last_run_summary = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        *,
        device="auto",
        spacy_model="en_core_web_lg",
        cpu_threads=None,
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
    ):
        """Load a local directory or a public/private Hugging Face model repository."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install InnoBERT runtime dependencies with `pip install innobert`.") from exc

        path = Path(model_name_or_path)
        if path.exists():
            if path.is_file():
                raise ValueError(
                    "model_name_or_path must be an extracted Hugging Face model directory, not a ZIP or weight file."
                )
            required = ("config.json", "tokenizer_config.json")
            missing = [name for name in required if not (path / name).exists()]
            if missing:
                raise ValueError(f"Local model directory is missing required file(s): {missing}.")

        resolved_device = _resolve_device(device, torch)
        if resolved_device.type == "cpu" and cpu_threads is not None:
            if not isinstance(cpu_threads, int) or cpu_threads < 1:
                raise ValueError("cpu_threads must be a positive integer or None.")
            torch.set_num_threads(cpu_threads)
        hub_options = {
            "revision": revision,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        hub_options = {key: value for key, value in hub_options.items() if value is not None}
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hub_options)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **hub_options)
        if int(model.config.num_labels) != len(LABELS):
            raise ValueError(
                f"InnoBERT requires exactly 8 classifier outputs; loaded model has {model.config.num_labels}."
            )
        model.config.id2label = dict(enumerate(LABELS))
        model.config.label2id = {label: i for i, label in enumerate(LABELS)}
        model.to(resolved_device)
        model.eval()
        return cls(tokenizer, model, resolved_device, spacy_model=spacy_model)

    def predict(
        self,
        data,
        *,
        industry=None,
        year=None,
        filer_name=None,
        unit="term",
        context_mode="auto",
        thresholds=None,
        uncategorized_rule="auto",
        batch_size=None,
        max_length=None,
        long_text_strategy="auto",
        stride=32,
        min_sentence_words=1,
        text_col="text",
        industry_col="industry",
        year_col="year",
        filer_name_col=None,
        source_id_col=None,
        source_id_cols=None,
        metadata_cols=None,
        output="summary",
        include_model_input=False,
        progress="auto",
    ):
        """Classify one text, aligned lists, or rows of a pandas DataFrame.

        `term` and `noun_chunk` use the training prompt by default. `sentence` and
        `paragraph` follow the conference-call notebook and use raw text by default.
        """
        _validate_choice("unit", unit, SUPPORTED_UNITS)
        _validate_choice("context_mode", context_mode, SUPPORTED_CONTEXT_MODES)
        _validate_choice("uncategorized_rule", uncategorized_rule, SUPPORTED_UNCATEGORIZED_RULES)
        _validate_choice("output", output, ("summary", "full"))
        resolve_progress(progress)
        summary_metadata_cols = [metadata_cols] if isinstance(metadata_cols, str) else list(metadata_cols or [])
        if context_mode == "auto":
            context_mode = "industry_year" if unit in {"term", "noun_chunk"} else "none"
        if uncategorized_rule == "auto":
            uncategorized_rule = "gatekeeper" if unit in {"term", "noun_chunk"} else "fallback"
        if long_text_strategy == "auto":
            long_text_strategy = "window" if unit == "paragraph" else "error"
        _validate_choice("long_text_strategy", long_text_strategy, ("truncate", "window", "error"))
        if not isinstance(stride, int) or stride < 0:
            raise ValueError("stride must be a nonnegative integer.")

        documents = normalize_documents(
            data,
            industry=industry,
            year=year,
            filer_name=filer_name,
            text_col=text_col,
            industry_col=industry_col,
            year_col=year_col,
            filer_name_col=filer_name_col,
            source_id_col=source_id_col,
            source_id_cols=source_id_cols,
            metadata_cols=metadata_cols,
            require_context=context_mode == "industry_year",
        )
        if unit == "paragraph":
            _warn_if_paragraph_boundaries_missing(documents)
        started = perf_counter()
        units = expand_documents(
            documents,
            unit,
            min_sentence_words=min_sentence_words,
            spacy_model=self.spacy_model,
            progress=progress,
        )
        if not units:
            raise ValueError(
                f"No {unit} units were produced. Check the text and preprocessing settings "
                "(especially min_sentence_words or noun-chunk filters)."
            )
        model_inputs = [
            _format_training_input(record.processed_text, record.industry, record.year)
            if context_mode == "industry_year" else record.processed_text
            for record in units
        ]
        if max_length is None:
            max_length = 64 if unit in {"term", "noun_chunk"} else 160
        if not isinstance(max_length, int) or not 4 <= max_length <= 512:
            raise ValueError("max_length must be an integer between 4 and the model limit of 512.")
        if stride >= max_length - 2:
            raise ValueError("stride must be smaller than max_length - 2.")
        if batch_size is None:
            batch_size = 256 if self.device.type == "cuda" else 32
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        probabilities, token_counts, window_counts, model_batches = self._probabilities(
            model_inputs,
            batch_size=batch_size,
            max_length=max_length,
            strategy=long_text_strategy,
            stride=stride,
            progress=progress,
        )
        threshold_map = resolve_thresholds(thresholds)
        rows = []
        for record, model_input, probs, token_count, window_count in zip(
            units, model_inputs, probabilities, token_counts, window_counts
        ):
            predicted = assign_labels(probs, threshold_map, uncategorized_rule)
            (
                predicted_display,
                predicted_probabilities,
                main_categories,
                highest_probability_label,
                highest_probability,
            ) = _summarize_assignments(probs, predicted)
            row = {
                **dict(record.metadata),
                "source_index": record.source_index,
                "source_id": record.source_id,
                "unit_id": f"{record.source_id}:{record.unit_index}",
                "unit_index": record.unit_index,
                "unit_type": record.unit_type,
                "source_text": record.source_text,
                "processed_text": record.processed_text,
                "industry": record.industry,
                "year": record.year,
                "token_count": token_count,
                "window_count": window_count,
                "truncated_or_windowed": token_count > max_length,
                "long_text_action": (
                    "windowed" if window_count > 1 else
                    "truncated" if long_text_strategy == "truncate" and token_count > max_length else
                    "none"
                ),
                "uncategorized_rule": uncategorized_rule,
                "assignment_reason": _assignment_reason(probs, predicted, threshold_map, uncategorized_rule),
                **{f"prob_{label}": float(prob) for label, prob in zip(LABELS, probs)},
                "predicted_subcat_labels": predicted_display,
                "predicted_subcat_probs": predicted_probabilities,
                "main_categories": main_categories,
                "top_subcat_label": highest_probability_label,
                "top_subcat_prob": highest_probability,
                "device_used": str(self.device),
            }
            if include_model_input:
                row["model_input"] = model_input
            rows.append(row)
        import pandas as pd
        result = pd.DataFrame(rows)
        self.last_run_summary = {
            "source_documents": len(documents),
            "extracted_units": len(units),
            "classified_units": len(units),
            "model_batches": model_batches,
            "windowed_units": sum(count > 1 for count in window_counts),
            "truncated_units": sum(
                count > max_length for count in token_counts
            ) if long_text_strategy == "truncate" else 0,
            "device": str(self.device),
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
        if output == "summary":
            columns = list(dict.fromkeys([*summary_metadata_cols, *SUMMARY_COLUMNS]))
            if include_model_input:
                columns.append("model_input")
            summary = result[columns].copy()
            summary["predicted_subcat_probs"] = summary["predicted_subcat_probs"].map(
                lambda values: [round(value, 3) for value in values]
            )
            summary["top_subcat_prob"] = summary["top_subcat_prob"].round(3)
            return summary
        return result

    def _probabilities(self, texts, *, batch_size, max_length, strategy, stride, progress="auto"):
        import numpy as np
        import torch

        token_counts = [len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
        if strategy == "error":
            too_long = [i for i, count in enumerate(token_counts) if count > max_length]
            if too_long:
                raise ValueError(
                    f"{len(too_long)} input unit(s) exceed max_length={max_length}; first affected row(s): {too_long[:10]}. "
                    "Use long_text_strategy='window' or 'truncate'."
                )
            strategy = "truncate"

        if strategy == "truncate":
            all_probs = []
            starts = range(0, len(texts), batch_size)
            iterator = progress_iter(
                starts, total=ceil(len(texts) / batch_size),
                description="Classifying units", progress=progress,
            )
            for start in iterator:
                encoded = self.tokenizer(
                    texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    all_probs.append(torch.sigmoid(self.model(**encoded).logits).cpu().numpy())
            return np.vstack(all_probs), token_counts, [1] * len(texts), ceil(len(texts) / batch_size)

        chunks, owners = [], []
        for owner, text in enumerate(texts):
            for chunk in _build_complete_token_windows(
                self.tokenizer, text, max_length=max_length, stride=stride
            ):
                chunks.append(chunk)
                owners.append(owner)
        aggregate = np.full((len(texts), len(LABELS)), -np.inf, dtype=float)
        starts = range(0, len(chunks), batch_size)
        iterator = progress_iter(
            starts, total=ceil(len(chunks) / batch_size),
            description="Classifying windows", progress=progress,
        )
        for start in iterator:
            batch_chunks = chunks[start:start + batch_size]
            encoded = self.tokenizer.pad(batch_chunks, padding=True, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                probs = torch.sigmoid(self.model(**encoded).logits).cpu().numpy()
            for owner, row in zip(owners[start:start + batch_size], probs):
                aggregate[owner] = np.maximum(aggregate[owner], row)
        window_counts = np.bincount(owners, minlength=len(texts)).tolist()
        return aggregate, token_counts, window_counts, ceil(len(chunks) / batch_size)


def _build_complete_token_windows(tokenizer, text, *, max_length, stride):
    """Build overlapping windows while guaranteeing coverage of every content token.

    This deliberately avoids tokenizer overflow metadata, whose behavior has varied
    across tokenizer and Transformers versions. The text is tokenized once without
    special tokens, then sliced explicitly and wrapped with model-specific specials.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        raise RuntimeError("Expected one unbatched token sequence while constructing long-text windows.")
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_capacity = max_length - special_count
    if content_capacity < 1:
        raise ValueError(
            f"max_length={max_length} leaves no room for content after {special_count} special tokens."
        )
    if stride >= content_capacity:
        raise ValueError(
            f"stride must be smaller than the content capacity ({content_capacity}) "
            f"for max_length={max_length}."
        )
    if not token_ids:
        starts = [0]
        slices = [[]]
    else:
        step = content_capacity - stride
        starts = list(range(0, len(token_ids), step))
        slices = [token_ids[start:start + content_capacity] for start in starts]
        if slices[-1] and len(slices) > 1 and len(slices[-1]) <= stride:
            starts.pop()
            slices.pop()
    windows = [
        dict(tokenizer.prepare_for_model(
            token_slice,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
        ))
        for token_slice in slices
    ]
    if token_ids and starts[-1] + len(slices[-1]) != len(token_ids):
        raise RuntimeError("Internal long-text coverage check failed: final content token was omitted.")
    if any(len(window["input_ids"]) > max_length for window in windows):
        raise RuntimeError("Internal long-text coverage check failed: a window exceeds max_length.")
    return windows


def assign_labels(probabilities, thresholds, rule):
    """Apply the notebook-defined multi-label decision rule."""
    if len(probabilities) != len(LABELS):
        raise ValueError(f"Expected 8 probabilities; received {len(probabilities)}.")
    if rule not in {"gatekeeper", "fallback"}:
        raise ValueError("rule must be 'gatekeeper' or 'fallback'.")
    uncat = LABELS[-1]
    if rule == "gatekeeper" and probabilities[-1] >= thresholds[uncat]:
        return [uncat]
    selected = [
        label for label, probability in zip(LABELS[:7], probabilities[:7])
        if probability >= thresholds[label]
    ]
    return selected or [uncat]


def _assignment_reason(probabilities, predicted_labels, thresholds, rule):
    uncat_probability = float(probabilities[-1])
    uncat_threshold = float(thresholds[LABELS[-1]])
    if rule == "gatekeeper" and predicted_labels == [LABELS[-1]]:
        return (
            f"uncategorized gatekeeper: {uncat_probability:.3f} >= {uncat_threshold:.3f}; "
            "innovation labels suppressed"
        )
    if predicted_labels == [LABELS[-1]]:
        return "fallback: no innovation subcategory met its threshold"
    return "assigned subcategory threshold(s) met"


def _select_highest_probability_assignment(probabilities, predicted_labels):
    """Return the highest-probability label among the labels actually assigned."""
    if len(probabilities) != len(LABELS):
        raise ValueError(f"Expected 8 probabilities; received {len(probabilities)}.")
    if not predicted_labels:
        raise ValueError("predicted_labels must contain at least one assigned label.")
    try:
        assigned_indices = [LABELS.index(label) for label in predicted_labels]
    except ValueError as exc:
        raise ValueError("predicted_labels contains an unknown InnoBERT label.") from exc
    highest_index = max(assigned_indices, key=lambda i: probabilities[i])
    return LABELS[highest_index], float(probabilities[highest_index])


def _summarize_assignments(probabilities, predicted_labels):
    highest_label, highest_probability = _select_highest_probability_assignment(
        probabilities, predicted_labels
    )
    predicted_display = [DISPLAY_LABELS[label] for label in predicted_labels]
    predicted_probabilities = [
        float(probabilities[LABELS.index(label)]) for label in predicted_labels
    ]
    assigned_main = {MAIN_CATEGORIES[label] for label in predicted_labels}
    main_categories = [category for category in MAIN_CATEGORY_ORDER if category in assigned_main]
    return (
        predicted_display,
        predicted_probabilities,
        main_categories,
        DISPLAY_LABELS[highest_label],
        highest_probability,
    )


def _format_training_input(text, industry, year):
    return f"Industry: {industry}. Year: {year}. The term is: <TERM> {text} </TERM>."


def _warn_if_paragraph_boundaries_missing(documents):
    affected = [
        document.source_id for document in documents
        if len(document.text) > 20_000 and not re.search(r"\n\s*\n", document.text)
    ]
    if affected:
        warnings.warn(
            "Long source text appears to contain no blank-line paragraph boundaries for source_id(s) "
            f"{affected[:10]}. Paragraph mode will treat each source as one paragraph and aggregate "
            "category-wise maxima across many windows. Preserve paragraph or section boundaries, or "
            "use sentence/noun_chunk mode.",
            UserWarning,
            stacklevel=2,
        )


def _resolve_device(requested, torch):
    requested = str(requested)
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is False. Use device='cpu' or 'auto'.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("An MPS device was requested, but it is unavailable. Use device='cpu' or 'auto'.")
    try:
        return torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'.") from exc


def _validate_choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}; received {value!r}.")
