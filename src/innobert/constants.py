"""Scientific constants fixed by the final InnoBERT notebooks."""

LABELS = (
    "inno_product",
    "inno_process",
    "inno_organizational",
    "inno_marketing",
    "inno_businessmodel",
    "inno_sustainability",
    "inno_AI",
    "inno_uncategorized",
)

DISPLAY_LABELS = {
    "inno_product": "product",
    "inno_process": "process",
    "inno_organizational": "organizational",
    "inno_marketing": "marketing",
    "inno_businessmodel": "business_model",
    "inno_sustainability": "sustainability",
    "inno_AI": "AI",
    "inno_uncategorized": "uncategorized",
}

MAIN_CATEGORIES = {
    "inno_product": "product",
    "inno_process": "business_process",
    "inno_organizational": "business_process",
    "inno_marketing": "business_process",
    "inno_businessmodel": "business_process",
    "inno_sustainability": "sustainability",
    "inno_AI": "AI",
    "inno_uncategorized": "uncategorized",
}

MAIN_CATEGORY_ORDER = ("product", "business_process", "sustainability", "AI", "uncategorized")
CATEGORY_HIERARCHY = {
    DISPLAY_LABELS[label]: MAIN_CATEGORIES[label]
    for label in LABELS
}

DEFAULT_THRESHOLDS = dict(zip(LABELS, (0.65, 0.45, 0.55, 0.55, 0.45, 0.50, 0.50, 0.25)))

LABEL_ALIASES = {
    "product": "inno_product",
    "process": "inno_process",
    "organizational": "inno_organizational",
    "marketing": "inno_marketing",
    "business_model": "inno_businessmodel",
    "businessmodel": "inno_businessmodel",
    "sustainability": "inno_sustainability",
    "ai": "inno_AI",
    "AI": "inno_AI",
    "uncategorized": "inno_uncategorized",
}

SUPPORTED_UNITS = ("term", "noun_chunk", "sentence", "paragraph")
SUPPORTED_CONTEXT_MODES = ("auto", "industry_year", "none")
SUPPORTED_UNCATEGORIZED_RULES = ("auto", "gatekeeper", "fallback")


def resolve_thresholds(overrides=None):
    """Return defaults updated by a validated partial mapping or length-8 sequence."""
    values = DEFAULT_THRESHOLDS.copy()
    if overrides is None:
        return values
    if isinstance(overrides, dict):
        unknown = []
        for supplied, value in overrides.items():
            label = LABEL_ALIASES.get(supplied, supplied)
            if label not in values:
                unknown.append(supplied)
                continue
            values[label] = _validate_probability(value, supplied)
        if unknown:
            raise ValueError(
                f"Unknown threshold label(s): {unknown}. Expected any of {list(LABELS)} "
                f"or aliases {sorted(LABEL_ALIASES)}."
            )
        return values
    if isinstance(overrides, (str, bytes)):
        raise TypeError("thresholds must be a mapping, an 8-value sequence, or None.")
    try:
        supplied_values = list(overrides)
    except TypeError as exc:
        raise TypeError("thresholds must be a mapping, an 8-value sequence, or None.") from exc
    if len(supplied_values) != len(LABELS):
        raise ValueError(f"A threshold sequence must contain 8 values; received {len(supplied_values)}.")
    return {
        label: _validate_probability(value, label)
        for label, value in zip(LABELS, supplied_values)
    }


def _validate_probability(value, name):
    if isinstance(value, bool):
        raise TypeError(f"Threshold for {name!r} must be numeric, not bool.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Threshold for {name!r} must be numeric; received {value!r}.") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"Threshold for {name!r} must lie in [0, 1]; received {value}.")
    return value
