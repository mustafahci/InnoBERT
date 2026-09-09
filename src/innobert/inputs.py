"""Input normalization with explicit row-alignment checks."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class Document:
    source_index: int
    source_id: object
    text: str
    industry: str | None
    year: int | None
    filer_name: str | None


def normalize_documents(
    data,
    *,
    industry=None,
    year=None,
    filer_name=None,
    text_col="text",
    industry_col="industry",
    year_col="year",
    filer_name_col=None,
    source_id_col=None,
    require_context=False,
):
    """Normalize a string, sequence, or pandas DataFrame into aligned documents."""
    if _is_dataframe(data):
        columns = set(data.columns)
        required = [text_col]
        if require_context and industry is None:
            required.append(industry_col)
        if require_context and year is None:
            required.append(year_col)
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required column(s): {missing}.")
        texts = data[text_col].tolist()
        industries = industry if industry is not None else _column_or_none(data, industry_col)
        years = year if year is not None else _column_or_none(data, year_col)
        filer_names = filer_name if filer_name is not None else _column_or_none(data, filer_name_col)
        source_ids = data[source_id_col].tolist() if source_id_col else list(data.index)
    else:
        texts = [data] if isinstance(data, str) else _as_list(data, "texts")
        industries, years, filer_names = industry, year, filer_name
        source_ids = list(range(len(texts)))

    if not texts:
        raise ValueError("At least one text is required.")
    texts = _validate_texts(texts)
    n = len(texts)
    industries = _broadcast(industries, n, "industry")
    years = _broadcast(years, n, "year")
    filer_names = _broadcast(filer_names, n, "filer_name")
    source_ids = _broadcast(source_ids, n, "source_id", allow_scalar=False)

    if require_context:
        missing_industry = [i for i, value in enumerate(industries) if value is None or not str(value).strip()]
        missing_year = [i for i, value in enumerate(years) if value is None]
        if missing_industry or missing_year:
            raise ValueError(
                "industry and year are required when context_mode='industry_year'. "
                f"Missing industry at row(s) {missing_industry[:10]}; missing year at row(s) {missing_year[:10]}."
            )

    normalized_years = [_normalize_year(value, i) for i, value in enumerate(years)]
    return [
        Document(
            source_index=i,
            source_id=source_ids[i],
            text=text,
            industry=None if industries[i] is None else str(industries[i]).strip(),
            year=normalized_years[i],
            filer_name=None if filer_names[i] is None else str(filer_names[i]).strip(),
        )
        for i, text in enumerate(texts)
    ]


def _is_dataframe(value):
    return hasattr(value, "columns") and hasattr(value, "iloc") and hasattr(value, "index")


def _column_or_none(frame, column):
    return frame[column].tolist() if column and column in frame.columns else None


def _as_list(value, name):
    if value is None:
        raise TypeError(f"{name} must be a string, a sequence of strings, or a pandas DataFrame.")
    try:
        return list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a string, a sequence of strings, or a pandas DataFrame.") from exc


def _validate_texts(values):
    out = []
    for i, value in enumerate(values):
        if value is None or not isinstance(value, str):
            raise TypeError(f"Text at row {i} must be a string; received {type(value).__name__}.")
        value = value.strip()
        if not value:
            raise ValueError(f"Text at row {i} is empty after stripping whitespace.")
        out.append(value)
    return out


def _broadcast(value, n, name, allow_scalar=True):
    if value is None:
        return [None] * n
    scalar_types = (str, Integral)
    if allow_scalar and isinstance(value, scalar_types):
        return [value] * n
    try:
        values = list(value)
    except TypeError as exc:
        if allow_scalar:
            return [value] * n
        raise TypeError(f"{name} must be a sequence of length {n}.") from exc
    if len(values) != n:
        raise ValueError(
            f"{name} must contain one value per input text or be a scalar. "
            f"Received {len(values)} value(s) for {n} text(s)."
        )
    return values


def _normalize_year(value, row):
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"year at row {row} must be an integer, not bool.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"year at row {row} must be integer-like; received {value!r}.") from exc
    if isinstance(value, float) and value != numeric:
        raise ValueError(f"year at row {row} must be a whole number; received {value!r}.")
    if isinstance(value, bool):
        raise TypeError(f"year at row {row} must be an integer, not bool.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"year at row {row} must be an integer; received {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"year at row {row} must be a whole number; received {value!r}.")
    if not 1800 <= numeric <= 2200:
        raise ValueError(f"year at row {row} must lie between 1800 and 2200; received {numeric}.")
    return numeric
