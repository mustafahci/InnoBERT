"""Text segmentation and the final 2026 noun-chunk extraction rules."""

import re
from dataclasses import dataclass
from importlib.resources import files

from .inputs import Document


RANKING_ADJECTIVES = {
    "best", "best-in-class", "bestinclass", "breakthrough", "cutting-edge", "cuttingedge",
    "disruptive", "dominant", "first", "first-ever", "firstever", "game-changing",
    "gamechanging", "greatest", "groundbreaking", "highest", "largest", "latest", "leading",
    "lowest", "most", "numberone", "novel", "primary", "record-breaking", "revolutionary",
    "single", "top", "ultimate", "unprecedented", "world-class", "worldclass",
}

QUALITY_ADJECTIVES = {
    "advancing", "advanced", "adverse", "average", "best", "better", "broad", "broader",
    "considerable", "elite", "enhanced", "exceptional", "exclusive", "expanding", "fast",
    "faster", "fastest", "great", "greater", "growing", "high", "higher", "highest",
    "improved", "increased", "increasing", "large", "larger", "little", "less", "superior",
    "long", "low", "lower", "lowest", "many", "new", "numerous", "outstanding", "premium",
    "principal", "small", "strong", "substantial", "ultimate", "unique", "unparalleled",
}

VAGUE_ADJECTIVES = {
    "additional", "annual", "applied", "appropriate", "certain", "common", "complete",
    "comprehensive", "core", "critical", "current", "different", "effective", "essential",
    "exclusive", "extensive", "full", "fundamental", "general", "important", "integral", "key",
    "limited", "major", "material", "multiple", "necessary", "notable", "other", "overall",
    "own", "particular", "potential", "present", "prior", "proprietary", "recent", "relevant",
    "regulatory", "several", "significant", "similar", "specific", "standard", "strategic",
    "subject", "such", "sufficient", "subsequent", "total", "traditional", "typical",
    "underlying", "various",
}

OTHER_FREQUENT = {
    "acceptable", "additionally", "administrative", "also", "annual", "annually", "applicable",
    "associated", "attractive", "available", "billion", "chief", "civil", "clear", "collective",
    "commercial", "comparable", "competitive", "complex", "consolidated", "consistent",
    "continuous", "continued", "corporate", "custom", "dedicated", "dependent", "direct",
    "disparate", "economic", "emerging", "entire", "equal", "even", "existing", "few", "final",
    "financial", "fiscal", "five", "following", "foreign", "forward", "four", "fourth", "free",
    "fully", "fulltime", "further", "future", "generally", "geographic", "global", "good",
    "governmental", "gross", "highly", "historical", "industry", "influential", "institutional",
    "internal", "international", "joint", "last", "least", "legal", "less", "likely", "local",
    "longer", "materially", "maximum", "million", "minimum", "more", "much", "multi", "national",
    "net", "often", "one", "ongoing", "only", "open", "operational", "original", "outside",
    "past", "periodic", "personal", "possible", "previously", "primarily", "principally",
    "private", "professional", "public", "pursuant", "qualified", "quarterly", "rapid", "raw",
    "real", "reasonable", "reasonably", "recently", "regional", "relative", "relatively",
    "respective", "respectively", "responsible", "retail", "safe", "same", "seasonal", "second",
    "senior", "separate", "seven", "short", "significantly", "six", "smaller", "special",
    "specifically", "substantially", "successful", "successfully", "statutory", "ten", "third",
    "thirdparty", "three", "two", "typically", "unable", "valuable", "vast", "weekly", "wholly",
    "wide", "worldwide",
}

CORPORATE_SELF_REFERENCES = {
    "company", "corporation", "corp", "inc", "incorporated", "registrant", "subsidiary",
    "subsidiaries", "affiliate", "affiliates", "group", "holdings",
}

_NLP_CACHE = {}


@dataclass(frozen=True)
class UnitRecord:
    source_index: int
    source_id: object
    unit_index: int
    unit_type: str
    source_text: str
    processed_text: str
    industry: str | None
    year: int | None


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def split_sentences(text, min_words=1):
    """Apply the conference-call notebook's deterministic sentence splitter."""
    text = clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= min_words]


def split_paragraphs(text):
    """Split blank-line-delimited paragraphs before normalizing internal whitespace."""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    paragraphs = re.split(r"\n\s*\n+", normalized)
    return [clean_text(p) for p in paragraphs if clean_text(p)]


def extract_noun_chunks(text, filer_name=None, spacy_model="en_core_web_lg"):
    """Extract terms using the final 2026 noun-chunk algorithm."""
    nlp = _get_nlp(spacy_model)
    stoplist = _modifying_adjectives()
    filer_tokens = set(re.findall(r"[A-Za-z0-9-]+", (filer_name or "").lower()))
    self_references = CORPORATE_SELF_REFERENCES | filer_tokens
    doc = nlp(re.sub(r"[#*]", "", text))
    phrases = set()

    def stoplisted(word):
        word = word.lower()
        return word in stoplist or word.replace("-", "") in stoplist

    def token_passes(token, is_conj_partner=False):
        if token.is_stop or token.is_punct or token.like_num or token.like_email or token.like_url:
            return False
        if stoplisted(token.text) or token.pos_ in {"CCONJ", "DET"} or token.dep_ == "poss":
            return False
        if token.text.lower() in self_references or token.ent_type_ == "PERSON":
            return False
        allowed = {"amod", "compound", "conj"} if is_conj_partner else {"amod", "compound"}
        return token.dep_ in allowed or token.pos_ in {"NOUN", "PROPN"}

    def build_words(tokens, is_conj_partner=False):
        words = []
        for token in tokens:
            if not token_passes(token, is_conj_partner=is_conj_partner):
                continue
            source = token._.lemma() if token.pos_ in {"NOUN", "PROPN"} and token.dep_ != "compound" else token.text.lower()
            cleaned = _clean_token(source)
            if cleaned:
                words.append(cleaned)
        return words

    def finalize(words):
        words = words[-4:]
        return " ".join(words) if words else None

    def chunk_phrases(chunk):
        tokens = list(chunk)
        cc_positions = [i for i, token in enumerate(tokens) if token.dep_ == "cc"]
        if not cc_positions:
            phrase = finalize(build_words(tokens))
            return [phrase] if phrase else []
        cc_index = cc_positions[0]
        conj_token = next((token for token in tokens[cc_index + 1:] if token.dep_ == "conj"), None)
        if conj_token is None:
            phrase = finalize(build_words(tokens))
            return [phrase] if phrase else []
        conj_index = tokens.index(conj_token)
        shared_tail = tokens[conj_index + 1:]
        candidates = (
            build_words(tokens[:cc_index]) + build_words(shared_tail),
            build_words([conj_token], is_conj_partner=True) + build_words(shared_tail),
        )
        return [phrase for words in candidates if (phrase := finalize(words))]

    for chunk in doc.noun_chunks:
        phrases.update(chunk_phrases(chunk))

    ordered = sorted(phrases, key=len)
    filtered = [
        phrase for i, phrase in enumerate(ordered)
        if not any(_is_subphrase(phrase, longer) and phrase != longer for longer in ordered[i + 1:])
    ]
    standalone = set()
    for token in doc:
        if (
            token.is_stop or token.is_punct or token.like_num or token.like_email or token.like_url
            or stoplisted(token.text) or token.pos_ not in {"NOUN", "PROPN"}
            or token.text.lower() in self_references or token.ent_type_ == "PERSON" or token.dep_ == "poss"
        ):
            continue
        noun = _clean_token(token._.lemma())
        if noun and not any(_is_subphrase(noun, phrase) for phrase in filtered):
            standalone.add(noun)
    return sorted(set(filtered).union(standalone))


def expand_documents(documents, unit, *, min_sentence_words=1, spacy_model="en_core_web_lg"):
    records = []
    for document in documents:
        if unit == "term":
            pieces = [document.text]
        elif unit == "sentence":
            pieces = split_sentences(document.text, min_words=min_sentence_words)
        elif unit == "paragraph":
            pieces = split_paragraphs(document.text)
        elif unit == "noun_chunk":
            pieces = extract_noun_chunks(document.text, filer_name=document.filer_name, spacy_model=spacy_model)
        else:
            raise ValueError(f"Unsupported unit: {unit!r}.")
        for unit_index, piece in enumerate(pieces):
            records.append(UnitRecord(
                source_index=document.source_index,
                source_id=document.source_id,
                unit_index=unit_index,
                unit_type=unit,
                source_text=document.text,
                processed_text=piece,
                industry=document.industry,
                year=document.year,
            ))
    return records


def _get_nlp(model):
    if model in _NLP_CACHE:
        return _NLP_CACHE[model]
    try:
        import spacy
        import lemminflect  # noqa: F401 -- registers token._.lemma()
        from spacy.tokenizer import Tokenizer
        from spacy.util import compile_infix_regex
    except ImportError as exc:
        raise ImportError(
            "Noun-chunk mode requires optional dependencies. Install InnoBERT with its "
            "`noun-chunks` extra, then install the requested spaCy model."
        ) from exc
    try:
        nlp = spacy.load(model)
    except OSError as exc:
        raise OSError(
            f"spaCy model {model!r} is not installed. Run `python -m spacy download {model}`."
        ) from exc
    nlp.max_length = 3_000_000
    infixes = [pattern for pattern in nlp.Defaults.infixes if "-" not in pattern]
    infix_regex = compile_infix_regex(infixes)
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_regex.finditer,
        token_match=nlp.tokenizer.token_match,
        rules=nlp.Defaults.tokenizer_exceptions,
    )
    _NLP_CACHE[model] = nlp
    return nlp


def _modifying_adjectives():
    resource = files("innobert.resources").joinpath("common_10k_stoplist.txt")
    with resource.open("r", encoding="utf-8") as handle:
        corpus_stoplist = {line.strip() for line in handle if line.strip() and not line.startswith("#")}
    return RANKING_ADJECTIVES | QUALITY_ADJECTIVES | VAGUE_ADJECTIVES | OTHER_FREQUENT | corpus_stoplist


def _clean_token(value):
    if not isinstance(value, str) or not value.strip():
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9-\s]", "", value.lower()).strip()
    if cleaned.isdigit() or sum(ch.isdigit() for ch in cleaned) > len(cleaned) * 0.5:
        return ""
    if len(value) == 1 or cleaned.startswith("-") or cleaned.endswith("-"):
        return ""
    return cleaned


def _is_subphrase(short, long_phrase):
    short_words, long_words = short.split(), long_phrase.split()
    size = len(short_words)
    return bool(size) and size <= len(long_words) and any(
        long_words[i:i + size] == short_words for i in range(len(long_words) - size + 1)
    )
