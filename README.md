# InnoBERT

InnoBERT classifies business text into eight innovation categories using a FinBERT model fine-tuned for multi-label innovation classification. It accepts individual texts, aligned batches, or pandas DataFrames and runs on CPU, CUDA, or Apple MPS.

> Release status: private alpha. The code and fine-tuned model are licensed under Apache License 2.0. Model weights are intentionally not stored in this Git repository.

## Installation

Local development install:

```bash
python -m pip install -e .
```

Include noun-chunk processing and install the pinned spaCy language model separately:

```bash
python -m pip install -e ".[noun-chunks,notebook]"
python -m spacy download en_core_web_lg
```

After the repository owner and first release tag are configured:

```bash
python -m pip install "innobert[noun-chunks] @ git+https://github.com/mustafahci/InnoBERT.git@v0.1.0"
```

The model weights should be hosted in a separate Hugging Face model repository. They may also be loaded from an extracted local model directory.

## Load the model

```python
from innobert import InnoBERT

classifier = InnoBERT.from_pretrained(
    "mustafahci/InnoBERT",
    device="auto",  # auto, cpu, cuda, cuda:0, or mps
    spacy_model="en_core_web_lg",  # used only by noun_chunk mode
)
```

While the model repository is private, first authenticate locally with Hugging Face or pass an access token through a secure environment variable:

```bash
hf auth login
```

```python
import os

classifier = InnoBERT.from_pretrained(
    "mustafahci/InnoBERT",
    token=os.environ["HF_TOKEN"],
    revision="83bc233b62f981503c0dbb40324b11f428d5a8c7",
    device="auto",
)
```

Do not place access tokens directly in notebooks or commit them to Git. The `revision` argument is optional but gives an immutable model version.

For the supplied local archive, extract it first and pass the directory containing `config.json`, `model.safetensors`, and the tokenizer files. Passing the ZIP itself raises an error deliberately.

## Classify aligned terms

`term` is for already-extracted terms. Each term can have a different industry and year:

```python
results = classifier.predict(
    ["cloud-based document platform", "automated production system"],
    industry=["Software", "Manufacturing"],
    year=[2024, 2023],
    unit="term",
)
```

A scalar context is broadcast:

```python
results = classifier.predict(
    ["mobile payment service", "recommendation engine"],
    industry="Retail",
    year=2024,
    unit="term",
)
```

If parallel lists differ in length, the package stops with an error showing the expected and received counts. It never silently recycles, drops, or reorders context values.

## Process documents by noun chunk, sentence, or paragraph

```python
text = """We introduced a cloud-based analytics platform. It automates inventory planning.

The redesigned subscription model supports smaller customers."""

noun_chunks = classifier.predict(
    text,
    industry="Software",
    year=2024,
    filer_name="Example Corporation",
    unit="noun_chunk",
)

sentences = classifier.predict(text, unit="sentence")
paragraphs = classifier.predict(text, unit="paragraph")
```

The modes preserve the two notebook pipelines:

| Unit | Segmentation | Default model input | Default uncategorized rule | Long-text default |
| --- | --- | --- | --- | --- |
| `term` | none; one input is one term | exact industry–year training prompt | gatekeeper | truncate at 64 tokens |
| `noun_chunk` | final 2026 spaCy extraction | exact industry–year training prompt | gatekeeper | truncate at 64 tokens |
| `sentence` | conference-call regex splitter | raw sentence | fallback | truncate at 160 tokens |
| `paragraph` | blank-line boundaries | raw paragraph | fallback | overlapping 160-token windows |

For sentence or paragraph experiments that feed industry and year to the model, set `context_mode="industry_year"`. This wraps the full unit in the original term-training template and is therefore an extension, not the validated training use. Conversely, `context_mode="none"` removes context from term or noun-chunk inputs.

## DataFrame input

```python
import pandas as pd

documents = pd.DataFrame({
    "document_id": ["A", "B"],
    "business_text": ["We launched a digital service.", "We automated quality control."],
    "industry_name": ["Software", "Manufacturing"],
    "fyear": [2024, 2023],
})

results = classifier.predict(
    documents,
    unit="sentence",
    text_col="business_text",
    industry_col="industry_name",
    year_col="fyear",
    source_id_col="document_id",
)
```

For `sentence`, industry and year remain in the output but are not fed to the model unless `context_mode="industry_year"` is requested.

## Thresholds and decision rules

Published defaults, in model-output order:

```python
{
    "inno_product": 0.65,
    "inno_process": 0.45,
    "inno_organizational": 0.55,
    "inno_marketing": 0.55,
    "inno_businessmodel": 0.45,
    "inno_sustainability": 0.50,
    "inno_AI": 0.50,
    "inno_uncategorized": 0.25,
}
```

Override only what is needed; canonical names and short aliases are accepted:

```python
results = classifier.predict(
    text,
    unit="sentence",
    thresholds={"product": 0.70, "AI": 0.40},
)
```

- `gatekeeper`: if uncategorized crosses its threshold, return only uncategorized; otherwise retain every innovation label crossing its own threshold.
- `fallback`: ignore the uncategorized probability for label assignment and return uncategorized only when no innovation category crosses.

Set `uncategorized_rule="gatekeeper"` or `"fallback"` to override the mode default.

## Output

By default, `predict()` returns a compact DataFrame with five columns:

- `unit_id`, which preserves the source and within-source unit;
- `processed_text`;
- `predicted_labels`;
- `dominant_label`;
- `dominant_probability`.

`dominant_label` is the highest-probability label among `predicted_labels`, so it never contradicts the gatekeeper or fallback assignment.

Request the complete auditable output when you need source lineage, context, diagnostics, or threshold analysis:

```python
full_results = classifier.predict(
    text,
    unit="sentence",
    output="full",
)
```

The full DataFrame adds `source_index`, `source_id`, `unit_index`, `unit_type`, `source_text`, `industry`, `year`, `token_count`, `window_count`, `truncated_or_windowed`, `device_used`, and all eight `prob_*` columns. Use these probability columns to inspect classifications and select thresholds appropriate for the application.

Set `include_model_input=True` to include the exact string sent to the tokenizer; it is retained in either compact or full output. Results can be saved normally:

```python
results.to_csv("innobert_results.csv", index=False)
results.to_parquet("innobert_results.parquet", index=False)
```

## Long inputs

`long_text_strategy` can be:

- `"truncate"`: match the notebook behavior and record whether truncation occurred;
- `"window"`: use overlapping token windows and take the maximum probability for each category across windows;
- `"error"`: stop and report the affected rows.

The category-wise maximum makes paragraph output an “innovation evidence anywhere in the paragraph” score. It is not a calibrated probability for the paragraph as a whole.

Use `output="full"` to inspect `window_count`, `truncated_or_windowed`, and the window-aggregated category probabilities.

## Common errors

- **Industry/year missing:** required for the default `term` and `noun_chunk` profile. Supply both, or explicitly use `context_mode="none"`.
- **List length mismatch:** provide one industry/year per source text, or use a scalar to broadcast.
- **CUDA unavailable:** use `device="auto"` or `device="cpu"`, or install a CUDA-compatible PyTorch build.
- **spaCy model missing:** install the noun-chunk extra, then run `python -m spacy download en_core_web_lg`.
- **No units produced:** reduce `min_sentence_words` or inspect whether noun-chunk filters remove all candidates.
- **Input too long:** select `long_text_strategy="window"` or increase `max_length` up to 512.

See [`examples/InnoBERT_manual.ipynb`](examples/InnoBERT_manual.ipynb) for a runnable walkthrough.

## Scientific scope

The trained model uses the exact input template:

```text
Industry: {industry}. Year: {year}. The term is: <TERM> {term} </TERM>.
```

Accordingly, contextual term and noun-chunk classification are the validated/default use. Sentence and paragraph modes reproduce or extend the conference-call application but should not be described as independently validated without further evaluation. The package performs inference only: it does not reconstruct historical novelty, compare prior years, build NEW or STOCK measures, or redistribute training data.

## Licensing and citation

Copyright 2026 Mustafa Ahci. The package code and fine-tuned InnoBERT weights are released under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The base checkpoint is `yiyanghkust/finbert-pretrain` from the Apache-2.0-licensed FinBERT project. Its Hugging Face repository does not independently declare license metadata, so the upstream provenance and this qualification are retained in the model card and notice.

Please cite the software metadata in [`CITATION.cff`](CITATION.cff), together with the associated paper once its final citation is available, and cite the upstream FinBERT paper:

> Huang, A. H., Wang, H., and Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. *Contemporary Accounting Research*, 40(2), 806–841.
