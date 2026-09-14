# InnoBERT

InnoBERT classifies innovation-related business text using a FinBERT model fine-tuned for multi-label classification. It accepts individual terms, raw documents, aligned lists, and pandas DataFrames; supports noun-chunk, sentence, and paragraph processing; and runs on CPU, CUDA GPU, or Apple MPS.

An InnoBERT label indicates the category or topic most closely associated with the text submitted to the classifier. When InnoBERT is used independently, its output should be interpreted as category-specific topic salience rather than evidence of implemented innovation. In the accompanying research, the innovation interpretation arises from the complete measurement procedure, which first identifies economy-wide novel terms and then uses InnoBERT to classify their nature. See Ahci and Joos (2026) below for the complete measure construction and interpretation.

The seven innovation subcategories are product, process, organizational, marketing, business model, sustainability, and AI. An additional uncategorized category filters out terms irrelevant to innovation and is retained as a fallback. The innovation subcategories are also organized into four main categories:

| Granular category | Main category |
| --- | --- |
| product | product |
| process, organizational, marketing, business model | business process |
| sustainability | sustainability |
| AI | AI |

The innovation-type framework is informed by the Oslo Manual 2018. InnoBERT extends the reporting taxonomy with business model, sustainability, and AI.

## Installation

Python 3.10–3.12 is supported. Python 3.11 is recommended.

### Conda and Jupyter

Install [Anaconda or Miniconda](https://www.anaconda.com/docs/getting-started/main), open Anaconda Prompt on Windows or a terminal on macOS/Linux, and run:

```bash
conda create -n innobert python=3.11 -y
conda activate innobert
python -m pip install --upgrade pip
python -m pip install "innobert[noun-chunks,notebook] @ git+https://github.com/mustafahci/InnoBERT.git@main"
python -m spacy download en_core_web_lg
python -m ipykernel install --user --name innobert --display-name "Python (InnoBERT)"
jupyter lab
```

Select **Python (InnoBERT)** as the notebook kernel. Always use `python -m pip` after activating the environment so packages are installed into the selected interpreter.

### Existing Python or IPython environment

```bash
python -m pip install "innobert[noun-chunks,notebook] @ git+https://github.com/mustafahci/InnoBERT.git@main"
python -m spacy download en_core_web_lg
```

Noun-chunk extraction uses `en_core_web_lg` by default to preserve the preprocessing used in development. Sentence, paragraph, and already-extracted term processing do not require spaCy.

### Google Colab

Run these cells at the beginning of a Colab notebook:

```python
%pip install "innobert[noun-chunks] @ git+https://github.com/mustafahci/InnoBERT.git@main"
!python -m spacy download en_core_web_lg
```

Then restart the runtime if Colab requests it. To use a GPU, select a GPU runtime and keep `device="auto"`.

## Authenticate with Hugging Face

The model downloads from `mustafahci/InnoBERT`. When authentication is required, use the secure login prompt:

```python
from huggingface_hub import notebook_login

notebook_login()
```

For terminal scripts, use:

```bash
hf auth login
```

## Load the model

```python
from innobert import InnoBERT

classifier = InnoBERT.from_pretrained(
    "mustafahci/InnoBERT",
    device="auto",
    spacy_model="en_core_web_lg",
)
```

`device="auto"` selects CUDA, then Apple MPS, then CPU. Users can explicitly request `cpu`, `cuda`, `cuda:0`, or `mps`.

## Classify supplied terms

```python
results = classifier.predict(
    [
        "cloud-based analytics platform",
        "automated inventory replenishment system",
        "employee collaboration network",
    ],
    industry=["Software", "Retail", "Manufacturing"],
    year=[2024, 2023, 2022],
    unit="term",
)

results
```

A scalar industry or year is broadcast to every input. Aligned lists must have exactly one value per text.

## Process raw text

```python
text = """Example Corporation expanded its cloud-based subscription platform for
small-business customers and introduced personalized product recommendations.
It deployed an artificial-intelligence forecasting tool, real-time analytics,
and an automated inventory replenishment system.

The company created a cross-functional product team and an employee collaboration
network. A data-driven order-routing process improved warehouse allocation,
distribution operations, purchasing, and delivery scheduling.

It also adopted an energy-efficient manufacturing process and recyclable packaging
materials to reduce operating emissions and waste."""
```

### Noun chunks

```python
noun_results = classifier.predict(
    text,
    industry="Business Services",
    year=2024,
    filer_name="Example Corporation",
    unit="noun_chunk",
)
```

Noun-chunk mode extracts short candidate terms first and then classifies them using an optional industry–year prompt. Because noun-chunk extraction removes the surrounding sentence, these classifications do not capture negation, timing, attribution, or whether the described activity belongs to the focal firm. Researchers seeking to measure adoption or realized activity should apply additional contextual criteria appropriate to their research design.

### Sentences

```python
sentence_results = classifier.predict(text, unit="sentence")
```

### Paragraphs

```python
paragraph_results = classifier.predict(text, unit="paragraph")
```

Sentence and paragraph processing use raw text. These modes are documented applications beyond the contextual term input used during training.

## DataFrame input and identifiers

```python
import pandas as pd

documents = pd.DataFrame({
    "gvkey": ["001234", "001234"],
    "fyear": [2023, 2024],
    "Industry": ["Business Services", "Business Services"],
    "long_passage": [text, text],
})

results = classifier.predict(
    documents,
    unit="noun_chunk",
    text_col="long_passage",
    industry_col="Industry",
    year_col="fyear",
    source_id_cols=["gvkey", "fyear"],
    metadata_cols=["gvkey", "fyear"],
)
```

`source_id_cols` constructs a unique source identifier such as `001234_2023`. Requested `metadata_cols` are copied to every extracted unit. Duplicate or missing source identifiers raise an error because they would make `unit_id` ambiguous.

Missing spreadsheet values in required text, industry, year, or source-identifier fields raise an error; they are not converted into model inputs such as `"nan"`.

The package returns one row per extracted or supplied unit. It does not aggregate predictions to, for example, a firm-year measure; users retain control over any subsequent counting, weighting, dummy creation, or aggregation.

For very large collections of complete filings, submit manageable DataFrame chunks and save each result before continuing. This prevents all extracted units from thousands of filings being held in memory simultaneously.

## Output

The default compact output contains:

- `unit_id`
- `processed_text`
- `predicted_subcat_labels`: every granular category passing its threshold
- `predicted_subcat_probs`: probabilities in the same order as the subcategory labels
- `main_categories`: all corresponding main categories, deduplicated
- `top_subcat_label`: the assigned subcategory with the highest raw probability
- `top_subcat_prob`: that subcategory's raw probability

Request the complete auditable output to inspect model probabilities and processing details:

```python
full_results = classifier.predict(text, unit="sentence", output="full")
```

Full output adds source lineage, industry and year, all eight `prob_*` columns, token and window counts, long-input actions, the uncategorized rule, a human-readable assignment reason, and the device used.

Compact-output probabilities are rounded to three decimals for readability. Threshold decisions are made using the original full-precision values. `output="full"` preserves full precision in `predicted_subcat_probs`, `top_subcat_prob`, and all eight `prob_*` columns.

InnoBERT is a multi-label classifier. Its eight sigmoid probabilities are estimated independently and do not sum to one. `top_subcat_label` is provided only as a convenience: it is the maximum raw probability among the assigned subcategories and should not be interpreted as a conceptually dominant category.

## Thresholds

The default thresholds are:

```python
{
    "product": 0.65,
    "process": 0.45,
    "organizational": 0.55,
    "marketing": 0.55,
    "business_model": 0.45,
    "sustainability": 0.50,
    "AI": 0.50,
    "uncategorized": 0.25,
}
```

Override only the categories needed for a particular application:

```python
adjusted = classifier.predict(
    text,
    unit="sentence",
    thresholds={"product": 0.70, "AI": 0.40},
    output="full",
)
```

Users should inspect the full probability output and validate alternative thresholds for their own corpus and research setting.

For `term` and `noun_chunk`, the default `gatekeeper` rule first examines the uncategorized probability. `Uncategorized` operates as a rejection gate rather than as an ordinary competing category. If its score reaches the default threshold of 0.25, the observation is assigned to uncategorized, even when another category has a higher score. This conservative rule was selected to prioritize the detection of terms that may not represent a substantive innovation category and thereby reduce false-positive category assignments. Its cost is lower recall: some plausible category associations may be suppressed. Users can inspect all category scores and modify the thresholds or decision rule for applications with different error costs. 

Industry and year are model inputs—not merely descriptive metadata—for contextual `term` and `noun_chunk` classification. The same phrase can therefore receive different scores across industries or years. Choose a documented industry classification, apply it consistently across the research sample, and do not select the industry after observing model results. Use `context_mode="none"` only as an explicitly documented alternative application.

## Long documents

A long source document is not passed to BERT as one input when `unit="noun_chunk"`, `"sentence"`, or `"paragraph"`:

- noun-chunk mode parses the source, extracts short terms, and classifies the terms in batches;
- sentence mode classifies each detected sentence;
- paragraph mode classifies each blank-line-delimited paragraph and divides long paragraphs into overlapping token windows.

Do not pass an entire filing with `unit="term"`. Long term, noun-chunk, and sentence units raise an error by default rather than being silently truncated. Users must explicitly select `long_text_strategy="truncate"` or `"window"` when that behavior is intended.

Beginning with version 0.2.2, paragraph windows are constructed by explicitly slicing the complete content-token sequence. Every content token is covered by at least one window, including the end of the paragraph, and internal checks fail rather than returning incomplete coverage. The default `max_length=160` includes model special tokens, and the default `stride=32` repeats 32 content tokens between adjacent windows.

Paragraph windows use category-wise maximum probabilities. These values indicate whether relevant evidence appears anywhere in the paragraph; they are not calibrated probabilities for the paragraph as a whole. If a long source has no blank-line paragraph boundaries, the package warns that it may be treated as one giant paragraph. Explicitly selecting `long_text_strategy="truncate"` omits content after `max_length`; use it only when that loss is intended.

For 10-K research, extracting the intended filing section before classification is recommended. Processing an entire filing introduces risk factors, MD&A, notes, controls, and other content that may change the measured construct. If complete filings are required, retain section identifiers and process each section as a separate source.

Use `output="full"` to inspect `token_count`, `window_count`, `long_text_action`, and all category probabilities.

## Progress and run diagnostics

`progress="auto"` is the default. It displays adaptive progress bars in interactive notebooks and terminals and remains silent in non-interactive jobs.

```python
results = classifier.predict(text, unit="noun_chunk", industry="Services", year=2024)
classifier.last_run_summary
```

Use `progress=True` to force progress or `progress=False` for clean logs and saved notebook outputs. Progress is written separately from returned results.

## Saving output

```python
results.to_csv("innobert_results.csv", index=False)
results.to_parquet("innobert_results.parquet", index=False)
```

CSV stores list-valued columns as text. Parquet preserves list-like values more naturally.

## Scripts and non-notebook use

The same interface works in a `.py` script:

```python
from innobert import InnoBERT

classifier = InnoBERT.from_pretrained("mustafahci/InnoBERT", device="auto")
results = classifier.predict(
    ["automated production system"],
    industry="Manufacturing",
    year=2024,
    unit="term",
    progress=False,
)
results.to_csv("innobert_results.csv", index=False)
```

Run it from the activated environment with `python your_script.py`.

## Common errors

- **Wrong environment:** compare `python -c "import sys; print(sys.executable)"` with the notebook kernel.
- **pip resolves to a user folder:** use `python -m pip`; in Conda, set `conda env config vars set PYTHONNOUSERSITE=1` and reactivate the environment if needed.
- **Hugging Face authorization fails:** authenticate with `notebook_login()` or `hf auth login`.
- **CUDA unavailable:** use `device="auto"` or `device="cpu"`, or install a compatible CUDA-enabled PyTorch build.
- **spaCy model missing:** run `python -m spacy download en_core_web_lg` in the active environment.
- **List length mismatch:** provide one industry/year per source text or use a scalar.
- **Input too long:** choose the appropriate unit; only explicitly request truncation or token windows when scientifically justified.

See [`examples/InnoBERT_manual.ipynb`](examples/InnoBERT_manual.ipynb) for a guided walkthrough.

## Scientific scope

The trained model uses:

```text
Industry: {industry}. Year: {year}. The term is: <TERM> {term} </TERM>.
```

Contextual term and noun-chunk classification are therefore the validated/default use. The package performs inference only. It does not reconstruct economy-wide novelty, compare prior years, build NEW or STOCK measures, redistribute training data, or automatically create firm-year measures.

## Citation

Ahci, Mustafa and Joos, Philip, **Beyond Invention: The Composition and Economic Relevance of Innovation-Related Capabilities** (Updated September 1, 2026). Available at SSRN: [https://ssrn.com/abstract=4797745](https://ssrn.com/abstract=4797745) or [http://dx.doi.org/10.2139/ssrn.4797745](http://dx.doi.org/10.2139/ssrn.4797745).

Please also cite the upstream FinBERT paper:

Huang, A. H., Wang, H., and Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. *Contemporary Accounting Research*, 40(2), 806–841.

Taxonomy reference:

OECD/Eurostat (2018), *Oslo Manual 2018: Guidelines for Collecting, Reporting and Using Data on Innovation*, 4th Edition, The Measurement of Scientific, Technological and Innovation Activities, OECD Publishing, Paris/Eurostat, Luxembourg, [https://doi.org/10.1787/9789264304604-en](https://doi.org/10.1787/9789264304604-en).

## License

Copyright 2026 Mustafa Ahci. The package code and fine-tuned InnoBERT weights are provided under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The model was independently fine-tuned from `yiyanghkust/finbert-pretrain`; upstream provenance and licensing qualifications are documented in [`MODEL_CARD.md`](MODEL_CARD.md).
