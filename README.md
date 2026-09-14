# InnoBERT

InnoBERT helps researchers classify business language into seven economically meaningful categories: **product, process, organizational, marketing, business model, sustainability, and artificial intelligence (AI)**. It can process individual terms, raw documents, aligned lists, and pandas DataFrames on a CPU, CUDA-enabled GPU, or Apple MPS device.

Importantly, InnoBERT classifies the semantic content of submitted text, but it does not determine whether that language is novel. For example, it may classify “machine learning” as AI, but it does not establish whether the term is new for the firm, relative to other firms, or at that point in time. An InnoBERT classification should therefore not, by itself, be interpreted as evidence of innovation.

The measurement approach developed by Ahci and Joos (2026) in _Beyond Invention: The Composition and Development of Innovation-Related Capabilities_ separates these tasks. It first identifies novel terms by comparing firm disclosures with prior economy-wide business disclosures and then uses InnoBERT to classify those terms. When used independently, InnoBERT’s output should be interpreted as a semantic category assignment. See Ahci and Joos (2026) below for the complete measure construction and interpretation.

## Category framework

The seven substantive subcategories are product, process, organizational, marketing, business model, sustainability, and AI. The classification framework draws on both the 2005 and 2018 editions of the Oslo Manual. The 2005 edition distinguishes product, process, organizational, and marketing innovation. The 2018 edition reorganizes these concepts into two broader types, namely product innovation and business-process innovation, with process, organizational and marketing functions included within the broader business-process category. InnoBERT preserves the more granular distinctions from the 2005 framework, while additionally identifying business model, sustainability, and AI as separate categories relevant to contemporary business disclosures.

The seven substantive subcategories are therefore organized into four broader categories in accordance with updated Oslo Manual (2018):
| Granular category | Broad category |
| --- | --- |
| product | product |
| process, organizational, marketing, business model | business process |
| sustainability | sustainability |
| AI | AI |

An eighth category, `uncategorized`, is used when the submitted text does not receive sufficient support for any substantive innovation category under the applicable decision rule. It commonly captures routine financial, corporate, or reporting terminology, such as “cash flow statement” or “financing activities”. An `uncategorized` assignment does not necessarily mean that the text is economically irrelevant; it means that no substantive innovation-related category was assigned. See the `Thresholds` section below for more information about decision rules.

## Installation

Python 3.10–3.12 is supported. Python 3.11 is recommended.

### Conda and Jupyter

Install [Anaconda or Miniconda](https://www.anaconda.com/docs/getting-started/main), open Anaconda Prompt on Windows or a terminal on macOS/Linux, and run:

```bash
conda create -n innobert python=3.11 -y
conda activate innobert
python -m pip install --upgrade pip
python -m pip install "innobert[noun-chunks,notebook] @ git+https://github.com/mustafahci/InnoBERT.git@v0.2.4"
python -m spacy download en_core_web_lg
python -m ipykernel install --user --name innobert --display-name "Python (InnoBERT)"
jupyter lab
```

Select **Python (InnoBERT)** as the notebook kernel. Always use `python -m pip` after activating the environment so packages are installed into the selected interpreter.

### Existing Python or IPython environment

```bash
python -m pip install "innobert[noun-chunks,notebook] @ git+https://github.com/mustafahci/InnoBERT.git@v0.2.4"
python -m spacy download en_core_web_lg
```

Noun-chunk extraction uses `en_core_web_lg` by default to preserve the preprocessing used in development. Sentence, paragraph, and already-extracted term processing do not require spaCy.

## Hugging Face access

The public model downloads from `mustafahci/InnoBERT` without a Hugging Face account or access token. Authentication is necessary only if access to the model repository is restricted in the future. We also provide a simple interface app using Hugging Face Spaces for non-technical users or simple tasks. 

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


Noun-chunk mode produces candidate phrases rather than a final set of novel innovation terms. Identical processed phrases are deduplicated within each source document, so repetition within one document does not increase the number of classified units. The same phrase is retained separately when it appears in different source documents or firm-years, and each row remains linked to its source through `source_id` and `unit_id`. The `unit_id` is an output identifier, not a character position or sentence location; InnoBERT does not report within-document mention counts or text offsets.

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

InnoBERT produces multiple output rows from each input document (e.g., document-noun chunks). Researchers should therefore provide a unique document identifier so that every extracted term, sentence, or paragraph can be linked back to its source.

- Use `source_id_col="document_id"` when the DataFrame already contains a unique identifier. <br>
- Use `source_id_cols=["gvkey", "fyear"]` to construct an identifier from multiple columns. <br>
- Use `metadata_cols=["gvkey", "fyear"]` to copy identifying variables into every output row. <br>

If neither option is supplied, InnoBERT uses the DataFrame index. This may be less transparent and can change after filtering or resetting the index.

Source identifiers must be unique across input rows. If a firm-year contains multiple documents or passages, include an additional identifier such as an accession number or passage number.

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

When a document contains multiple paragraphs, InnoBERT processes each paragraph separately. Any paragraph that exceeds the model’s length limit is divided into the overlapping sections. Each section contains up to 158 text tokens, plus two tokens required by the model. Adjacent sections repeat 32 tokens so that language near a section boundary is not interpreted without its surrounding context. Processing continues until the end of the paragraph. InnoBERT checks that every token has been included and raises an error instead of silently omitting text if complete coverage cannot be confirmed.

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
