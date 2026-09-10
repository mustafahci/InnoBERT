---
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: yiyanghkust/finbert-pretrain
license: apache-2.0
author: mustafahci
tags:
  - finance
  - accounting
  - innovation
  - multi-label-classification
---

# InnoBERT model card

## Model description

InnoBERT is a BERT-base sequence classifier fine-tuned from `yiyanghkust/finbert-pretrain`. It has eight sigmoid outputs in this fixed order:

1. `inno_product`
2. `inno_process`
3. `inno_organizational`
4. `inno_marketing`
5. `inno_businessmodel`
6. `inno_sustainability`
7. `inno_AI`
8. `inno_uncategorized`

The supplied checkpoint is a `BertForSequenceClassification` model with 12 layers, hidden size 768, 12 attention heads, 30,873 vocabulary items, and an 8-by-768 classifier head.

The Python package reports reader-facing granular labels without the internal `inno_` prefix and maps process, organizational, marketing, and business-model predictions to the main category `business_process`. This reporting hierarchy does not alter the model outputs or thresholds.

The innovation-type framework is informed by the Oslo Manual 2018. Sustainability, AI, and the uncategorized outcome are additional reporting categories used by InnoBERT rather than being presented as separate official Oslo Manual innovation types.

## Intended use

The primary use is multi-label classification of innovation-related terms extracted from business disclosures. The validated training input includes an industry, year, and marked term. The package also supports sentence and paragraph processing as documented extensions.

It is not intended for individual-level decisions, high-stakes automated decisions, or claims about realized innovation without separate validation.

## Preprocessing and thresholds

The noun-chunk mode preserves internal hyphens, removes specified generic modifiers, lemmatizes noun heads, splits selected coordinated noun phrases, limits phrases to four terms, removes subphrases, and optionally removes filer-name tokens. Its corpus-derived stoplist is frozen in the Python package.

Default thresholds are 0.65, 0.45, 0.55, 0.55, 0.45, 0.50, 0.50, and 0.25 in the label order above.

## Limitations

- The training data are not distributed with this release.
- Sentence and paragraph inputs differ from the contextual term format used for training.
- Windowed paragraph scores use category-wise maxima and are not calibrated paragraph-level probabilities.
- Results can vary if noun chunks are extracted with a different spaCy model or version.
- Disclosure language is not equivalent to realized innovation capability or outcomes.
- The training data and annotations are not distributed, so users cannot reconstruct training from this release.
- Applying the classifier outside the disclosure setting, industries, or years represented in development requires separate validation.

## Citation

Ahci, Mustafa and Joos, Philip, **Beyond Invention: The Composition and Economic Relevance of Innovation-Related Capabilities** (Updated September 1, 2026). Available at SSRN: https://ssrn.com/abstract=4797745 or http://dx.doi.org/10.2139/ssrn.4797745.

OECD/Eurostat (2018), *Oslo Manual 2018: Guidelines for Collecting, Reporting and Using Data on Innovation*, 4th Edition, The Measurement of Scientific, Technological and Innovation Activities, OECD Publishing, Paris/Eurostat, Luxembourg. https://doi.org/10.1787/9789264304604-en.

## Upstream attribution

InnoBERT was independently fine-tuned from Huang, Wang, and Yang's FinBERT checkpoint and is not affiliated with or endorsed by the original FinBERT or BERT developers.

## License

The InnoBERT fine-tuned weights and accompanying code are provided under Apache License 2.0. The official FinBERT GitHub project is Apache-2.0 licensed and directly identifies the pretrained FinBERT models; however, `yiyanghkust/finbert-pretrain` does not independently declare license metadata on its Hugging Face model card. This provenance qualification is disclosed rather than obscured. Users requiring formal legal certainty—especially for commercial redistribution—should independently confirm that the upstream license covers the checkpoint weights.
