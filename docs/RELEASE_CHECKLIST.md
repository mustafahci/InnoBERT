# Release checklist

## Confirm before publishing

- [x] Replace repository and model-host placeholders with `mustafahci/InnoBERT`.
- [x] Set the author to Mustafa Ahci and initial repository visibility to private.
- [x] Choose Apache License 2.0 for code and fine-tuned weights.
- [ ] Obtain or document confirmation that the upstream Apache-2.0 license covers the distributed FinBERT checkpoint weights.
- [x] Choose and document Apache License 2.0 for the fine-tuned weights.
- [x] Add `LICENSE`, `NOTICE`, and `CITATION.cff`.
- [x] Create the private Hugging Face repository and upload the verified model files.
- [x] Record immutable model commit `83bc233b62f981503c0dbb40324b11f428d5a8c7`.
- [ ] Complete the model card's training and evaluation fields.
- [ ] Upload only the minimal model files to the model host; do not upload training data or notebooks with private outputs.
- [ ] Set human-readable label metadata in the hosted `config.json`.
- [ ] Pin or record the exact spaCy model version used for the release benchmark.

## Verified source archive

- ZIP SHA-256: `2a0a59b7eb8dcc9c4421926cfab7d2ed26c89406ed3210673db68874dcb619c8`
- `model.safetensors` SHA-256: `9227ecde489ea1302e7f110d9fef33d05aae92b9e6ddfb57ca3232ef1aa143ef`
- Weight file size: 439,055,376 bytes
- Architecture: `BertForSequenceClassification`
- Classifier tensors: weight shape `[8, 768]`, bias shape `[8]`
- Tensor count: 201

## Pre-release verification

1. Run unit tests and a syntax check.
2. Build the source distribution and wheel; inspect their contents for weights, ZIPs, paths, outputs, and secrets.
3. Install the wheel in a clean environment outside the repository and import `innobert`.
4. Run the example terms on CPU against the exact hosted checkpoint.
5. Run the same example terms on CUDA and compare probabilities within a documented tolerance.
6. Run a frozen noun-chunk benchmark with the selected spaCy model.
7. Initialize Git only after the artifact audit is clean.
8. After explicit authorization, push and create an immutable `v0.1.0` tag.
9. Verify installation from that tag in a clean environment.

## Files intentionally excluded

- training data and annotations;
- prior-year novelty comparison and STOCK construction;
- checkpoints, optimizer state, logs, and cached datasets;
- original research notebooks and their machine-specific paths;
- model weights from the Git repository (host separately as model artifacts).
