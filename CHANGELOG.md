# Changelog

## 0.2.2 — 2026-09-14

- Fixed a correctness bug that could omit overflow content from long paragraph inputs under some tokenizer and Transformers combinations.
- Replaced dependency-specific overflow handling with explicit overlapping token windows and internal coverage checks.
- Added regression tests verifying complete coverage through the final token of a long input.
- Constructed BERT windows explicitly from tokenizer-provided `[CLS]` and `[SEP]`
  identifiers for compatibility with both Transformers 4 and 5.
- Reject spreadsheet-style missing text and context values instead of converting them to model input strings.
- Added auditable `uncategorized_rule` and `assignment_reason` fields to full output.
- Expanded the documentation on noun-chunk interpretation, industry context, the uncategorized gatekeeper, long inputs, and corpus-specific validation.
- Made the notebook installation cell executable when required and removed stale illustrative output tables.

Paragraph-window results created with versions through 0.2.1 should be rerun.
