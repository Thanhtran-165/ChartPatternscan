# Scanner V2 Book-to-Scanner Contract

This document locks the rebuild path for the source-book-to-scanner layer.

Scanner V2 is not a rewrite of the whole project. It replaces the unsafe core
chain that turns source-book knowledge into scanner logic.

## Scope

The first V2 cohort is deliberately small:

- Broadening Bottoms
- Double Bottoms
- Double Tops
- Head-and-Shoulders Bottoms
- Head-and-Shoulders Tops
- Triangles, Ascending
- Cup with Handle

The legacy scanner can remain as a benchmark and exploration tool. It is not
the research-grade source of truth for new Vietnam-market conclusions.

Legacy active scanner entrypoints are now quarantined. See
[`legacy-scanner-burn-notice.md`](legacy-scanner-burn-notice.md).

## Rule Contract

Every official scanner rule must include:

- `book_chapter`
- `source_page`
- `source_section`
- `evidence_excerpt`
- `interpreted_rule`
- `numeric_threshold`
- `confidence`
- `notes_when_ambiguous`

Rules without provenance are not allowed into the official scanner.

## Taxonomy Lineage

Every signal must be traceable through this chain:

```text
Bulkowski chapter
-> extracted source rule
-> normalized rule spec
-> scanner pattern key
-> result payload
-> book v2 chapter
```

The active seed map lives in `scanner/v2/taxonomy_lineage.json`.

## Activation Gate

A pattern is not official until all gates pass:

1. Provenance validates for every rule.
2. Evidence excerpts align to the claimed PDF pages.
3. Taxonomy lineage is complete.
4. Every `rule_type` maps to a V2 module.
5. Full normalized spec hash is persisted into result metadata.
6. Golden fixtures exist and pass.
7. `official_candidate` is explicitly set to `true`.

`broadening_bottoms` is the first official V2 candidate. The remaining core
patterns intentionally fail the official gate until their golden fixtures are
created.

## Commands

Run the contract audit:

```bash
.venv/bin/python scanner/audit_scanner_v2_contract.py --out artifacts/scanner_v2/contract_audit.json
```

Run the source-alignment audit:

```bash
.venv/bin/python scanner/audit_scanner_v2_source_alignment.py --pattern broadening_bottoms --out artifacts/scanner_v2/source_alignment_broadening_bottoms.json
```

Run the V2 tests:

```bash
.venv/bin/python -m pytest tests/test_scanner_v2_contract.py -q
```

Run all tests:

```bash
.venv/bin/python -m pytest -q
```

## Next Step

Promote one pattern at a time. `broadening_bottoms` is now the reference pattern:

1. Use its provenance fields as the minimum rule standard.
2. Use its fixture set as the minimum pass/fail standard.
3. Only set `official_candidate` after a pattern can compile with `require_official=True`.

The next promotion target should be either `double_bottoms` or `double_tops`.

For the PDF/chapter outcome, follow
[`v2-pdf-monograph-task-list.md`](v2-pdf-monograph-task-list.md). That list is
the source of truth for reaching the expected "run result -> research PDF"
workflow.

## Methodology Gate

The source-to-scanner contract is necessary but not sufficient for the
"Bulkowski for Vietnam" target. Research chapters must also satisfy the
methodology contract in
[`bulkowski-vietnam-methodology-contract.md`](bulkowski-vietnam-methodology-contract.md).
The required statistics layer is specified in
[`bulkowski-vietnam-statistics-contract.md`](bulkowski-vietnam-statistics-contract.md).
The chapter scoring framework is specified in
[`bulkowski-vietnam-chapter-framework.md`](bulkowski-vietnam-chapter-framework.md).
The final release gate is specified in
[`bulkowski-vietnam-release-gate.md`](bulkowski-vietnam-release-gate.md).
The final P1-P5 target standard is specified in
[`bulkowski-vietnam-85-90-standard.md`](bulkowski-vietnam-85-90-standard.md).

In short:

- Scanner V2 decides whether a pattern detection is source-backed.
- The methodology contract decides whether a chapter is strong enough to be used
  as a Vietnam investment reference.
- The statistics contract decides whether the chapter has enough event-path
  measurements, denominators, quantiles, and context splits.
- The chapter framework decides whether hard-gate issues cap the chapter below
  the 85-90 target.
- The release gate decides whether a chapter is held, published with caveats, or
  strong enough to use as the standard template.
- A detector can be official while the chapter remains a research draft if data
  integrity, robustness, confidence intervals, or market-context requirements are
  still incomplete.
