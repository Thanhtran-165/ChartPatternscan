# Scanner V2 PDF Monograph Task List

## Target Outcome

The expected end state is:

```text
source PDF
-> sourced rules
-> Scanner V2 detections on real market data
-> Vietnam statistics
-> DeepSeek V4 Flash commentary
-> one research-style PDF chapter per pattern
```

The first target PDF is:

```text
broadening_bottoms.pdf
```

This PDF should feel like a compact chart-pattern research chapter. It should
not copy Bulkowski's layout or prose, but it should preserve the same research
discipline: definition, evidence, statistics, examples, caveats, and traceability.

The methodology standard for deciding whether the chapter reaches the
"Bulkowski for Vietnam" target lives in
[`bulkowski-vietnam-methodology-contract.md`](bulkowski-vietnam-methodology-contract.md).
The required statistics layer lives in
[`bulkowski-vietnam-statistics-contract.md`](bulkowski-vietnam-statistics-contract.md).
The chapter scoring framework lives in
[`bulkowski-vietnam-chapter-framework.md`](bulkowski-vietnam-chapter-framework.md).
The final release gate lives in
[`bulkowski-vietnam-release-gate.md`](bulkowski-vietnam-release-gate.md).
The final P1-P5 standard lives in
[`bulkowski-vietnam-85-90-standard.md`](bulkowski-vietnam-85-90-standard.md).

## Current Status

The first Broadening Bottoms V2 PDF pipeline has reached the initial target
outcome.

Completed foundation:

- [x] `broadening_bottoms` has source-backed rule provenance.
- [x] `broadening_bottoms` evidence excerpts align to claimed PDF pages.
- [x] `broadening_bottoms` has a Scanner V2 detector.
- [x] `broadening_bottoms` has golden fixtures.
- [x] `broadening_bottoms` is `official_ready=true`.
- [x] DeepSeek default model is `deepseek-v4-flash`.
- [x] Legacy scanner entrypoints are quarantined from the rebuild path.
- [x] Scanner V2 has been run against real Vietnam OHLCV data.
- [x] V2 detections exist.
- [x] Vietnam statistics exist.
- [x] V2 `chapter_payload.json` exists for `broadening_bottoms`.
- [x] V2 `chapter_core.md` exists for `broadening_bottoms`.
- [x] DeepSeek V4 Flash is connected to the V2 monograph payload.
- [x] V2 `chapter_final.md` exists.
- [x] `broadening_bottoms.pdf` has been rendered.

Methodology gaps that remain after the first PDF run:

- [ ] Add methodology status to the payload and rendered PDF.
- [ ] Add P0 statistics status to the payload and rendered PDF.
- [ ] Add chapter framework score and hard-gate status to the payload and PDF.
- [ ] Add release gate status and red-team risk notes to the payload and PDF.
- [ ] Add final classification label: `not-usable`, `research-only`,
  `watchlist-reference`, `investment-reference`, or `tradable-setup`.
- [ ] Export event-level JSON and CSV that can reproduce the PDF tables.
- [ ] Add example-selection rule: median, strong tail, failure, borderline.
- [ ] Add data-integrity disclosure for universe, delisting/transfer coverage,
  corporate actions, status flags, liquidity filters, and benchmark data.
- [ ] Persist event-level OHLC path after breakout, not only aggregate high/low.
- [ ] Add `B_ref` and `B_exec` anchors.
- [ ] Add disjoint market-group panels: `VN30`, `VN100 ex VN30`,
  `Outside VN100`.
- [ ] Add signed close return 20/60 and benchmark excess vs VNINDEX 20/60.
- [ ] Add confidence intervals or bootstrap intervals for key statistics.
- [ ] Add failure ladder 5/10/20/40.
- [ ] Add target-first-before-adverse-5%.
- [ ] Add Race(+5%,-5%), Race(Target,-5%), and RTR diagnostics.
- [ ] Add throwback/pullback and time-to-event reporting.
- [ ] Add concentration metrics: `N ticker`, HHI or top10 share.
- [ ] Add ECDF/forest/KM/heatmap/scatter diagnostics where data supports them.
- [ ] Formalize overlap and nested-pattern policy.
- [ ] Add sensitivity tables before using the template for broad replication.
- [ ] Add multiple-comparison correction before making broad multi-pattern claims.

## Completion Definition

Do not call this task list complete until all of these are true:

- [x] `artifacts/scanner_v2/broadening_bottoms/detections.json` exists.
- [x] `artifacts/scanner_v2/broadening_bottoms/statistics.json` exists.
- [x] `artifacts/scanner_v2/broadening_bottoms/chapter_payload.json` exists.
- [x] `artifacts/scanner_v2/broadening_bottoms/chapter_core.md` exists.
- [x] `artifacts/scanner_v2/broadening_bottoms/chapter_commentary.md` exists.
- [x] `artifacts/scanner_v2/broadening_bottoms/chapter_final.md` exists.
- [x] `artifacts/scanner_v2/broadening_bottoms/broadening_bottoms.pdf` exists.
- [x] Commentary validation confirms DeepSeek did not introduce unsupported numbers.
- [x] Source alignment audit passes.
- [x] Scanner V2 contract audit passes.
- [x] Full test suite passes.

The first-PDF task is complete. The next task is methodology hardening toward
the 85-90% "Bulkowski for Vietnam" bar described in
[`bulkowski-vietnam-methodology-contract.md`](bulkowski-vietnam-methodology-contract.md).

## Phase A - Data Runner

Goal:

Run `broadening_bottoms` V2 against real OHLCV data.

Tasks:

- [ ] Decide the canonical input data source for the first V2 run.
- [ ] Add a V2 runner CLI for `broadening_bottoms`.
- [ ] Normalize OHLCV data before detector input.
- [ ] Convert market data into V2 pivots and close sequences.
- [ ] Persist raw detections with scanner metadata:
  - `scanner_version`
  - `pattern_key`
  - `spec_hash`
  - `source_chapters`
  - matched rules
  - breakout direction
  - breakout date/index
  - symbol
- [ ] Add tests using a small deterministic sample.

Done when:

- [ ] The runner can produce `detections.json`.
- [ ] Detections include enough metadata to trace back to the source PDF and spec hash.

## Phase B - Post-Breakout Evaluation

Goal:

Turn detections into research statistics.

Tasks:

- [ ] Define evaluation window for first V2 monograph.
- [ ] Compute max favorable excursion.
- [ ] Compute max adverse excursion.
- [ ] Compute 5% failure rate or a clearly labeled alternative.
- [ ] Compute target-hit rate using the sourced measure rule, once that rule is added.
- [ ] Compute up/down breakout split.
- [ ] Compute sample counts and confirmed count.
- [ ] Select example detections for chart rendering.
- [ ] Persist `statistics.json`.

Done when:

- [ ] `statistics.json` has no AI-generated facts.
- [ ] Every statistic records its sample size and calculation rule.

## Phase C - Chapter Payload

Goal:

Create the locked fact payload that DeepSeek is allowed to use.

Tasks:

- [ ] Define `schemas/scanner_v2/monograph_payload.schema.json`.
- [ ] Build `chapter_payload.json` from:
  - source provenance
  - source alignment result
  - scanner contract audit
  - detections
  - statistics
  - golden fixture summary
  - Bulkowski benchmark references
  - methodology status:
    - chapter lane
    - data scope
    - data-integrity status
    - metric denominators
    - limitations
    - unresolved methodology gates
  - statistics status:
    - event model completeness
    - anchor mode: `B_ref` and/or `B_exec`
    - required table coverage
    - quantile coverage
    - CI/bootstrap coverage
    - censoring policy
    - retest epsilon
    - benchmark symbol
    - concentration metrics
  - chapter framework status:
    - score breakdown
    - hard-gate caps
    - point-in-time assumptions
    - required figures coverage
  - release gate status:
    - high severity pass/fail
    - medium severity notes
    - release status
    - reviewer sign-off
    - example selection rule
    - claim-to-metric links
    - JSON/CSV artifact paths
    - classification label
- [ ] Validate payload against schema.
- [ ] Add tests for missing required sections.

Done when:

- [ ] `chapter_payload.json` is deterministic and schema-valid.
- [ ] The payload contains all numbers DeepSeek may mention.

## Phase D - Deterministic Core Chapter

Goal:

Generate a readable chapter core without AI.

Required sections:

- [ ] Results Snapshot
- [ ] Research Lane And Data Scope
- [ ] Statistics Contract Status
- [ ] Chapter Framework Score And Hard Gates
- [ ] Release Gate And Red-Team Notes
- [ ] Identification Guidelines
- [ ] Scanner Rule Provenance
- [ ] Source Traceability
- [ ] Vietnam Market Statistics
- [ ] Market Regime And Context Splits
- [ ] Failure, Target, Retest, And Event-Path Behavior
- [ ] Quantile, CI, And Survival Diagnostics
- [ ] Reproducibility Artifacts And Example Selection
- [ ] Golden Fixture Summary
- [ ] Governance Status
- [ ] Caveats / Known Limitations

Tasks:

- [ ] Add a core chapter renderer.
- [ ] Render `chapter_core.md`.
- [ ] Add tests for required headings and required tables.

Done when:

- [ ] `chapter_core.md` can stand alone without DeepSeek commentary.

## Phase E - DeepSeek V4 Flash Commentary

Goal:

Use DeepSeek as the editorial layer only.

Tasks:

- [ ] Add V2 monograph prompt builder.
- [ ] Use default model `deepseek-v4-flash`.
- [ ] Allow override with `DEEPSEEK_MODEL`.
- [ ] Pass only `chapter_payload.json` and `chapter_core.md` to DeepSeek.
- [ ] Generate `chapter_commentary.md`.
- [ ] Validate commentary:
  - no unsupported numeric claims
  - no strategy recommendation beyond payload
  - required commentary headings present
- [ ] Cache commentary by prompt fingerprint.

Done when:

- [ ] DeepSeek can improve readability without changing facts.
- [ ] Invalid commentary is rejected rather than silently included.

## Phase F - Final Markdown And PDF

Goal:

Produce the first V2 PDF chapter.

Tasks:

- [ ] Merge deterministic core and DeepSeek commentary into `chapter_final.md`.
- [ ] Add PDF renderer command.
- [ ] Render `broadening_bottoms.pdf`.
- [ ] Store render metadata:
  - source payload hash
  - spec hash
  - DeepSeek model
  - commentary fingerprint
  - render timestamp
- [ ] Add fallback path when `pandoc` or PDF engine is unavailable.

Done when:

- [ ] A user can open `broadening_bottoms.pdf` and inspect the research chapter.

## Phase G - Human Review Gate

Goal:

Decide whether the PDF is good enough to become the pattern template.

Review questions:

- [ ] Does the PDF read like a real chart-pattern research chapter?
- [ ] Are source references visible enough?
- [ ] Are Vietnam statistics clear and not overclaimed?
- [ ] Does DeepSeek commentary stay grounded in the payload?
- [ ] Are caveats prominent enough?
- [ ] Is this template reusable for `double_bottoms`?

Done when:

- [ ] The review decision is written as `approved_template`, `needs_revision`, or `blocked`.

## Phase H - Replication

Goal:

Only after `broadening_bottoms.pdf` is approved, replicate the workflow.

Order:

- [ ] `double_bottoms`
- [ ] `double_tops`
- [ ] `head_and_shoulders_tops`
- [ ] `head_and_shoulders_bottoms`
- [ ] `triangles_ascending`
- [ ] `cup_with_handle`

Done when:

- [ ] Each replicated pattern has its own official-ready V2 detector and PDF monograph.
