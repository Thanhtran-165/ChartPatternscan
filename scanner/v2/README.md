# Scanner V2 Contract Layer

Scanner V2 starts from source fidelity, not detector convenience.

The current package is a contract-first foundation:

1. `core_patterns.json` stores the first provenance-seeded rules for the core pattern group.
2. `taxonomy_lineage.json` maps each pattern across the required chain:
   `Bulkowski chapter -> extracted source rule -> normalized rule spec -> scanner pattern key -> result payload -> book v2 chapter`.
3. `contracts.py` compiles a pattern only when provenance, lineage, and rule coverage are valid.

Activation policy:

- No provenance, no official scanner.
- Evidence excerpts must align to the claimed PDF pages.
- Unsupported rule type is a compile error.
- Golden fixtures are required before a pattern can be activated as official.
- Result metadata uses a full normalized spec hash, so any rule change changes the scanner identity.

Current official V2 patterns:

- `broadening_bottoms`
- `bull_flags` (available-series watchlist-reference candidate; active Market Stats universe gate passes, but no full point-in-time universe claim)

Remaining core patterns are still draft and should not be used as official research scanners until
their provenance and fixtures pass the same gate.

The legacy scanner remains useful as a benchmark/prototype. V2 is the path for research-grade scans.

## Scanner Matrix

The scanner matrix is the standard expansion path for multiple chart patterns:

```text
Independent pattern scanner -> scanner matrix event contract -> common metrics/charts/PDF/watchlist
```

Bull Flag is the reference implementation. It remains an independent scanner, but its output is normalized by
`scanner.v2.matrix.normalize_bull_flag_events` into the shared event schema:

- `pattern_id`, `scanner_pattern_key`, `spec_hash`, `source_chapters`
- `formation_start`, `formation_end`, `confirmation_date`, `direction`, `confirmation_price`
- `target_family`, `setup_score`, `confirmation_score`, `followthrough_score`, `context_score`
- `market_regime`, `liquidity_bucket`, `path_quality`, `data_quality_bucket`

The rule for scaling is strict:

- scanner logic is pattern-specific;
- event output is common;
- Bull Flag is the template for matrix output, not the geometry template for other patterns.

Build the current matrix artifacts with:

```bash
python scanner/run_scanner_matrix.py
```

Outputs:

- `artifacts/scanner_v2/scanner_matrix/scanner_matrix_events.csv`
- `artifacts/scanner_v2/scanner_matrix/scanner_matrix_manifest.json`
