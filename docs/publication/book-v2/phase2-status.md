# Book V2 Phase 2 Status

## Status

Phase 2 is complete.

The deterministic chapter core for Book v2 now exists as a working build path.

## What Was Delivered

### 1. Deterministic monograph builder

- [build_pattern_monographs.py](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/scanner/build_pattern_monographs.py)

This script builds:

- `chapter_payload.json`
- `chapter_core.md`

for each pattern.

### 2. Sample-selection policy

- [sample-selection-rules.md](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/docs/publication/book-v2/sample-selection-rules.md)

This freezes deterministic rules for:

- `best_case`
- `typical_case`
- `stress_case`
- `calib_reference`

### 3. Full deterministic build

Full output was built to:

- [index.md](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/scan_results/audits/spec-audit-20260308/book-v2-workbench/phase2-core-en/index.md)
- [index.json](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/scan_results/audits/spec-audit-20260308/book-v2-workbench/phase2-core-en/index.json)

Pattern count built:

- `53`

Example chapter cores:

- [broadening_bottoms](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/scan_results/audits/spec-audit-20260308/book-v2-workbench/phase2-core-en/broadening_bottoms/chapter_core.md)
- [double_bottoms_adam_adam](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/scan_results/audits/spec-audit-20260308/book-v2-workbench/phase2-core-en/double_bottoms_adam_adam/chapter_core.md)

## What Phase 2 Now Guarantees

- every pattern chapter has a deterministic payload
- every chapter can render without AI
- representative cases are selected deterministically
- benchmark and governance status are attached to the chapter core
- the publication layer is now downstream of the research engine

## What Phase 2 Does Not Do

Phase 2 does not yet add:

- AI commentary
- final book assembly
- PDF/book packaging for the new flow

Those belong to Phase 3.

## Next Step

Phase 3 should add:

- optional DeepSeek commentary
- commentary cache keyed by deterministic payload
- final chapter assembly
- final Book v2 assembly
