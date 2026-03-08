# Book V1 Legacy Status

## Status

`Book v1` is now a legacy publication path.

Primary legacy entrypoint:

- [build_book_vi.py](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/scanner/build_book_vi.py)

This path remains in the repository for:

- historical comparison
- artifact inspection
- prompt and layout reference
- backward-compatible regeneration of older outputs when explicitly needed

It is no longer the primary publication or research path for the project.

## Why It Was Frozen

Book v1 was built around a `chapter + AI narrative` flow.

That was useful as a prototype, but it no longer matches the direction of the project:

- the project is research-first, not prose-first
- deterministic Vietnam research outputs now exist
- the publication layer should be downstream of the research engine
- AI commentary should not be responsible for making the report feel complete

## Support Policy

Book v1 should receive only:

- bug fixes needed to keep legacy regeneration possible
- compatibility fixes when shared scanner/report contracts change
- audit fixes if a legacy artifact must be reproduced for comparison

Book v1 should not receive:

- new publication features
- new chapter logic
- new report structure work
- new AI prompting work for mainline research output

## Replacement Path

The replacement architecture is documented in:

- [architecture.md](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/docs/publication/book-v2/architecture.md)
- [data-contracts.md](/Users/bobo/Library/Mobile%20Documents/com~apple~CloudDocs/main%20sonet/Nghie%CC%82n%20cu%CC%9B%CC%81u%20mo%CC%82%20hi%CC%80nh%20ne%CC%82%CC%81n/docs/publication/book-v2/data-contracts.md)

The intended replacement flow is:

`scanner -> Vietnam research dataset -> deterministic chapter core -> optional DeepSeek commentary -> final publication`

## Exit Rule

Phase 1 is considered complete when:

- Book v1 is clearly marked as legacy in code and documentation
- Book v2 data contracts are defined
- no new feature planning assumes Book v1 is the main publication path
