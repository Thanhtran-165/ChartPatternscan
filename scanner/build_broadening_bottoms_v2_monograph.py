"""Build the first Scanner V2 PDF monograph: Broadening Bottoms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.v2.broadening_bottoms_monograph import (
    DEFAULT_INDEX_DB,
    DEFAULT_INDEX_SYMBOL,
    DEFAULT_OUT_DIR,
    DEFAULT_SOURCE_DIR,
    run_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Market Stats V1 stock_series directory")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output artifact directory")
    parser.add_argument("--limit-symbols", type=int, default=None, help="Optional symbol limit for smoke runs")
    parser.add_argument("--skip-ai", action="store_true", help="Skip DeepSeek commentary")
    parser.add_argument("--example-universe", default="VN100", help="Preferred universe for example detections")
    parser.add_argument("--index-db", default=str(DEFAULT_INDEX_DB), help="Index price DB used for VNINDEX regime split")
    parser.add_argument("--index-symbol", default=DEFAULT_INDEX_SYMBOL, help="Index symbol used for regime split")
    args = parser.parse_args()

    paths = run_pipeline(
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        limit_symbols=args.limit_symbols,
        skip_ai=bool(args.skip_ai),
        example_universe=str(args.example_universe),
        index_db=Path(args.index_db),
        index_symbol=str(args.index_symbol),
    )
    print(paths["pdf"])


if __name__ == "__main__":
    main()
