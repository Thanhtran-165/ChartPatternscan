"""
Legacy wrapper for backward compatibility.

Use `scanner/review_book_v1_output.py` for the standardized entrypoint.
"""

from __future__ import annotations

from review_book_v1_output import *  # type: ignore


if __name__ == "__main__":
    from review_book_v1_output import main

    main()
