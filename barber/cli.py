"""Trim plain text on stdin before it enters a coding agent's context."""
from __future__ import annotations

import argparse
import math
import sys

from . import trim


def main() -> int:
    parser = argparse.ArgumentParser(prog="barber-trim")
    parser.add_argument("--query", required=True, help="question the text should answer")
    parser.add_argument("--keep", type=float, default=0.8,
                        help="fraction of chunks to keep (default: 0.8)")
    args = parser.parse_args()
    if not math.isfinite(args.keep) or not 0 <= args.keep <= 1:
        parser.error("--keep must be between 0 and 1")

    source = sys.stdin.read()
    result = trim([{"role": "user", "content": source},
                   {"role": "user", "content": args.query}], keep=args.keep)
    selected = result.messages[0]["content"]
    sys.stdout.write(selected if result.tokens_saved > 0 and len(selected) < len(source)
                     else source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
