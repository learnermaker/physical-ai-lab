"""
Module 5 — Foundation Models utilities.

Exports:
    load_cache  — load cached_responses.json, exit(1) on missing/empty file
    get_cached_response  — normalised key lookup with random fallback
"""

import json
import random
import re
import sys
from pathlib import Path


def load_cache(path: Path | None = None) -> dict:
    """
    Load cached_responses.json. Returns the dict.
    Raises SystemExit(1) if the file is absent or contains no valid entries.

    Args:
        path: Explicit path to the JSON file.  When None (default), resolves
              to ``<this_file's_directory>/cached_responses.json`` so the
              function always finds the asset regardless of working directory.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "cached_responses.json"
    else:
        path = Path(path)

    if not path.exists():
        print(f"ERROR: Cache file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open() as f:
        data = json.load(f)

    if not data:
        print("ERROR: cached_responses.json contains no valid entries.", file=sys.stderr)
        sys.exit(1)

    return data


def _normalise_key(prompt: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    return re.sub(r"\s+", " ", prompt.strip().lower())


def get_cached_response(prompt: str, cache: dict) -> object:
    """
    Look up a prompt in the cache using normalised key matching.

    Tries an exact normalised match first; falls back to a uniformly random
    entry if no match is found.

    Args:
        prompt: The prompt string to look up.
        cache:  Dict returned by ``load_cache()``.

    Returns:
        The cached value (str or dict) for the matched key.
    """
    key = _normalise_key(prompt)
    if key in cache:
        return cache[key]
    return random.choice(list(cache.values()))
