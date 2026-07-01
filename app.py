"""
Fuzzy string matching utility.
Accepts a JSON request file, finds N most similar strings, writes JSON response.

Usage:
    app.exe <request_json_path> <response_json_path>
"""

import sys
import json
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process


def read_request(path: Path) -> dict[str, Any]:
    """Read and parse the JSON request file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_response(path: Path, payload: dict[str, Any]) -> None:
    """Write the JSON response file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def validate_request(data: dict[str, Any]) -> tuple[str, int, list[tuple[int, str]]]:
    """
    Extract and validate fields from the parsed request.
    Returns (query, limit, items) where items is a list of (original_index, text).
    Original indices are preserved so the caller can map results back to the input array.
    """
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Поле 'query' должно быть непустой строкой")

    limit = data.get("limit", 10)
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Поле 'limit' должно быть целым числом >= 1")

    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("Поле 'items' должно быть массивом")

    # Keep original index so results can be mapped back; skip null and blank strings
    items: list[tuple[int, str]] = [
        (i, item)
        for i, item in enumerate(raw_items)
        if isinstance(item, str) and item.strip()
    ]

    return query.strip(), limit, items


def score_item(query: str, item: str) -> int:
    """
    Compute a composite similarity score between query and item.

    Combines three rapidfuzz metrics for best quality on Russian text:
    - token_sort_ratio  : handles word-order differences
    - token_set_ratio   : handles substring / superset cases
    - WRatio            : weighted combination of multiple algorithms

    The maximum of the three is used so that whichever metric best captures
    the similarity wins.
    """
    scores = (
        fuzz.token_sort_ratio(query, item),
        fuzz.token_set_ratio(query, item),
        fuzz.WRatio(query, item),
    )
    return max(scores)


def find_matches(query: str, items: list[tuple[int, str]], limit: int) -> list[dict[str, Any]]:
    """
    Find the top-N most similar strings to query from items.
    Returns list of dicts sorted by score descending, each including the original index.
    """
    if not items:
        return []

    scored: list[tuple[int, str, float]] = [
        (idx, text, score_item(query, text)) for idx, text in items
    ]

    # Sort by score descending; for equal scores preserve original order (stable sort)
    scored.sort(key=lambda x: x[2], reverse=True)

    # Clamp limit to available items
    top = scored[:limit]

    return [{"index": idx, "text": text, "score": round(score)} for idx, text, score in top]


def run(request_path: Path, response_path: Path) -> None:
    """Main processing pipeline."""
    data = read_request(request_path)
    query, limit, items = validate_request(data)
    results = find_matches(query, items, limit)
    write_response(response_path, {"success": True, "results": results})


def main() -> None:
    """Entry point: parse CLI arguments and run, writing errors to response file."""
    if len(sys.argv) != 3:
        print(
            "Использование: app.exe <request_json_path> <response_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])

    try:
        run(request_path, response_path)
    except Exception as exc:  # noqa: BLE001
        # Any error is written to the response file so the caller can handle it
        error_payload = {"success": False, "error": str(exc)}
        try:
            write_response(response_path, error_payload)
        except Exception as write_exc:
            # Last resort: if we cannot write the response, print to stderr
            print(f"Не удалось записать файл ответа: {write_exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
