from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple metrics.json files into one comparison table.")
    parser.add_argument(
        "--entries",
        type=str,
        nargs="+",
        required=True,
        help="Entries in the format alias=outputs/run_name/metrics.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "comparison_table.json",
    )
    return parser.parse_args()


def parse_entries(raw_entries: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for entry in raw_entries:
        if "=" not in entry:
            raise ValueError(f"Invalid entry (missing '='): {entry}")
        alias, path_str = entry.split("=", 1)
        alias = alias.strip()
        path = Path(path_str.strip())
        if not alias:
            raise ValueError(f"Invalid alias in entry: {entry}")
        out[alias] = path
    return out


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "map": payload.get("map"),
        "macro_f1": payload.get("macro_f1"),
        "micro_f1": payload.get("micro_f1"),
        "per_attribute_ap": payload.get("per_attribute_ap"),
    }


def main() -> None:
    args = parse_args()
    entries = parse_entries(args.entries)

    table: Dict[str, Any] = {}
    for alias, rel_path in entries.items():
        abs_path = rel_path if rel_path.is_absolute() else REPO_ROOT / rel_path
        table[alias] = load_metrics(abs_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, ensure_ascii=True)
    print(f"Wrote comparison table: {args.output_path}")


if __name__ == "__main__":
    main()

