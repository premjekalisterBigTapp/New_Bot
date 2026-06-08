from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def normalize_product_key(name: str) -> str:
    base = name.lower().strip()
    # Strip common suffixes
    if base.endswith("_benefits"):
        base = base[:-9]  # len("_benefits") == 9
    if base.endswith("-benefits"):
        base = base[:-10]  # len("-benefits") == 10 (hyphen + 9 letters)
    # Keep only alphanumerics
    return "".join(ch for ch in base if ch.isalnum())


def read_benefit_files(input_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    data: Dict[str, Dict[str, List[str]]] = {}
    for path in sorted(input_dir.glob("*benefits.txt")):
        try:
            key = normalize_product_key(path.stem)
            text = path.read_text(encoding="utf-8")
            if key not in data:
                data[key] = {"docs": [], "sources": []}
            data[key]["docs"].append(text)
            data[key]["sources"].append(str(path))
        except Exception:
            continue
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benefits_raw.json from plain-text sources.")
    parser.add_argument(
        "--input",
        default=str(Path.cwd() / "Admin" / "source_db" / "benefits"),
        help="Directory containing *_benefits.txt files",
    )
    parser.add_argument(
        "--output",
        default=str(Path.cwd() / "bigtapp" / "src" / "bigtapp" / "config" / "benefits_raw.json"),
        help="Path to write the consolidated JSON file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = read_benefit_files(input_dir)

    # Write pretty JSON for readability; preserve unicode
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


