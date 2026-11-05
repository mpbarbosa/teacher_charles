#!/usr/bin/env python3
"""Regenera ``data/schema/contrastive-item.schema.json`` a partir do modelo.

O modelo pydantic em ``teaching/contrastive.py`` é a fonte única da verdade; o
JSON Schema existe para editor e validador externos olharem o YAML sem importar
Python. Manter os dois à mão divergiria — então um é gerado do outro, e
``tests/test_contrastive.py`` falha se o arquivo versionado sair de sincronia.

    uv run python scripts/dump_contrastive_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from teacher_charles.teaching.contrastive import (  # noqa: E402
    SCHEMA_RELATIVE_PATH,
    schema_json_text,
)


def main() -> int:
    target = REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    text = schema_json_text()

    if target.is_file() and target.read_text(encoding="utf-8") == text:
        print(f"já em sincronia: {target.relative_to(REPO_ROOT)}")
        return 0

    target.write_text(text, encoding="utf-8")
    print(f"escrito: {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
