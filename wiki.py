from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from skills.wiki.wiki_cli import main as wiki_main


def main(argv: list[str] | None = None) -> int:
    return wiki_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
