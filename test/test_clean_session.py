#!/usr/bin/env python3
"""
Backward-compatible entrypoint.

Canonical runner moved to `apps/runners/clean_session.py`.
"""

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from apps.runners.clean_session import main


class CleanSessionSmokeTests(unittest.TestCase):
    def test_runner_entrypoint_exists(self) -> None:
        self.assertTrue(callable(main))


if __name__ == "__main__":
    main()
