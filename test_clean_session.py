#!/usr/bin/env python3
"""
Backward-compatible entrypoint.

Canonical runner moved to `apps/runners/clean_session.py`.
"""

from apps.runners.clean_session import main


if __name__ == "__main__":
    main()
