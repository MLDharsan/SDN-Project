#!/usr/bin/env python3
"""Compatibility entry point for the valid-random fair baseline."""

import sys

from evaluate_fair_baselines import main


if __name__ == "__main__":
    if "--policies" not in sys.argv:
        sys.argv.extend(["--policies", "random_valid"])
    main()
