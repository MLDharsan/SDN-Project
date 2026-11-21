#!/usr/bin/env python3
"""Remove leftover Mininet state after a crash (root).

    sudo venv/bin/python -m mininet_integration.cleanup

Runs Mininet's own cleanup (``mn -c``) and terminates os-ken / iperf3
processes started by this extension. It never touches project files.
"""

from __future__ import annotations

import subprocess
import sys

sys.dont_write_bytecode = True

from mininet_integration import _paths  # noqa: E402


def main() -> int:
    subprocess.run(["pkill", "-f", "mininet_integration/osken_apps/proactive_routing_app.py"], check=False)
    subprocess.run(["pkill", "-f", "iperf3 -s -p"], check=False)
    subprocess.run(["pkill", "-f", "iperf3 -c 10.0."], check=False)
    _paths.ensure_mininet_importable()
    from mininet.clean import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
