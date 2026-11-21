"""Path helpers shared by every Mininet integration module.

Importing this module:
- disables .pyc writing, so running under ``sudo`` never creates root-owned
  ``__pycache__`` files inside the existing simulator packages;
- puts the project root on ``sys.path`` so the unchanged simulator modules
  can be imported read-only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "mininet_results"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mininet_os3e.json"
OSKEN_APP_PATH = PACKAGE_DIR / "osken_apps" / "proactive_routing_app.py"

# Directories the Mininet extension must never write into.
PROTECTED_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "experiments",
    PROJECT_ROOT / "environments",
    PROJECT_ROOT / "utils",
    PROJECT_ROOT / "forecasting",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "tensorboard",
    PROJECT_ROOT / "report",
]

SYSTEM_MININET_CANDIDATES = [
    Path("/usr/lib/python3/dist-packages/mininet"),
    Path("/usr/local/lib/python3/dist-packages/mininet"),
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve(path: str | Path | None) -> Path | None:
    """Resolve a config path relative to the project root."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def assert_safe_output_dir(path: Path) -> Path:
    """Refuse to write Mininet output anywhere outside ``mininet_results/``."""
    path = path.resolve()
    for protected in PROTECTED_DIRS:
        if path == protected.resolve() or protected.resolve() in path.parents:
            raise ValueError(f"Refusing to write Mininet output into protected directory: {path}")
    if RESULTS_ROOT.resolve() not in path.parents and path != RESULTS_ROOT.resolve():
        raise ValueError(f"Mininet output must live under {RESULTS_ROOT}, got {path}")
    return path


def ensure_mininet_importable() -> None:
    """Make the system Mininet package importable from the project venv.

    Mininet is installed for the system Python, while PyTorch lives in the
    project venv. Instead of installing anything into the venv we expose only
    the pure-Python ``mininet`` package through a symlink inside this folder.
    """
    try:
        import mininet  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in SYSTEM_MININET_CANDIDATES:
        if (candidate / "net.py").exists():
            vendor = PACKAGE_DIR / "_vendor"
            vendor.mkdir(exist_ok=True)
            link = vendor / "mininet"
            if not link.exists():
                link.symlink_to(candidate, target_is_directory=True)
                chown_to_invoking_user(vendor)
            sys.path.insert(0, str(vendor))
            import mininet  # noqa: F401
            return
    raise ImportError(
        "Mininet Python package not found. Install it with `sudo apt install mininet`."
    )


def chown_to_invoking_user(path: Path) -> None:
    """When run via sudo, give generated files back to the real user."""
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or uid is None or gid is None or not path.exists():
        return
    uid_i, gid_i = int(uid), int(gid)
    targets = [path] + list(path.rglob("*")) if path.is_dir() else [path]
    for item in targets:
        try:
            os.lchown(item, uid_i, gid_i)
        except OSError:
            pass
