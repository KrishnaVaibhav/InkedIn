"""Workspace-local storage roots.

Owner requirement: every byte InkedIn writes stays inside the workspace.
No %LOCALAPPDATA%, no system temp, no user cache directories.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller exe: __file__ lives in a throwaway temp extract dir — anchor
    # the workspace next to the executable instead (portable-app layout).
    _DEFAULT_ROOT = Path(sys.executable).resolve().parent
else:
    # Repo root = parent of python/inkedin_core/, overridable for tests.
    _DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return Path(os.environ.get("INKEDIN_ROOT", _DEFAULT_ROOT))


def data_dir() -> Path:
    return workspace_root() / "data"


def jobs_dir() -> Path:
    return data_dir() / "jobs"


def models_dir() -> Path:
    return workspace_root() / "models" / "weights"


def db_path() -> Path:
    return data_dir() / "inkedin.db"


def exports_dir() -> Path:
    return data_dir() / "exports"


def ensure_dirs() -> None:
    for d in (data_dir(), jobs_dir(), models_dir(), exports_dir()):
        d.mkdir(parents=True, exist_ok=True)
