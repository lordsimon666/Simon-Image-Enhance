"""Non-destructive edit catalogue: settings per image are stored as JSON in the app
data directory (the original files are never modified).

Portable mode: if a file named ``portable.txt`` sits next to the executable, all data
is kept in a ``data`` folder beside it instead of the user profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from .processing import Params

APP_NAME = "SimonImageEnhance"


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_portable() -> bool:
    return os.path.exists(os.path.join(app_dir(), "portable.txt"))


def data_dir() -> str:
    if is_portable():
        d = os.path.join(app_dir(), "data")
    elif sys.platform == "win32":
        d = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP_NAME)
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        d = os.path.join(base, APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


class EditStore:
    def __init__(self, root: str | None = None):
        self.root = os.path.join(root or data_dir(), "edits")
        os.makedirs(self.root, exist_ok=True)

    def _file(self, image_path: str) -> str:
        key = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()
        return os.path.join(self.root, key + ".json")

    def get(self, image_path: str) -> Params | None:
        try:
            with open(self._file(image_path), "r", encoding="utf-8") as f:
                return Params.from_dict(json.load(f)["params"])
        except Exception:
            return None

    def set(self, image_path: str, params: Params) -> None:
        f = self._file(image_path)
        if params.is_default():
            if os.path.exists(f):
                os.remove(f)
            return
        tmp = f + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"path": os.path.abspath(image_path), "params": params.to_dict()}, fh, indent=1)
        os.replace(tmp, f)


class Settings:
    def __init__(self):
        self.path = os.path.join(data_dir(), "settings.json")
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=1)
        except Exception:
            pass
