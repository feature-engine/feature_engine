import pathlib

import feature_engine

PACKAGE_ROOT = pathlib.Path(feature_engine.__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"

name = "feature_engine"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
