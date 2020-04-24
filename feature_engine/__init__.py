name = "feature_engine"

VERSION_PATH = 'feature_engine/VERSION'

with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()
