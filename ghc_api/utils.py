import os
import platform


def get_config_dir():
    """Get the config directory path based on the OS"""
    if platform.system() == "Windows":
        return os.path.expandvars("%APPDATA%/ghc-api")
    else:
        return os.path.expanduser("~/.ghc-api")
