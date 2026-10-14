import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

# import ./*.py and ./dirs
for _p in _pkg_path.iterdir():
    # must be python file or directory with __init__.py
    if _p.is_file() and _p.suffix != ".py":
        continue
    if _p.is_dir() and not (_p / "__init__.py").exists():
        continue

    # skip hidden/private files, incl. __init__.py
    if _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
