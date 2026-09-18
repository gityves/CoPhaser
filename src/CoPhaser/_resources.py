import atexit
from contextlib import ExitStack
from importlib import resources

_PACKAGE = "CoPhaser"
_RESOURCE_DIR = "resources"

# Kept open for the life of the process; only does anything for non-filesystem installs.
_extracted = ExitStack()
atexit.register(_extracted.close)
_cache: dict[str, str] = {}


def resource_path(name: str) -> str:
    """Absolute path to ``CoPhaser/resources/<name>``.

    Parameters
    ----------
    name: file name inside the packaged resources directory, e.g. "CCG_annotated.csv".

    Returns
    -------
    A filesystem path, as a string, suitable for ``open`` or ``pandas.read_csv``.
    """
    if name not in _cache:
        ref = resources.files(_PACKAGE) / _RESOURCE_DIR / name
        _cache[name] = str(_extracted.enter_context(resources.as_file(ref)))
    return _cache[name]
