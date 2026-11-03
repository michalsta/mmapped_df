from .main import *
from .main import __all__ as _main_all

# Numba helpers, formerly re-exported through `from .main import *`. Resolved
# lazily (PEP 562) so that `import mmapped_df` does not load Numba.
_LAZY_NUMBA = ("mkindex", "count_falses_in_groups")

__all__ = [*_main_all, *_LAZY_NUMBA]


def __getattr__(name):
    if name in _LAZY_NUMBA:
        from . import numba_helper

        return getattr(numba_helper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NUMBA))
