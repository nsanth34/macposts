"""Transportation network modeling toolkit.

This package provides tools for various tasks in transportation network
modeling.

"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from _macposts_ext import set_random_state, Tdsp  # noqa: F401
from ._compat import *  # noqa: F401,F403
from .backends import (  # noqa: F401
    configure_array_backend,
    get_array_module,
    get_backend_state,
    is_gpu_enabled,
    reset_array_backend,
    use_array_backend,
)
from .dta import Dta, Mcdta, Mmdta  # noqa: F401
from .graph import Graph  # noqa: F401
