"""Public API for the official implementation of the SelfTR paper.

The VGGT implementation remains available under ``vggt_omega`` for checkpoint
and upstream compatibility.  New code should import :class:`SelfTR` from this
package.
"""

from .config import SelfTRConfig
from .checkpoint import CheckpointInfo, inspect_checkpoint
from .identity import DEFAULT_METHOD_NAME, METHOD_ID, resolve_method_name

__version__ = "0.1.0"


def __getattr__(name: str):
    # Delay the backbone import so ``vggt_omega`` can use selftr.identity
    # during package initialization without an import cycle.
    if name == "SelfTR":
        from .model import SelfTR

        return SelfTR
    raise AttributeError(name)


__all__ = [
    "DEFAULT_METHOD_NAME",
    "CheckpointInfo",
    "METHOD_ID",
    "SelfTR",
    "SelfTRConfig",
    "inspect_checkpoint",
    "resolve_method_name",
    "__version__",
]
