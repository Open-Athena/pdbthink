"""Programmatic question generators and gold-answer oracles.

Importing this package registers every V1 family. The extension backlog in
section 8 is intentionally not implemented; new families only need to subclass
:class:`~pdbthink.generators.base.Generator` and call ``register``.
"""

from . import geometry_family, interface, local, parsing  # noqa: F401  (registration)
from .base import (
    Analysis,
    GenerationContext,
    Generator,
    OracleResult,
    Proposal,
    Rejection,
    all_generators,
    get_generator,
    register,
)
from .two_state import T01, TwoStateContext, build_two_state_context

#: The 19 required V1 families, in specification order.
V1_FAMILIES = (
    "P01", "P02", "P03",
    "G01", "G02", "G03", "G04",
    "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
    "I01", "N01", "T01",
)

__all__ = [
    "Analysis",
    "GenerationContext",
    "Generator",
    "OracleResult",
    "Proposal",
    "Rejection",
    "T01",
    "TwoStateContext",
    "V1_FAMILIES",
    "all_generators",
    "build_two_state_context",
    "get_generator",
    "register",
]
