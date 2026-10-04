"""Probabilistic Constraint Graphs for Uncertainty-Aware Code Verification.

22AIE301 course project.
"""

from .blocks import Block, extract_blocks
from .evidence import Evidence, collect_all
from .graph import build_graph, structural_importance
from .inference import BlockPosterior, infer, repair_targets, review_ranking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Analysis, analyze


def __getattr__(name: str):
    if name in {"Analysis", "analyze"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)


__version__ = "0.1.0"

__all__ = [
    "Block",
    "extract_blocks",
    "Evidence",
    "collect_all",
    "build_graph",
    "structural_importance",
    "BlockPosterior",
    "infer",
    "repair_targets",
    "review_ranking",
    "Analysis",
    "analyze",
]
