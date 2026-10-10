"""Local AI outgoing invoice checker.

A hybrid pipeline: a small local language model handles unstructured extraction, while all legal,
arithmetic and identifier validation is done by deterministic rules.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
