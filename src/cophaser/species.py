"""Guess mouse vs human from gene symbol capitalisation."""

from __future__ import annotations

import re
from typing import Iterable, Literal

Species = Literal["mouse", "human", "ambiguous"]

_ALPHA_ONLY = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*$")
_ENSEMBL_ID = re.compile(r"^ENS[A-Z]*G\d+")


def _is_human_like(symbol: str) -> bool:
    return symbol.isupper()


def _is_mouse_like(symbol: str) -> bool:
    return symbol[0].isupper() and symbol[1:] == symbol[1:].lower()


def detect_species(
    var_names: Iterable[str],
    min_genes: int = 50,
    threshold: float = 0.6,
) -> tuple[Species, dict]:
    """Guess whether ``var_names`` are mouse (``Per1``) or human (``PER1``) symbols.

    Returns ("mouse" | "human" | "ambiguous", diagnostic counts). "ambiguous" when fewer
    than `min_genes` symbols are usable or neither convention reaches `threshold`.
    """
    usable = [
        g
        for g in var_names
        if isinstance(g, str)
        and _ALPHA_ONLY.match(g)
        and len(g) >= 2
        and not _ENSEMBL_ID.match(g)
    ]

    n_human = sum(_is_human_like(g) for g in usable)
    n_mouse = sum(_is_mouse_like(g) and not _is_human_like(g) for g in usable)
    n_other = len(usable) - n_human - n_mouse

    diagnostics = {
        "n_usable": len(usable),
        "n_human_like": n_human,
        "n_mouse_like": n_mouse,
        "n_other": n_other,
    }

    n_conventioned = n_human + n_mouse
    if len(usable) < min_genes or n_conventioned == 0:
        return "ambiguous", diagnostics

    if n_human / n_conventioned >= threshold:
        return "human", diagnostics
    if n_mouse / n_conventioned >= threshold:
        return "mouse", diagnostics
    return "ambiguous", diagnostics
