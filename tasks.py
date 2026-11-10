"""The two tasks the lab runs. Each exposes the same small surface so crew.py does not care which it is."""
from __future__ import annotations

import json
from types import SimpleNamespace

import data as _planted
import needle as _needle
from crew_rules import planted_rule_scout

PLANTED = SimpleNamespace(
    name="planted",
    question="Why did blended checkout conversion fall last week?",
    generate=_planted.generate,
    slices=_planted.slices,
    labels_text=lambda: "Labels: " + ", ".join(f"{k} ({v})" for k, v in _planted.CAUSES.items()) + ", unclear.",
    normalize=lambda c: (lambda x: x if x in set(_planted.CAUSES) | {"unclear"} else "unclear")(c.strip().lower().replace(" ", "_")),
    in_slice=lambda answer, table: True,          # any label may be named from any slice
    render=lambda table: json.dumps(table),
    rule_scout=planted_rule_scout,
    score=lambda answer, table: None,            # no single number says how big a cause is; the vote falls back to confidence
)

NEEDLE = SimpleNamespace(
    name="needle",
    question=_needle.QUESTION,
    generate=_needle.generate,
    slices=_needle.slices,
    labels_text=_needle.labels_text,
    normalize=_needle.normalize,
    in_slice=_needle.in_slice,
    render=_needle.render,
    rule_scout=_needle.rule_scout,
    score=_needle.score,
)

NEEDLE_XL = SimpleNamespace(**{**vars(NEEDLE), "name": "needle-xl",
                              "generate": lambda seed, cause=None: _needle.generate(seed, cause, tiers=5),
                              "labels_text": _needle.labels_text_xl})

BY_NAME = {"planted": PLANTED, "needle": NEEDLE, "needle-xl": NEEDLE_XL}
