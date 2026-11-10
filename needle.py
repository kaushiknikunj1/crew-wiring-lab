"""Task B — the needle. 240 segments (12 regions x 4 platforms x 5 payment methods), 28 days.

Blended conversion dips a little last week. One segment's payment success rate collapsed; eight distractor
segments drifted by a few percent. The answer is the exact segment id, e.g. "west/android/upi".

The evidence does not shrink: every wiring gets the same 240 rows of baseline vs last-week rates. The solo
agent reads all 240 in one prompt; each scout reads 40. This is the task where splitting the haystack should
earn its cost — if it does not, that is the result.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

REGIONS = ["north", "south", "east", "west", "central", "ncr", "mumbai", "bengaluru", "hyderabad", "pune", "kolkata", "chennai"]
PLATFORMS = ["android", "ios", "web", "mweb"]
METHODS = ["upi", "card", "netbanking", "wallet", "emi"]
DAYS, BASE_DAYS, N_SCOUTS = 28, 21, 6


@dataclass
class Needle:
    seed: int
    cause: str                                   # the segment id
    distractors: list = field(default_factory=list)
    base: dict = field(default_factory=dict)     # seg -> {"sessions_per_day", "checkout_rate", "success_rate"}
    last: dict = field(default_factory=dict)

    def rows(self) -> list[dict]:
        out = []
        for seg in self.base:
            b, l = self.base[seg], self.last[seg]
            out.append({"segment": seg, "sessions_per_day": l["sessions_per_day"],
                        "checkout_rate_base": b["checkout_rate"], "checkout_rate_last": l["checkout_rate"],
                        "success_rate_base": b["success_rate"], "success_rate_last": l["success_rate"],
                        # derived, NOT shown in the prompt (see render); present so a scout's "down 45%" can be pinned
                        "success_rate_delta_pct": round((l["success_rate"] / b["success_rate"] - 1) * 100, 1),
                        "checkout_rate_delta_pct": round((l["checkout_rate"] / b["checkout_rate"] - 1) * 100, 1)})
        return out

    def headline(self) -> dict:
        def conv(d):
            s = sum(v["sessions_per_day"] for v in d.values())
            return sum(v["sessions_per_day"] * v["checkout_rate"] * v["success_rate"] for v in d.values()) / s
        b, l = conv(self.base), conv(self.last)
        return {"baseline_conversion": round(b, 4), "last_week_conversion": round(l, 4), "relative_change_pct": round((l / b - 1) * 100, 1)}


def generate(seed: int, cause: str | None = None, tiers: int = 1) -> Needle:
    """tiers=1 -> 240 segments; tiers=5 adds a merchant tier (t1..t5) -> 1,200 segments."""
    rng = random.Random(1000 + seed + 7919 * (tiers - 1))
    segs = [f"{r}/{p}/{m}" + (f"/t{t}" if tiers > 1 else "") for r in REGIONS for p in PLATFORMS for m in METHODS for t in range(1, tiers + 1)]
    base, last = {}, {}
    for seg in segs:
        m = seg.split("/")[2]
        sess = rng.randint(120, 2400)
        chk = round(rng.uniform(0.07, 0.16), 4)
        suc = round({"upi": 0.90, "card": 0.84, "netbanking": 0.78, "wallet": 0.93, "emi": 0.71}[m] * rng.uniform(0.97, 1.03), 4)
        base[seg] = {"sessions_per_day": sess, "checkout_rate": chk, "success_rate": suc}
        last[seg] = {"sessions_per_day": int(sess * rng.uniform(0.96, 1.04)),
                     "checkout_rate": round(chk * rng.uniform(0.985, 1.015), 4), "success_rate": round(suc * rng.uniform(0.99, 1.01), 4)}
    # the needle: a mid-volume segment whose success rate collapses
    mid = [s for s in segs if 600 <= base[s]["sessions_per_day"] <= 1600]
    cause = cause or rng.choice(mid)
    last[cause]["success_rate"] = round(base[cause]["success_rate"] * 0.55, 4)
    # distractors: visible but smaller moves, some in checkout rate (the wrong metric)
    pool = [s for s in segs if s != cause]
    distractors = rng.sample(pool, 8 * tiers)
    for i, s in enumerate(distractors):
        if i % 2 == 0:
            last[s]["success_rate"] = round(base[s]["success_rate"] * rng.uniform(0.86, 0.92), 4)
        else:
            last[s]["checkout_rate"] = round(base[s]["checkout_rate"] * rng.uniform(0.80, 0.88), 4)
    return Needle(seed=seed, cause=cause, distractors=distractors, base=base, last=last)


def slices(ds: Needle) -> dict[str, dict]:
    rows = ds.rows()
    per = len(rows) // N_SCOUTS
    out = {}
    for i in range(N_SCOUTS):
        chunk = rows[i * per:(i + 1) * per]
        out[f"scout{i+1}"] = {"slice": f"segments {i*per+1}-{(i+1)*per} of {len(rows)} ({chunk[0]['segment'].split('/')[0]} … {chunk[-1]['segment'].split('/')[0]})",
                              "table": chunk}
    return out


QUESTION = ("Blended checkout conversion fell last week. Exactly ONE segment's payment success rate collapsed and explains "
            "most of the fall; a few other segments drifted by smaller amounts. Name that one segment.")


def labels_text() -> str:
    return ('Answer with the exact segment id from the table (format region/platform/method), or "unclear". '
            "The cause is the segment with the largest relative fall in success_rate (last vs base). "
            "Columns: segment, sessions_per_day, checkout_rate_base, checkout_rate_last, success_rate_base, success_rate_last.")


def labels_text_xl() -> str:
    return labels_text().replace("format region/platform/method", "format region/platform/method/tier")


def render(table: list[dict]) -> str:
    """Compact CSV for the prompt (5x fewer tokens than JSON); the dict table stays the source for number pinning."""
    return "\n".join(f"{r['segment']},{r['sessions_per_day']},{r['checkout_rate_base']},{r['checkout_rate_last']},{r['success_rate_base']},{r['success_rate_last']}" for r in table)


def normalize(cause: str, sl_table=None) -> str:
    c = cause.strip().lower().replace(" ", "")
    return c if c.count("/") in (2, 3) or c == "unclear" else "unclear"


def in_slice(answer: str, table: list[dict]) -> bool:
    return answer == "unclear" or any(r["segment"] == answer for r in table)


def rule_scout(name: str, sl: dict) -> dict:
    """No key: biggest relative success-rate drop in the slice."""
    best, score = None, 0.0
    for r in sl["table"]:
        drop = 1 - r["success_rate_last"] / r["success_rate_base"]
        if drop > score:
            best, score = r, drop
    if best is None or (best["success_rate_last"] / best["success_rate_base"]) > 0.8:
        return {"cause": "unclear", "confidence": 0.2, "note": "no segment in this slice lost a large share of its success rate"}
    rel = round((best["success_rate_last"] / best["success_rate_base"] - 1) * 100, 1)
    return {"cause": best["segment"], "confidence": 0.9, "note": f"{best['segment']} success_rate {best['success_rate_base']} -> {best['success_rate_last']} ({rel}%)"}


def score(answer: str, table: list[dict]) -> float | None:
    """Evidence behind a draft, computed from the scout's own slice: relative fall in success rate of the named segment.
    None if the segment is not in the slice (the supervisor already rejects those)."""
    for r in table:
        if r["segment"] == answer:
            return round(1 - r["success_rate_last"] / r["success_rate_base"], 4)
    return None
