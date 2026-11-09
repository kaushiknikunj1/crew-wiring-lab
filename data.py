"""Synthetic checkout funnel with ONE planted cause per seed.

28 days x (platform x payment_method x traffic_source) cells. Days 1-21 are the baseline,
days 22-28 are "last week". Exactly one cause from CAUSES is planted in last week; the
question every wiring must answer is "why did blended conversion fall?".

Everything is deterministic in the seed, so a run can be replayed exactly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

PLATFORMS = ["android", "ios", "web"]
METHODS = ["upi", "card", "netbanking", "wallet"]
SOURCES = ["organic", "paid", "referral"]
DAYS, BASE_DAYS = 28, 21

# label -> (where it shows, what it does). The scouts only get slices; some slices cannot see some causes.
CAUSES = {
    "checkout_latency_android": "checkout-start rate fell on Android only (slow checkout page after a release)",
    "upi_gateway_outage":       "payment success rate fell for UPI only (gateway degraded)",
    "paid_traffic_mix":         "paid sessions surged with low-intent traffic: paid checkout-start rate fell, organic and referral rates held",
    "pricing_change":           "checkout-start rate fell on every platform and source (price shown earlier)",
    "ios_release_bug":          "payment-attempt rate fell on iOS only (pay button broke in the new build)",
    "card_bin_block":           "payment success rate fell for cards only (issuer BIN range blocked)",
}


@dataclass
class Dataset:
    seed: int
    cause: str
    rows: list = field(default_factory=list)   # dicts: day, platform, method, source, sessions, checkout, attempt, success

    # ---- aggregations the scouts and the solo agent read ---------------------------------------
    def rates(self, by: str | None, period: str) -> dict:
        """Funnel rates per segment of `by` (or overall) for 'base' or 'last'."""
        lo, hi = (1, BASE_DAYS) if period == "base" else (BASE_DAYS + 1, DAYS)
        acc: dict[str, dict] = {}
        for r in self.rows:
            if not (lo <= r["day"] <= hi):
                continue
            k = r[by] if by else "all"
            a = acc.setdefault(k, {"sessions": 0, "checkout": 0, "attempt": 0, "success": 0})
            for f in a:
                a[f] += r[f]
        out = {}
        n_days = hi - lo + 1
        for k, a in acc.items():
            out[k] = {
                "sessions_per_day": round(a["sessions"] / n_days),
                "checkout_rate": round(a["checkout"] / max(a["sessions"], 1), 4),
                "attempt_rate": round(a["attempt"] / max(a["checkout"], 1), 4),
                "success_rate": round(a["success"] / max(a["attempt"], 1), 4),
                "conversion": round(a["success"] / max(a["sessions"], 1), 4),
            }
        return out

    def daily(self) -> list[dict]:
        acc: dict[int, dict] = {}
        for r in self.rows:
            a = acc.setdefault(r["day"], {"sessions": 0, "success": 0})
            a["sessions"] += r["sessions"]; a["success"] += r["success"]
        return [{"day": d, "conversion": round(a["success"] / max(a["sessions"], 1), 4)} for d, a in sorted(acc.items())]

    def headline(self) -> dict:
        b, l = self.rates(None, "base")["all"], self.rates(None, "last")["all"]
        return {"baseline_conversion": b["conversion"], "last_week_conversion": l["conversion"],
                "relative_change_pct": round((l["conversion"] / b["conversion"] - 1) * 100, 1)}


def generate(seed: int, cause: str | None = None) -> Dataset:
    rng = random.Random(seed)
    cause = cause or rng.choice(sorted(CAUSES))
    # base per-segment propensities
    p_sess = {"android": 0.46, "ios": 0.22, "web": 0.32}
    m_share = {"upi": 0.48, "card": 0.27, "netbanking": 0.13, "wallet": 0.12}
    s_share = {"organic": 0.52, "paid": 0.30, "referral": 0.18}
    checkout = {"android": 0.118, "ios": 0.141, "web": 0.097}
    intent = {"organic": 1.0, "paid": 0.78, "referral": 1.12}
    attempt = {"android": 0.71, "ios": 0.74, "web": 0.69}
    success = {"upi": 0.906, "card": 0.842, "netbanking": 0.788, "wallet": 0.931}
    total_sessions = 38000 + rng.randint(-2500, 2500)

    ds = Dataset(seed=seed, cause=cause)
    for day in range(1, DAYS + 1):
        last = day > BASE_DAYS
        dow = 1.0 + 0.06 * ((day % 7) in (5, 6))          # weekend bump
        noise = lambda: rng.uniform(0.97, 1.03)
        for pf in PLATFORMS:
            for me in METHODS:
                for so in SOURCES:
                    share = p_sess[pf] * m_share[me] * s_share[so]
                    sess = total_sessions * share * dow * noise()
                    c_rate, a_rate, s_rate = checkout[pf] * intent[so], attempt[pf], success[me]
                    if last:
                        if cause == "checkout_latency_android" and pf == "android":
                            c_rate *= 0.80
                        elif cause == "upi_gateway_outage" and me == "upi":
                            s_rate *= 0.74
                        elif cause == "paid_traffic_mix" and so == "paid":
                            sess *= 1.65; c_rate *= 0.62
                        elif cause == "pricing_change":
                            c_rate *= 0.89
                        elif cause == "ios_release_bug" and pf == "ios":
                            a_rate *= 0.68
                        elif cause == "card_bin_block" and me == "card":
                            s_rate *= 0.64
                    sess_i = int(sess)
                    chk = int(sess_i * c_rate * noise())
                    att = int(chk * a_rate * noise())
                    suc = int(att * min(s_rate * noise(), 0.995))
                    ds.rows.append(dict(day=day, platform=pf, method=me, source=so,
                                        sessions=sess_i, checkout=chk, attempt=att, success=suc))
    return ds


# ---- the six scout slices (what each worker is allowed to see) ------------------------------------
def slices(ds: Dataset) -> dict[str, dict]:
    def pair(by):
        b, l = ds.rates(by, "base"), ds.rates(by, "last")
        return {k: {"baseline": b[k], "last_week": l[k],
                    "delta_pct": {f: round((l[k][f] / b[k][f] - 1) * 100, 1) if b[k][f] else 0.0
                                  for f in ("sessions_per_day", "checkout_rate", "attempt_rate", "success_rate", "conversion")}}
                for k in b}
    return {
        "scout1": {"slice": "by platform", "table": pair("platform")},
        "scout2": {"slice": "by payment method", "table": pair("method")},
        "scout3": {"slice": "by traffic source", "table": pair("source")},
        "scout4": {"slice": "overall funnel steps", "table": pair(None)},
        "scout5": {"slice": "daily blended conversion", "table": {"daily": ds.daily()}},
        "scout6": {"slice": "headline", "table": ds.headline()},
    }
