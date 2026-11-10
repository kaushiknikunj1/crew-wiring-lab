"""Deterministic no-key scout for task A (planted cause): biggest relative drop in the slice."""
from __future__ import annotations


def planted_rule_scout(name: str, sl: dict) -> dict:
    t = sl["table"]
    if name == "scout1":
        d = {k: v["delta_pct"] for k, v in t.items()}
        if d["android"]["checkout_rate"] < -10 and d["ios"]["checkout_rate"] > -5:
            return {"cause": "checkout_latency_android", "confidence": 0.8, "note": f"android checkout_rate {d['android']['checkout_rate']}% vs ios {d['ios']['checkout_rate']}%"}
        if d["ios"]["attempt_rate"] < -15:
            return {"cause": "ios_release_bug", "confidence": 0.8, "note": f"ios attempt_rate {d['ios']['attempt_rate']}%"}
        if all(d[p]["checkout_rate"] < -6 for p in d):
            return {"cause": "pricing_change", "confidence": 0.4, "note": f"checkout_rate down on all platforms: {d['android']['checkout_rate']}%, {d['ios']['checkout_rate']}%, {d['web']['checkout_rate']}%"}
        return {"cause": "unclear", "confidence": 0.2, "note": "platform rates moved together"}
    if name == "scout2":
        d = {k: v["delta_pct"] for k, v in t.items()}
        if d["upi"]["success_rate"] < -12:
            return {"cause": "upi_gateway_outage", "confidence": 0.85, "note": f"upi success_rate {d['upi']['success_rate']}%"}
        if d["card"]["success_rate"] < -12:
            return {"cause": "card_bin_block", "confidence": 0.85, "note": f"card success_rate {d['card']['success_rate']}%"}
        return {"cause": "unclear", "confidence": 0.2, "note": "success rates flat by method"}
    if name == "scout3":
        d = {k: v["delta_pct"] for k, v in t.items()}
        if d["paid"]["sessions_per_day"] > 30 and d["paid"]["checkout_rate"] < -20:
            return {"cause": "paid_traffic_mix", "confidence": 0.85, "note": f"paid sessions_per_day {d['paid']['sessions_per_day']}%, paid checkout_rate {d['paid']['checkout_rate']}%"}
        if all(d[s]["checkout_rate"] < -6 for s in d):
            return {"cause": "pricing_change", "confidence": 0.3, "note": "checkout_rate down on every source"}
        return {"cause": "unclear", "confidence": 0.2, "note": "source mix stable"}
    if name == "scout4":
        d = t["all"]["delta_pct"]
        worst = min(("checkout_rate", "attempt_rate", "success_rate"), key=lambda f: d[f])
        guess = {"checkout_rate": "pricing_change", "attempt_rate": "ios_release_bug", "success_rate": "upi_gateway_outage"}[worst]
        return {"cause": guess, "confidence": 0.25, "note": f"{worst} moved {d[worst]}% overall"}
    # scout5/6 see only blended numbers: they cannot know
    return {"cause": "unclear", "confidence": 0.1, "note": "blended series only"}


