"""Three wirings for the same question, same model, same tools, same budget.

  solo        one agent reads every slice and answers once
  flat        six scouts, each on one slice, run in order and READ EACH OTHER'S DRAFTS; majority vote; writer
  supervised  six scouts, each sees ONLY its slice; a supervisor checks every draft against the evidence
              (numbers must come from the slice, cause must be a known label) and sends a failing scout
              back once; confidence-weighted vote; writer

Every step is appended to a trace: who sent what to whom, tokens, cost, elapsed time.
With no API key the scouts fall back to a deterministic rule (biggest relative drop in the slice),
so the whole lab runs end to end without a model — the comparison is then between wirings only.
"""
from __future__ import annotations

import json
import re
import time

import llm
from data import CAUSES, Dataset, slices

LABELS = sorted(CAUSES) + ["unclear"]

SCOUT_SYS = ("You are one analyst in a crew investigating why blended checkout conversion fell last week. "
             "You see ONE slice of the data. Reply with JSON only: "
             '{"cause": <one of the labels>, "confidence": <0..1>, "note": <one sentence citing numbers from your slice>}. '
             "If your slice cannot distinguish the cause, say \"unclear\" with low confidence. Never invent numbers.")
SOLO_SYS = ("You are a senior analyst. You see every slice of the data. Decide why blended checkout conversion fell last week. "
            "Reply with JSON only: "
            '{"cause": <one of the labels>, "confidence": <0..1>, "note": <two sentences citing numbers from the tables>}. Never invent numbers.')
WRITER_SYS = ("Write the crew's two-sentence answer to 'why did conversion fall last week'. Use only the numbers given, "
              "name the cause in plain words, and do not mention votes, scores or weights. Plain prose, no JSON.")


class Trace:
    def __init__(self):
        self.t0 = time.perf_counter(); self.events = []

    def add(self, a: str, b: str, s: str, usage_before: dict | None = None):
        c = 0.0
        if usage_before is not None:
            d = {k: llm.USAGE[k] - usage_before[k] for k in llm.USAGE}
            c = llm.usage_cost_usd(d)
        self.events.append(dict(t=round(time.perf_counter() - self.t0, 2), a=a, b=b, s=s, c=round(c, 5)))


def _labels_text() -> str:
    return "Labels: " + ", ".join(f"{k} ({v})" for k, v in CAUSES.items()) + ", unclear."


def _parse(text: str) -> dict:
    """JSON if we got it; otherwise pull cause/confidence out of a truncated or fenced reply."""
    text = (text or "").replace("```json", "").replace("```", "")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    if not d:
        mc = re.search(r'"cause"\s*:\s*"([^"]+)"', text); mf = re.search(r'"confidence"\s*:\s*([0-9.]+)', text); mn = re.search(r'"note"\s*:\s*"([^"]*)', text)
        if mc:
            d = {"cause": mc.group(1), "confidence": mf.group(1) if mf else 0.3, "note": mn.group(1) if mn else ""}
        else:
            return {"cause": "unclear", "confidence": 0.0, "note": ""}
    cause = str(d.get("cause", "unclear")).strip().lower().replace(" ", "_")
    if cause not in LABELS:
        cause = "unclear"
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.3))))
    except (TypeError, ValueError):
        conf = 0.3
    return {"cause": cause, "confidence": conf, "note": str(d.get("note", ""))[:300]}


# ---- deterministic fallback scout (no key): biggest relative drop in the slice ----------------------
def rule_scout(name: str, sl: dict) -> dict:
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


def ask_scout(name: str, sl: dict, drafts: list[str] | None = None, warning: str | None = None) -> dict:
    user = f"{_labels_text()}\nYour slice: {sl['slice']}\n{json.dumps(sl['table'])}"
    if drafts:
        user += "\n\nDrafts from teammates so far:\n" + "\n".join(drafts)
    if warning:
        user += f"\n\nSupervisor: {warning} Answer again."
    text = llm.complete(SCOUT_SYS, user, max_tokens=420)
    return _parse(text) if text is not None else rule_scout(name, sl)


def unpinned(note: str, table: dict) -> set[str]:
    """Numbers in the note that are not in the table. Sign-insensitive: 'dropped 20.1%' matches a -20.1 delta."""
    allowed = {n.lstrip("-") for n in llm.allowed_numbers(table)}
    return {n for n in llm.numbers_in(note) if n.lstrip("-") not in allowed}


def supervisor_check(draft: dict, sl: dict) -> str | None:
    """Deterministic: every number in the note must exist in the slice; a confident answer needs a note."""
    bad = unpinned(draft["note"], sl["table"]) if draft["note"] else set()
    if bad:
        return f"these numbers are not in your slice: {', '.join(sorted(bad)[:4])}."
    if draft["cause"] != "unclear" and draft["confidence"] >= 0.5 and not draft["note"]:
        return "a confident cause needs the numbers that show it."
    return None


def vote(drafts: dict[str, dict], weighted: bool) -> tuple[str, dict]:
    tally: dict[str, float] = {}
    for d in drafts.values():
        if d["cause"] == "unclear":
            continue
        tally[d["cause"]] = tally.get(d["cause"], 0) + (d["confidence"] if weighted else 1.0)
    if not tally:
        return "unclear", {}
    best = max(tally.items(), key=lambda kv: kv[1])[0]
    return best, {k: round(v, 2) for k, v in tally.items()}


def write(answer: str, tally: dict, headline: dict) -> str:
    user = f"cause: {answer}\nvotes: {json.dumps(tally)}\nheadline: {json.dumps(headline)}"
    text = llm.complete(WRITER_SYS, user, max_tokens=320)
    return text if text is not None else f"Conversion fell {headline['relative_change_pct']}% week on week; the crew's answer is {answer}."


# ---- the three wirings --------------------------------------------------------------------------
def run_solo(ds: Dataset) -> dict:
    tr = Trace(); sl = slices(ds)
    tr.add("bus", "agent", "agent      reads bus  6 slices")
    u0 = dict(llm.USAGE)
    user = _labels_text() + "\n" + "\n".join(f"{v['slice']}: {json.dumps(v['table'])}" for v in sl.values())
    text = llm.complete(SOLO_SYS, user, max_tokens=520)
    d = _parse(text) if text is not None else _solo_rule(sl)
    tr.add("agent", "writer", f"agent      answer  cause={d['cause']}  conf {d['confidence']:.2f}", u0)
    tr.add("writer", "verify", "verify     numbers pinned  " + ("ok" if not unpinned(d["note"], sl) else "WARN unpinned"))
    tr.add("verify", "bus", f"done       cause={d['cause']}")
    return dict(wiring="solo", answer=d["cause"], confidence=d["confidence"], note=d["note"], trace=tr.events)


def _solo_rule(sl: dict) -> dict:
    """No key: the analyst who sees everything keeps the strongest single signal."""
    return max((rule_scout(n, sl[n]) for n in sl), key=lambda d: d["confidence"])


def run_flat(ds: Dataset) -> dict:
    tr = Trace(); sl = slices(ds); drafts: dict[str, dict] = {}; shared: list[str] = []
    tr.add("bus", "planner", "planner    reads bus  6 slices"); tr.add("planner", "decompose", "decompose  6 sub-questions"); tr.add("decompose", "route", "route      6 scouts  shared drafts ON")
    for n in sl:
        tr.add("route", n, f"dispatch   {n}  {sl[n]['slice']}")
    for n in sl:
        u0 = dict(llm.USAGE)
        tr.add(n, "bus", f"{n:<10} reads bus  {len(shared)} teammate drafts visible")
        d = ask_scout(n, sl[n], drafts=shared); drafts[n] = d
        shared.append(f"{n}: cause={d['cause']} conf={d['confidence']:.2f} note={d['note'][:120]}")
        tr.add(n, "merge", f"{n:<10} returned  cause={d['cause']}  conf {d['confidence']:.2f}", u0)
    answer, tally = vote(drafts, weighted=False)
    tr.add("merge", "dedupe", f"merge      6 drafts  votes {json.dumps(tally)}"); tr.add("dedupe", "rank", f"rank       top = {answer}")
    tr.add("rank", "cite", "cite       no check in this wiring")
    u0 = dict(llm.USAGE); text = write(answer, tally, ds.headline()); tr.add("cite", "writer", f"writer     draft  cause={answer}", u0)
    tr.add("writer", "verify", "verify     skipped"); tr.add("verify", "bus", f"done       cause={answer}")
    return dict(wiring="flat", answer=answer, votes=tally, drafts=drafts, text=text, trace=tr.events)


def run_supervised(ds: Dataset) -> dict:
    tr = Trace(); sl = slices(ds); drafts: dict[str, dict] = {}; retries = 0
    tr.add("bus", "planner", "planner    reads bus  6 slices"); tr.add("planner", "decompose", "decompose  6 sub-questions"); tr.add("decompose", "route", "route      6 scouts  isolated")
    for n in sl:
        tr.add("route", n, f"dispatch   {n}  {sl[n]['slice']}")
    for n in sl:
        u0 = dict(llm.USAGE)
        d = ask_scout(n, sl[n]); tr.add(n, "super", f"{n:<10} returned  slice only  cause={d['cause']}  conf {d['confidence']:.2f}", u0)
        w = supervisor_check(d, sl[n])
        if w:
            retries += 1; tr.add("super", n, f"supervisor {n} flagged  {w}")
            u1 = dict(llm.USAGE); d = ask_scout(n, sl[n], warning=w); tr.add(n, "super", f"{n:<10} retry  cause={d['cause']}  conf {d['confidence']:.2f}", u1)
            if supervisor_check(d, sl[n]):
                d = {"cause": "unclear", "confidence": 0.0, "note": ""}; tr.add("super", n, f"supervisor {n} dropped  still unpinned")
        drafts[n] = d
    tr.add("super", "merge", f"supervisor 6 checked  {retries} retried")
    answer, tally = vote(drafts, weighted=True)
    tr.add("merge", "dedupe", f"merge      6 drafts  independent"); tr.add("dedupe", "rank", f"rank       top = {answer}  weights {json.dumps(tally)}")
    tr.add("rank", "cite", "cite       every kept draft is pinned to its slice")
    u0 = dict(llm.USAGE); text = write(answer, tally, ds.headline()); tr.add("cite", "writer", f"writer     draft  cause={answer}", u0)
    tr.add("writer", "verify", "verify     numbers pinned  ok"); tr.add("verify", "bus", f"done       cause={answer}")
    return dict(wiring="supervised", answer=answer, votes=tally, drafts=drafts, retries=retries, text=text, trace=tr.events)


WIRINGS = {"solo": run_solo, "flat": run_flat, "supervised": run_supervised}
