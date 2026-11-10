"""Three wirings for the same question, same model, same tools, same budget.

  solo        one agent reads every slice and answers once
  flat        six scouts, each on one slice, run IN SEQUENCE and READ EACH OTHER'S DRAFTS; majority vote; writer
  supervised  six scouts, each sees ONLY its slice, run IN PARALLEL (isolation makes that possible); a deterministic
              supervisor checks every draft against the evidence (numbers must come from the slice, the answer must be
              something the slice can show) and sends a failing scout back once; confidence-weighted vote; writer

The task is pluggable (see tasks.py): task A "planted cause" (six labels), task B "needle" (240 segments).
Every step is appended to a trace: who sent what to whom, tokens, cost, elapsed time.
With no API key the scouts fall back to a deterministic rule, so the whole lab runs end to end without a model.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import llm
import tasks

TASK = tasks.PLANTED          # default; run_lab.py switches with set_task()
MERGE = __import__("os").environ.get("MERGE", "evidence")   # supervised merge: "evidence" (v2) or "confidence" (v1)


def set_task(t):
    global TASK
    TASK = t


SCOUT_SYS = ("You are one analyst in a crew. You see ONE slice of the data. {q} Reply with JSON only: "
             '{{"cause": <answer>, "confidence": <0..1>, "note": <one sentence citing numbers from your slice>}}. '
             "If your slice cannot settle it, answer \"unclear\" with low confidence. Never invent numbers.")
SOLO_SYS = ("You are a senior analyst. You see every slice of the data. {q} Reply with JSON only: "
            '{{"cause": <answer>, "confidence": <0..1>, "note": <two sentences citing numbers from the tables>}}. Never invent numbers.')
WRITER_SYS = ("Write the crew's two-sentence answer to the question. Use only the numbers given, name the cause in plain words, "
              "and do not mention votes, scores or weights. Plain prose, no JSON.")


class Trace:
    def __init__(self):
        self.t0 = time.perf_counter(); self.events = []

    def add(self, a: str, b: str, s: str, usage_before: dict | None = None, usage_after: dict | None = None):
        c = 0.0
        if usage_before is not None:
            after = usage_after or llm.USAGE
            c = llm.usage_cost_usd({k: after[k] - usage_before[k] for k in llm.USAGE})
        self.events.append(dict(t=round(time.perf_counter() - self.t0, 2), a=a, b=b, s=s, c=round(c, 5)))


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
    cause = TASK.normalize(str(d.get("cause", "unclear")))
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.3))))
    except (TypeError, ValueError):
        conf = 0.3
    return {"cause": cause, "confidence": conf, "note": str(d.get("note", ""))[:300]}


def _slice_text(sl: dict) -> str:
    return TASK.render(sl["table"])


def ask_scout(name: str, sl: dict, drafts: list[str] | None = None, warning: str | None = None) -> dict:
    user = f"{TASK.labels_text()}\nYour slice: {sl['slice']}\n{_slice_text(sl)}"
    if drafts:
        user += "\n\nDrafts from teammates so far:\n" + "\n".join(drafts)
    if warning:
        user += f"\n\nSupervisor: {warning} Answer again."
    text = llm.complete(SCOUT_SYS.format(q=TASK.question), user, max_tokens=420)
    return _parse(text) if text is not None else TASK.rule_scout(name, sl)


def unpinned(note: str, table) -> set[str]:
    """Numbers in the note that are not in the table. Sign-insensitive: 'dropped 20.1%' matches a -20.1 delta."""
    allowed = {n.lstrip("-") for n in llm.allowed_numbers(table)}
    return {n for n in llm.numbers_in(note) if n.lstrip("-") not in allowed}


def supervisor_check(draft: dict, sl: dict) -> str | None:
    """Deterministic: numbers must exist in the slice; the answer must be something this slice can show."""
    if not TASK.in_slice(draft["cause"], sl["table"]):
        return f"'{draft['cause']}' is not in your slice; you can only name what you can see."
    bad = unpinned(draft["note"], sl["table"]) if draft["note"] else set()
    if bad:
        return f"these numbers are not in your slice: {', '.join(sorted(bad)[:4])}."
    if draft["cause"] != "unclear" and draft["confidence"] >= 0.5 and not draft["note"]:
        return "a confident cause needs the numbers that show it."
    return None


def vote(drafts: dict[str, dict], weighted: bool, sl: dict | None = None) -> tuple[str, dict]:
    """flat: one scout one vote. supervised: rank by the evidence each draft cites (TASK.score on the scout's own slice),
    falling back to the scout's confidence when the task has no such number (MERGE=confidence forces the old behaviour)."""
    tally: dict[str, float] = {}
    for n, d in drafts.items():
        if d["cause"] == "unclear":
            continue
        w = 1.0
        if weighted:
            ev = TASK.score(d["cause"], sl[n]["table"]) if (sl and MERGE == "evidence") else None
            w = ev if ev is not None else d["confidence"]
        tally[d["cause"]] = tally.get(d["cause"], 0) + w
    if not tally:
        return "unclear", {}
    best = max(tally.items(), key=lambda kv: kv[1])[0]
    return best, {k: round(v, 3) for k, v in tally.items()}


def write(answer: str, tally: dict, headline: dict) -> str:
    user = f"question: {TASK.question}\ncause: {answer}\nvotes: {json.dumps(tally)}\nheadline: {json.dumps(headline)}"
    text = llm.complete(WRITER_SYS, user, max_tokens=320)
    return text if text is not None else f"Conversion fell {headline['relative_change_pct']}% week on week; the crew's answer is {answer}."


# ---- the three wirings --------------------------------------------------------------------------
def run_solo(ds) -> dict:
    tr = Trace(); sl = TASK.slices(ds)
    tr.add("bus", "agent", f"agent      reads bus  {len(sl)} slices")
    u0 = dict(llm.USAGE)
    user = TASK.labels_text() + "\n" + "\n".join(f"{v['slice']}:\n{_slice_text(v)}" for v in sl.values())
    text = llm.complete(SOLO_SYS.format(q=TASK.question), user, max_tokens=520)
    d = _parse(text) if text is not None else max((TASK.rule_scout(n, sl[n]) for n in sl), key=lambda x: x["confidence"])
    tr.add("agent", "writer", f"agent      answer  cause={d['cause']}  conf {d['confidence']:.2f}", u0)
    tr.add("writer", "verify", "verify     numbers pinned  " + ("ok" if not unpinned(d["note"], [v["table"] for v in sl.values()]) else "WARN unpinned"))
    tr.add("verify", "bus", f"done       cause={d['cause']}")
    return dict(wiring="solo", answer=d["cause"], confidence=d["confidence"], note=d["note"], trace=tr.events)


def run_flat(ds) -> dict:
    tr = Trace(); sl = TASK.slices(ds); drafts: dict[str, dict] = {}; shared: list[str] = []
    tr.add("bus", "planner", f"planner    reads bus  {len(sl)} slices"); tr.add("planner", "decompose", f"decompose  {len(sl)} sub-questions"); tr.add("decompose", "route", f"route      {len(sl)} scouts  sequential  shared drafts ON")
    for n in sl:
        tr.add("route", n, f"dispatch   {n}  {sl[n]['slice']}")
    for n in sl:
        u0 = dict(llm.USAGE)
        tr.add(n, "bus", f"{n:<10} reads bus  {len(shared)} teammate drafts visible")
        d = ask_scout(n, sl[n], drafts=shared); drafts[n] = d
        shared.append(f"{n}: cause={d['cause']} conf={d['confidence']:.2f} note={d['note'][:120]}")
        tr.add(n, "merge", f"{n:<10} returned  cause={d['cause']}  conf {d['confidence']:.2f}", u0)
    answer, tally = vote(drafts, weighted=False)
    tr.add("merge", "dedupe", f"merge      {len(sl)} drafts  votes {json.dumps(tally)}"); tr.add("dedupe", "rank", f"rank       top = {answer}")
    tr.add("rank", "cite", "cite       no check in this wiring")
    u0 = dict(llm.USAGE); text = write(answer, tally, ds.headline()); tr.add("cite", "writer", f"writer     draft  cause={answer}", u0)
    tr.add("writer", "verify", "verify     skipped"); tr.add("verify", "bus", f"done       cause={answer}")
    return dict(wiring="flat", answer=answer, votes=tally, drafts=drafts, text=text, trace=tr.events)


def run_supervised(ds) -> dict:
    tr = Trace(); sl = TASK.slices(ds); drafts: dict[str, dict] = {}; retries = 0
    tr.add("bus", "planner", f"planner    reads bus  {len(sl)} slices"); tr.add("planner", "decompose", f"decompose  {len(sl)} sub-questions"); tr.add("decompose", "route", f"route      {len(sl)} scouts  parallel  isolated")
    for n in sl:
        tr.add("route", n, f"dispatch   {n}  {sl[n]['slice']}")

    def one(n):
        """A scout plus its supervisor loop, run on its own thread. Usage deltas are measured per call."""
        local = []
        u0 = dict(llm.USAGE); d = ask_scout(n, sl[n]); u1 = dict(llm.USAGE)
        local.append((n, "super", f"{n:<10} returned  slice only  cause={d['cause']}  conf {d['confidence']:.2f}", u0, u1))
        w = supervisor_check(d, sl[n]); r = 0
        if w:
            r = 1; local.append(("super", n, f"supervisor {n} flagged  {w}", None, None))
            u2 = dict(llm.USAGE); d = ask_scout(n, sl[n], warning=w); u3 = dict(llm.USAGE)
            local.append((n, "super", f"{n:<10} retry  cause={d['cause']}  conf {d['confidence']:.2f}", u2, u3))
            if supervisor_check(d, sl[n]):
                d = {"cause": "unclear", "confidence": 0.0, "note": ""}; local.append(("super", n, f"supervisor {n} dropped  still unpinned", None, None))
        return n, d, r, local

    with ThreadPoolExecutor(max_workers=len(sl)) as ex:
        for n, d, r, local in ex.map(one, list(sl)):
            drafts[n] = d; retries += r
            for a, b, s, u0, u1 in local:
                tr.add(a, b, s, u0, u1)
    tr.add("super", "merge", f"supervisor {len(sl)} checked  {retries} retried")
    answer, tally = vote(drafts, weighted=True, sl=sl)
    tr.add("merge", "dedupe", f"merge      {len(sl)} drafts  independent"); tr.add("dedupe", "rank", f"rank       by {MERGE}  top = {answer}  {json.dumps(tally)}")
    tr.add("rank", "cite", "cite       every kept draft is pinned to its slice")
    u0 = dict(llm.USAGE); text = write(answer, tally, ds.headline()); tr.add("cite", "writer", f"writer     draft  cause={answer}", u0)
    tr.add("writer", "verify", "verify     numbers pinned  ok"); tr.add("verify", "bus", f"done       cause={answer}")
    return dict(wiring="supervised", answer=answer, votes=tally, drafts=drafts, retries=retries, text=text, trace=tr.events)


WIRINGS = {"solo": run_solo, "flat": run_flat, "supervised": run_supervised}
