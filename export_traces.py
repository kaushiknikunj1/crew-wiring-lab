#!/usr/bin/env python3
"""Bundle measured runs into one JSON the website's replay page reads.

    python3 export_traces.py --out ../nikunjkaushik.com/tools/crew_wiring_runs.json [--seed 7]

Picks ONE seed and ships its solo / flat / supervised traces (replay time compressed to ~16 s)
plus the summary over every seed in results/. Without --seed it prefers a seed where the wirings
disagree, so the replay shows the difference the numbers describe.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
ORDER = ["solo", "flat", "supervised"]
REPLAY_SECONDS = 16.0


def load() -> dict[int, dict[str, dict]]:
    runs: dict[int, dict[str, dict]] = {}
    for p in sorted(RES.glob("run_*.json")):
        d = json.loads(p.read_text())
        runs.setdefault(d["seed"], {})[d["wiring"]] = d
    return {s: r for s, r in runs.items() if all(w in r for w in ORDER)}


def summary(runs) -> dict:
    out = {}
    for w in ORDER:
        rs = [r[w] for r in runs.values()]
        out[w] = dict(runs=len(rs), accuracy=round(sum(r["correct"] for r in rs) / len(rs), 3),
                      calls=round(statistics.mean(r["calls"] for r in rs), 1),
                      tokens=round(statistics.mean(r["input_tokens"] + r["output_tokens"] for r in rs)),
                      usd=round(statistics.mean(r["cost_usd"] for r in rs), 4),
                      seconds=round(statistics.mean(r["seconds"] for r in rs), 1))
    total = dict(calls=sum(r[w]["calls"] for r in runs.values() for w in ORDER),
                 usd=round(sum(r[w]["cost_usd"] for r in runs.values() for w in ORDER), 4), seeds=len(runs))
    return dict(by_wiring=out, total=total)


def pick(runs) -> int:
    def score(s):
        r = runs[s]
        distinct = len({r[w]["correct"] for w in ORDER})
        retries = r["supervised"]["result"].get("retries", 0)
        return (distinct, 1 if r["supervised"]["correct"] else 0, min(retries, 2), -s)
    return max(runs, key=score)


def compress(trace: list[dict], truth: str, answer: str, calls: int) -> list[dict]:
    """Real seconds -> replay seconds; keep order, add the 5 input attach events the stage expects."""
    inputs = ["brief", "sources", "constraints", "budget", "prior runs"]
    ev = [dict(t=round(0.12 * (i + 1), 2), a=n, b="bus", s=f"bus.attach  {n.replace(' ', '_')}  ctx=shared", c=0.0) for i, n in enumerate(inputs)]
    t0 = 0.12 * len(inputs) + 0.3
    real_end = max(e["t"] for e in trace) or 1.0
    scale = min((REPLAY_SECONDS - t0) / real_end, 1.5)   # long runs squeeze to ~16 s; a 4 s solo run stays short, so the replay keeps the wall-time gap
    last = t0 - 0.2
    for e in trace:
        t = max(t0 + e["t"] * scale, last + 0.18)
        last = t
        s = e["s"]
        if s.startswith("done"):
            s += "  CORRECT" if answer == truth else f"  WRONG  (truth: {truth})"
        ev.append(dict(t=round(t, 2), a=e["a"], b=e["b"], s=s, c=e["c"]))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    runs = load()
    if not runs:
        raise SystemExit("no complete seeds in results/")
    seed = a.seed if a.seed in runs else pick(runs)
    r = runs[seed]
    out = dict(seed=seed, truth=r["solo"]["truth"], headline=r["solo"]["headline"], model=None, summary=summary(runs), wirings={})
    sm = RES / "summary.json"
    if sm.exists():
        tot = json.loads(sm.read_text())["total"]; out["model"] = tot.get("model"); out["thinking"] = tot.get("thinking")
    for w in ORDER:
        x = r[w]; res = x["result"]
        out["wirings"][w] = dict(answer=x["answer"], correct=x["correct"], calls=x["calls"], usd=x["cost_usd"], seconds=x["seconds"],
                                 tokens=x["input_tokens"] + x["output_tokens"], retries=res.get("retries", 0), votes=res.get("votes", {}),
                                 text=res.get("text") or res.get("note", ""), trace=compress(res["trace"], x["truth"], x["answer"], x["calls"]))
    pathlib.Path(a.out).write_text(json.dumps(out, indent=0, separators=(",", ":")))
    s = out["summary"]["by_wiring"]
    print(f"seed {seed} truth={out['truth']}  " + "  ".join(f"{w}:{'OK' if r[w]['correct'] else 'MISS'}" for w in ORDER))
    print("  ".join(f"{w} acc={s[w]['accuracy']:.0%} ${s[w]['usd']:.4f}" for w in ORDER), f"| total ${out['summary']['total']['usd']} over {out['summary']['total']['seeds']} seeds")


if __name__ == "__main__":
    main()
