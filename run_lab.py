#!/usr/bin/env python3
"""Run every wiring on the same seeds and score them against the planted cause.

    python3 run_lab.py --seeds 12            # results/ gets one JSON per run + summary.json + summary.md
    python3 run_lab.py --seeds 3 --wirings solo,supervised
    LLM_PROVIDER=none python3 run_lab.py     # deterministic scouts, no API calls

Accuracy = answer == planted cause. Cost = tokens actually billed (from the API's usage metadata).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import llm
from crew import WIRINGS
from data import generate

OUT = pathlib.Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--wirings", default="solo,flat,supervised")
    ap.add_argument("--only", help="comma-separated seeds to (re)run instead of a range; summary is rebuilt from every run in results/")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    wirings = [w.strip() for w in a.wirings.split(",")]
    rows = []
    print(f"provider={llm.provider()} model={llm.GEMINI_MODEL if llm.provider()=='gemini' else '-'} thinking={llm.GEMINI_THINKING}")
    seeds = [int(x) for x in a.only.split(",")] if a.only else list(range(a.start, a.start + a.seeds))
    for seed in seeds:
        ds = generate(seed)
        for w in wirings:
            u0 = dict(llm.USAGE); t0 = time.perf_counter()
            res = WIRINGS[w](ds)
            used = {k: llm.USAGE[k] - u0[k] for k in llm.USAGE}
            row = dict(seed=seed, truth=ds.cause, wiring=w, answer=res["answer"], correct=res["answer"] == ds.cause,
                       calls=used["calls"], input_tokens=used["input_tokens"], output_tokens=used["output_tokens"] + used["thinking_tokens"],
                       cost_usd=round(llm.usage_cost_usd(used), 5), seconds=round(time.perf_counter() - t0, 1))
            rows.append(row)
            (OUT / f"run_{seed:03d}_{w}.json").write_text(json.dumps(dict(row, headline=ds.headline(), result=res), indent=1))
            print(f"seed {seed:3d}  {w:<10} truth={ds.cause:<26} answer={res['answer']:<26} {'OK ' if row['correct'] else 'MISS'} calls={row['calls']:2d} ${row['cost_usd']:.4f} {row['seconds']}s")
    rows = [{k: v for k, v in json.loads(p.read_text()).items() if k not in ("headline", "result")} for p in sorted(OUT.glob("run_*.json"))]
    summary = {}
    for w in wirings:
        rs = [r for r in rows if r["wiring"] == w]
        summary[w] = dict(runs=len(rs), accuracy=round(sum(r["correct"] for r in rs) / len(rs), 3),
                          calls_per_run=round(statistics.mean(r["calls"] for r in rs), 1),
                          tokens_per_run=round(statistics.mean(r["input_tokens"] + r["output_tokens"] for r in rs)),
                          cost_per_run_usd=round(statistics.mean(r["cost_usd"] for r in rs), 5),
                          seconds_per_run=round(statistics.mean(r["seconds"] for r in rs), 1))
    total = dict(llm.USAGE, cost_usd=round(llm.usage_cost_usd(), 4), provider=llm.provider(), model=llm.GEMINI_MODEL, thinking=llm.GEMINI_THINKING)
    (OUT / "summary.json").write_text(json.dumps(dict(summary=summary, rows=rows, total=total), indent=1))
    md = ["| wiring | runs | accuracy | calls/run | tokens/run | $/run | s/run |", "|---|---|---|---|---|---|---|"]
    for w, s in summary.items():
        md.append(f"| {w} | {s['runs']} | {s['accuracy']:.0%} | {s['calls_per_run']} | {s['tokens_per_run']} | {s['cost_per_run_usd']:.4f} | {s['seconds_per_run']} |")
    md.append(f"\nTotal: {total['calls']} calls, {total['input_tokens']} in / {total['output_tokens']} out / {total['thinking_tokens']} thinking tokens, ${total['cost_usd']} on {total['model']} (thinking budget {total['thinking']}).")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
