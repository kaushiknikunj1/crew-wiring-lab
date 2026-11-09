# Crew Wiring Lab

Same question, same model, same tools, same data. Only the wiring between the agents changes.

The question is one every growth analyst gets on a Monday: **why did checkout conversion fall last week?**
A synthetic 28-day funnel (platform × payment method × traffic source) gets exactly one cause planted
in the last seven days, drawn from six candidates. Each wiring has to name it.

| wiring | what it is |
|---|---|
| **solo** | one agent reads all six slices and answers once |
| **flat crew** | six scouts, one slice each, run in sequence and **read each other's drafts**; majority vote; no check |
| **supervised crew** | six scouts, one slice each, **isolated**; a deterministic supervisor checks every number in a draft against that scout's slice and sends a failing scout back once; confidence-weighted vote |

The replay of one seed, with the run log, is on my site: <https://nikunjkaushik.com/crew_wiring.html>

## Results

| wiring | seeds | accuracy | LLM calls / run | tokens / run | $ / run | seconds / run |
|---|---|---|---|---|---|---|
| solo | 12 | 100% | 1 | 3,027 | 0.0027 | 3.8 |
| flat | 12 | 100% | 7 | 6,047 | 0.0082 | 30.9 |
| supervised | 12 | 100% | 7.1 | 5,524 | 0.0083 | 30.1 |

Model `gemini-3.8-flash`, thinking budget 0, temperature 0.2. The runs in `results/`: 181 calls, $0.23. With the first pass and the debugging calls, the whole project cost about $0.32 in API credit.

Yes: three identical accuracy columns. Twelve seeds, six causes, and every wiring named the planted
cause every time. The crews paid 3× the money and 8× the wall-clock for the same answer.

### The run before this one

The first full pass scored **solo 75 %, flat 75 %, supervised 83 %** (12 seeds, $0.22). All nine misses
were on one cause, `paid_traffic_mix`, and the reason was mine: the label text handed to the agents said
"every per-segment rate held" while the generator also cut paid checkout rate by 38 %. The agents were
reading a contradiction. How each wiring handled it is the most useful thing in this repo:

| wiring | seeds 9 · 11 · 12 (truth: paid_traffic_mix) | what happened |
|---|---|---|
| solo | unclear · unclear · unclear | saw all slices, noticed the contradiction, declined to pick |
| flat crew | pricing_change · pricing_change · pricing_change | scouts whose slice cannot tell (headline, daily, funnel) read the first draft and voted with it: 4–1, 2–1, 4–1 |
| supervised crew | pricing_change · paid_traffic_mix · pricing_change | isolated scouts said "unclear" instead of copying; the one slice that could see it won once on confidence |

Fixing one sentence in the label took every wiring to 100 %. The per-seed lines of that first pass are
in `results/first_pass.log`.

Accuracy is `answer == planted cause`. Cost is what the API billed, from its own usage metadata,
at $0.75 / M input and $3.75 / M output tokens (thinking counted as output; thinking budget 0).

What I take from it:

- **When the task is clean, wiring is a cost decision, not an accuracy one.** Same model, same evidence, same answer; the crew adds latency and tokens.
- **A crew amplifies a bad spec; a solo agent exposes it.** With a contradictory label the solo agent said "unclear" and the flat crew voted confidently for the wrong cause. Confidence from agreement is not evidence when the agreeing agents read the same draft.
- **Isolation beats shared drafts.** The only structural difference between the two crews is whether scouts see each other's work. Shared drafts turned "my slice cannot tell" into a vote.
- **The supervisor should be code.** A number-pinning check caught every invented figure for free; a second model would have cost as much as the scouts.
- **Fix the question before scaling the agents.** Nine misses, one sentence.

## Run it

```bash
LLM_PROVIDER=none python3 -m unittest -v        # tests, no network
LLM_PROVIDER=none python3 run_lab.py --seeds 12 # deterministic scouts, no API calls, $0
export GEMINI_API_KEY=...                        # or ANTHROPIC_API_KEY
python3 run_lab.py --seeds 12                    # the real thing; results/ gets every run as JSON
python3 export_traces.py --out runs.json         # bundle one seed's traces + the summary for the replay page
```

Without a key the scouts fall back to a rule (biggest relative drop in their slice), so the whole
pipeline runs end to end for free; the comparison is then between wirings only, with no model noise.

## Files

- `data.py` — synthetic funnel, six planted causes, the six slices each scout is allowed to see
- `crew.py` — the three wirings, the supervisor check, the vote, the trace writer
- `llm.py` — Gemini → Anthropic → none, with token accounting and number pinning
- `run_lab.py` — runs seeds × wirings, scores, writes `results/`
- `export_traces.py` — turns real runs into the replay JSON the web page reads
- `results/` — every run from the measured experiment, one JSON each, plus `summary.md`

## Design notes

- The scouts' slices are deliberately uneven. The daily series and the headline cannot distinguish any
  cause; the platform slice sees Android and iOS causes; the method slice sees UPI and card causes.
  That is what a real crew looks like: most workers hold evidence that cannot settle the question.
- The supervisor is not a model. It is thirty lines of Python that extract every number from a draft and
  check it exists in the slice the scout was given (sign-insensitive, with rounded and % forms allowed).
  A model judging a model would add cost and a second opinion to argue with; a lookup adds neither.
- "Read each other's drafts" is the flat crew's only difference from the supervised one in terms of
  information flow. It is also the common default in multi-agent frameworks.

MIT licence. Built by [Nikunj Kaushik](https://nikunjkaushik.com).
