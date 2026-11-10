# Crew Wiring Lab

Same question, same model, same data, same budget. Only the wiring between the agents changes.

| wiring | what it is |
|---|---|
| **solo** | one agent reads all the evidence in one prompt and answers once |
| **flat crew** | six scouts, one slice each, run in sequence and **read each other's drafts**; one scout one vote; no check |
| **supervised crew** | six scouts, one slice each, **isolated and run in parallel**; a deterministic supervisor checks every number in a draft against that scout's slice and sends a failing scout back once; drafts are ranked by the evidence they cite |

Three tasks, all with a planted truth so accuracy is exact:

| task | evidence per run | the answer |
|---|---|---|
| **planted cause** | a 28-day checkout funnel, six slices (platform, payment method, traffic source, funnel steps, daily series, headline) | one of six cause labels |
| **needle, 240** | 240 region × platform × payment segments, baseline vs last-week rates; one segment's success rate collapsed, 8 distractors drifted | the exact segment id |
| **needle, 1,200** | the same hunt across 1,200 segments (a merchant tier added), ~50k tokens of evidence; 40 distractors | the exact segment id |

The replay of real runs, with the run log, is on my site: <https://nikunjkaushik.com/crew_wiring.html>

## Results

| task | wiring | seeds | accuracy | calls / run | tokens / run | $ / run | seconds / run |
|---|---|---|---|---|---|---|---|
| Planted cause, 6 labels | solo | 12 | **100%** | 1 | 3,027 | 0.0027 | 3.8 |
|  | flat | 12 | **100%** | 7 | 6,047 | 0.0082 | 30.9 |
|  | supervised | 12 | **100%** | 7 | 5,409 | 0.0079 | 9.0 |
| Needle, 240 segments | solo | 12 | **100%** | 1 | 9,968 | 0.0082 | 3.9 |
|  | flat | 12 | **92%** | 7 | 13,093 | 0.0146 | 31.8 |
|  | supervised | 12 | **100%** | 7.1 | 12,914 | 0.0155 | 9.3 |
| Needle, 1,200 segments | solo | 12 | **58%** | 1 | 51,556 | 0.0399 | 6.2 |
|  | flat | 12 | **67%** | 7 | 54,759 | 0.0468 | 31.6 |
|  | supervised | 12 | **100%** | 7.1 | 55,066 | 0.0475 | 16.3 |

The runs in `results/`: 542 calls, $2.30. With the first pass, the v1 merge runs and debugging calls, the whole project cost about $3 in API credit.

Model `gemini-3.8-flash`, thinking budget 0, temperature 0.2. Cost is what the API billed, from its own usage
metadata, at $0.75 / M input and $3.75 / M output tokens. Wall time is measured; supervised scouts run on six threads.

What I take from it:

- **When the evidence fits one prompt, wiring is a cost decision, not an accuracy one.** Same model, same evidence, same answer; the crew adds tokens and latency.
- **When the evidence outgrows one prompt, the crew is the only thing that works.** At 1,200 segments the solo agent did not hallucinate; it said "unclear" in 5 of 12 runs. Splitting the haystack is what found the needle.
- **Cost per run converges as the evidence grows.** At 1,200 segments the three wirings cost the same, because the evidence is the cost and a crew splits it rather than multiplying it.
- **Isolation is where the speed comes from.** Isolated scouts can run in parallel; scouts that read each other's drafts cannot.
- **The vote is the weak link, not the scouts.** In the flat crew's misses the scout holding the truth usually found it (confidence 0.95–1.0) and lost the majority vote to scouts naming distractors. The supervised crew's two misses were exact confidence ties, 0.95 vs 0.95, between a 45% collapse and a 12% drift.
- **So rank by evidence, not confidence.** Version 2 of the supervised merge scores each draft by the drop it names, recomputed from the scout's own slice. Replayed offline on the recorded v1 drafts it answers 12/12 on both needle tasks; the measured v2 runs are in the table above and the v1 runs are kept in `results/*/v1_confidence_merge/`.
- **The supervisor should be code.** A number-pinning lookup caught every invented figure for free; a model judging a model would cost as much as the scouts.

### A first pass worth recording

The first full pass of the planted-cause task scored solo 75 %, flat 75 %, supervised 83 %, because the label text
I handed the agents contradicted the data (it said "per-segment rates held" while paid checkout rate fell 38 %).
The solo agent noticed and said "unclear"; the flat crew voted confidently for the wrong cause, 4–1, because scouts whose
slice could not tell copied the first draft. One sentence fixed, every wiring 100 %. The per-seed lines are in
`results/first_pass.log`. A crew amplifies a bad question; a solo agent exposes it.

## Run it

```bash
LLM_PROVIDER=none python3 -m unittest -v                    # tests, no network
LLM_PROVIDER=none python3 run_lab.py --task needle-xl       # deterministic scouts, no API calls, $0
export GEMINI_API_KEY=...                                    # or ANTHROPIC_API_KEY
python3 run_lab.py --task planted --seeds 12                 # results/
python3 run_lab.py --task needle --seeds 12                  # results/needle/
python3 run_lab.py --task needle-xl --seeds 12               # results/needle-xl/   (~$1.60)
MERGE=confidence python3 run_lab.py --task needle-xl --wirings supervised   # the v1 merge
python3 export_traces.py --out runs.json                     # bundle one replay seed per task + summaries for the page
```

Without a key the scouts fall back to a rule (biggest relative drop in their slice), so the whole pipeline runs
end to end for free; the comparison is then between wirings only, with no model noise.

## Files

- `data.py` — task A: synthetic funnel, six planted causes, the six slices
- `needle.py` — tasks B and C: 240 / 1,200 segments, one collapse, distractors, slices, the evidence score
- `tasks.py` — the task interface the crew uses (generate, slices, labels, normalise, in_slice, render, score)
- `crew.py` — the three wirings, the supervisor check, the two merges, the trace writer
- `llm.py` — Gemini → Anthropic → none, with retries, token accounting and number pinning
- `run_lab.py` — runs seeds × wirings, scores, writes `results/`
- `export_traces.py` — turns real runs into the replay JSON the web page reads
- `results/` — every run as JSON, per task, plus `summary.md`; `first_pass.log`; `v1_confidence_merge/`

## Design notes

- The scouts' slices are deliberately uneven on task A: the daily series and the headline cannot distinguish any
  cause. That is what a real crew looks like: most workers hold evidence that cannot settle the question.
- On the needle tasks the slices are even and the question is pure recall under load. That is the regime where a
  crew should win, and it did, at the size where one prompt starts to give up.
- The supervisor is not a model. It is a few lines of Python that extract every number from a draft and check it
  exists in the slice the scout was given (sign-insensitive, with rounded and % forms allowed), and that the thing
  the scout names is something its slice can show.
- "Read each other's drafts" is the flat crew's only difference from the supervised one in terms of information
  flow. It is also the common default in multi-agent frameworks.

MIT licence. Built by [Nikunj Kaushik](https://nikunjkaushik.com).
