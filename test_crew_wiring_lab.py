"""stdlib tests; run with:  LLM_PROVIDER=none python3 -m unittest -v"""
import os
import unittest

os.environ["LLM_PROVIDER"] = "none"

import crew
import llm
from data import CAUSES, generate, slices
import needle
import tasks


class DataTests(unittest.TestCase):
    def test_deterministic(self):
        a, b = generate(7), generate(7)
        self.assertEqual(a.cause, b.cause)
        self.assertEqual(a.rows[100], b.rows[100])

    def test_every_cause_lowers_conversion(self):
        for c in CAUSES:
            ds = generate(3, cause=c)
            self.assertLess(ds.headline()["relative_change_pct"], -2, c)

    def test_slices_cover_six_scouts(self):
        self.assertEqual(sorted(slices(generate(1))), [f"scout{i}" for i in range(1, 7)])


class CrewTests(unittest.TestCase):
    def test_parse_tolerates_junk(self):
        d = crew._parse('here you go {"cause": "UPI gateway outage", "confidence": "0.9", "note": "x"} thanks')
        self.assertEqual(d["cause"], "upi_gateway_outage"); self.assertEqual(d["confidence"], 0.9)
        self.assertEqual(crew._parse("not json")["cause"], "unclear")

    def test_supervisor_rejects_invented_numbers(self):
        sl = slices(generate(2))["scout2"]
        self.assertIsNotNone(crew.supervisor_check({"cause": "card_bin_block", "confidence": 0.9, "note": "card success fell 99.9%"}, sl))
        self.assertIsNone(crew.supervisor_check({"cause": "unclear", "confidence": 0.1, "note": ""}, sl))

    def test_vote(self):
        drafts = {"a": {"cause": "x", "confidence": 0.9}, "b": {"cause": "y", "confidence": 0.3}, "c": {"cause": "y", "confidence": 0.3}}
        self.assertEqual(crew.vote(drafts, weighted=False)[0], "y")
        self.assertEqual(crew.vote(drafts, weighted=True)[0], "x")

    def test_all_wirings_run_without_a_key(self):
        ds = generate(5)
        for name, fn in crew.WIRINGS.items():
            res = fn(ds)
            self.assertIn(res["answer"], set(CAUSES) | {"unclear"}, name)
            self.assertTrue(res["trace"] and res["trace"][-1]["s"].startswith("done"), name)
        self.assertEqual(llm.USAGE["calls"], 0)


if __name__ == "__main__":
    unittest.main()


class NeedleTests(unittest.TestCase):
    def test_needle_rule_scouts_find_it(self):
        ds = needle.generate(3); sl = needle.slices(ds)
        found = [needle.rule_scout(n, s)["cause"] for n, s in sl.items()]
        self.assertIn(ds.cause, found)
        self.assertEqual(sum(c != "unclear" for c in found), 1)

    def test_needle_supervisor_rejects_foreign_segment(self):
        crew.set_task(tasks.NEEDLE)
        try:
            sl = needle.slices(needle.generate(3))
            other = sl["scout6"]["table"][0]["segment"]
            self.assertIsNotNone(crew.supervisor_check({"cause": other, "confidence": 0.9, "note": ""}, sl["scout1"]))
            for name, fn in crew.WIRINGS.items():
                res = fn(needle.generate(3))
                self.assertEqual(res["answer"], needle.generate(3).cause, name)
        finally:
            crew.set_task(tasks.PLANTED)


class MergeTests(unittest.TestCase):
    def test_evidence_rank_beats_confidence_tie(self):
        crew.set_task(tasks.NEEDLE)
        try:
            ds = needle.generate(3); sl = needle.slices(ds)
            holder = [k for k, v in sl.items() if any(r["segment"] == ds.cause for r in v["table"])][0]
            other = [k for k in sl if k != holder][0]; rival = sl[other]["table"][0]["segment"]
            drafts = {holder: {"cause": ds.cause, "confidence": 0.95, "note": ""}, other: {"cause": rival, "confidence": 0.95, "note": ""}}
            self.assertEqual(crew.vote(drafts, weighted=True, sl=sl)[0], ds.cause)
        finally:
            crew.set_task(tasks.PLANTED)
