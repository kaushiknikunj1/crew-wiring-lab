"""stdlib tests; run with:  LLM_PROVIDER=none python3 -m unittest -v"""
import os
import unittest

os.environ["LLM_PROVIDER"] = "none"

import crew
import llm
from data import CAUSES, generate, slices


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
            self.assertIn(res["answer"], crew.LABELS, name)
            self.assertTrue(res["trace"] and res["trace"][-1]["s"].startswith("done"), name)
        self.assertEqual(llm.USAGE["calls"], 0)


if __name__ == "__main__":
    unittest.main()
