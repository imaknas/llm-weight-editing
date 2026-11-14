import random
import unittest

from wse.defects import PatchDefect, evaluate, make_defective, scattered_patches
from wse.llm import ScriptedLLM
from wse.mlp import MLP
from wse.scatter_exp import baseline_rows, build, changed_slots, keep_top, llm_rows, min_sparse, summarize
from wse.tasks import TASKS


def py_trainer(epochs):
    return lambda net, data, rng: net.train(data, epochs, 0.02, rng)


class PatchTest(unittest.TestCase):
    def test_area_is_constant_and_patches_do_not_share_edges(self):
        for k in (1, 4, 9, 16):
            d = scattered_patches(random.Random(k), k)
            self.assertEqual(d.n_patches, k)
            self.assertAlmostEqual(d.area_fraction(), 0.25)
            idx = [(round((x + 1) / d.side), round((y + 1) / d.side)) for x, y in d.cells]
            for a in idx:
                for b in idx:
                    self.assertNotEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_contains(self):
        d = PatchDefect(((0.0, 0.0),), 1.0)
        self.assertTrue(d.contains(0.5, 0.5))
        self.assertFalse(d.contains(-0.5, 0.5))

    def test_defective_net_is_wrong_only_inside(self):
        rng = random.Random(0)
        d = PatchDefect(((0.0, 0.0),), 1.0)
        net = make_defective(TASKS["diagonal"], d, [2, 8, 1], rng, py_trainer(60), py_trainer(60),
                             n_train=200, ft_n_train=200)
        e = evaluate(net, TASKS["diagonal"], d, rng, 400)
        self.assertLess(e["defect_region"], 0.3)
        self.assertGreater(e["outside"], 0.85)


class ScatterExpTest(unittest.TestCase):
    def setUp(self):
        self.cases = build([1], 1, random.Random(0), TASKS["diagonal"], py_trainer(40), py_trainer(40),
                           sizes=[2, 6, 1])

    def test_keep_top_all_equals_target(self):
        a = MLP.random_init([2, 3, 1], random.Random(1))
        b = MLP.random_init([2, 3, 1], random.Random(2))
        ranked = changed_slots(a, b)
        self.assertEqual(len(ranked), a.num_params())
        self.assertEqual(keep_top(a, ranked, len(ranked)).to_dict(), b.to_dict())
        self.assertEqual(keep_top(a, ranked, 0).to_dict(), a.to_dict())

    def test_min_sparse_finds_first_passing_size(self):
        a = MLP.random_init([2, 3, 1], random.Random(1))
        ranked = changed_slots(a, MLP.random_init([2, 3, 1], random.Random(2)))
        calls = []

        def score(net):
            calls.append(net)
            ok = len(calls) >= 3  # 第三次(n=4)才過
            return {"defect_region": 1.0 if ok else 0.0, "outside": 1.0}

        self.assertEqual(min_sparse(a, ranked, score), 4)
        self.assertIsNone(min_sparse(a, ranked, lambda n: {"defect_region": 0.0, "outside": 1.0}))

    def test_rows_for_every_arm(self):
        c = self.cases[0]
        base, ranked = baseline_rows(c, py_trainer(40), random.Random(3))
        self.assertEqual([r["arm"] for r in base], ["no_edit", "dense_ft"])
        llm = ScriptedLLM(lambda p: {"queries": [[0.5, 0.5]]} if "queries" in p
                          else {"edits": [{"layer": 2, "row": 0, "col": -1, "value": 9.0}], "reason": "r"})
        rows = llm_rows(c, llm, 5, ranked, random.Random(4), draws=3)
        self.assertEqual([r["arm"] for r in rows], ["llm", "random_edit", "sparse_ft"])
        self.assertEqual((rows[0]["queries_used"], rows[0]["changed"], rows[0]["max_abs_value"]), (1, 1, 9.0))
        self.assertEqual(len(summarize(base + rows)), 5)

    def test_prompt_does_not_reveal_patch_count(self):
        from wse.scatter_exp import intro, load_case
        net, _, task = load_case(self.cases[0])
        text = intro(net, task)
        self.assertIn(task.description, text)
        self.assertNotIn("象限", text)


if __name__ == "__main__":
    unittest.main()
