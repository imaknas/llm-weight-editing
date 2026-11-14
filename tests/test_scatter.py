import random
import unittest

from wse.defects import PatchDefect, evaluate, make_defective, scattered_patches
from wse.llm import ScriptedLLM
from wse.mlp import MLP
from wse.scatter_exp import (baseline_rows, build, clip_edits, intro, layer_bounds, llm_rows, load_case,
                             random_edits, summarize)
from wse.tasks import TASKS, _best_line_accuracy, random_wavy_waves, wavy_task


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


class WavyTaskTest(unittest.TestCase):
    def test_targets_are_balanced_and_not_a_line(self):
        rng = random.Random(0)
        for _ in range(3):
            task = wavy_task(random_wavy_waves(rng))
            frac = sum(y for _, y in task.sample(rng, 1000)) / 1000
            self.assertTrue(0.35 <= frac <= 0.65)
            self.assertLessEqual(_best_line_accuracy(task), 0.75)

    def test_same_waves_same_labels(self):
        waves = random_wavy_waves(random.Random(1))
        a, b = wavy_task(waves), wavy_task([list(w) for w in waves])  # 存檔再讀回會變成 list
        self.assertEqual([a.label(0.1 * i, -0.3) for i in range(-9, 10)],
                         [b.label(0.1 * i, -0.3) for i in range(-9, 10)])


class BoundsTest(unittest.TestCase):
    def setUp(self):
        self.net = MLP([2, 2, 1], [[[1.0, -3.0], [0.5, 0.2]], [[0.4, -0.6]]], [[0.1, -5.0], [0.05]])

    def test_layer_bounds_cover_weights_and_biases(self):
        self.assertEqual(layer_bounds(self.net), [5.0, 0.6])

    def test_clip_keeps_sign_counts_and_passes_malformed(self):
        edits = [{"layer": 1, "row": 0, "col": 0, "value": 99}, {"layer": 2, "row": 0, "col": 0, "value": -7},
                 {"layer": 1, "row": 0, "col": 1, "value": 2.5}, {"layer": 9, "row": 0, "col": 0, "value": 1},
                 "junk"]
        out, clipped = clip_edits(edits, [5.0, 0.6])
        self.assertEqual(clipped, 2)
        self.assertEqual([e["value"] for e in out[:3]], [5.0, -0.6, 2.5])
        self.assertEqual(out[3:], edits[3:])

    def test_random_edits_stay_in_bounds(self):
        bounds = layer_bounds(self.net)
        for e in random_edits(self.net, 50, bounds, random.Random(0)):
            self.assertLessEqual(abs(e["value"]), bounds[e["layer"] - 1])


class ScatterExpTest(unittest.TestCase):
    def setUp(self):
        self.case = build([1], 1, random.Random(0), py_trainer(30), py_trainer(30), sizes=[2, 6, 1],
                          n_train=150)[0]

    def test_baseline_rows(self):
        rows = baseline_rows(self.case, py_trainer(20), random.Random(3))
        self.assertEqual([r["arm"] for r in rows], ["no_edit", "dense_ft"])

    def test_llm_rows_protocol(self):
        prompts = []

        def fn(prompt):
            prompts.append(prompt)
            if "queries" in prompt:
                return {"queries": [[0.5, 0.5], [0.1, 0.2], "bad", [0.3, 0.3], [9, 9]]}
            return {"edits": [{"layer": 2, "row": 0, "col": -1, "value": 1e6}], "reason": "r"}

        rows = llm_rows(self.case, ScriptedLLM(fn), 3, py_trainer(5), random.Random(4), draws=3)
        self.assertEqual([r["arm"] for r in rows], ["llm", "random_edit", "query_ft"])
        llm = rows[0]
        self.assertEqual((llm["queries_used"], llm["changed"], llm["clipped"]), (3, 1, 1))  # 預算截斷、超限被截
        self.assertEqual(llm["edits"][0]["value"], 1e6)  # 存的是 LLM 原本給的值
        self.assertIn("正確類別", prompts[1])
        self.assertIn("上限", prompts[1])
        self.assertEqual(len(summarize(rows)), 3)

    def test_prompt_gives_no_rule_or_patch_count(self):
        net, _, task = load_case(self.case)
        text = intro(net)
        self.assertIn("沒有公式", text)
        for leak in ("sin", "象限", "x + y"):
            self.assertNotIn(leak, text)


if __name__ == "__main__":
    unittest.main()
