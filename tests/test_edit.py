import random
import unittest

from wse.edit_exp import apply_edits, evaluate, make_defective, run, summarize
from wse.llm import ScriptedLLM
from wse.mlp import MLP


class EditTest(unittest.TestCase):
    def test_defect_is_localized(self):
        rng = random.Random(0)
        net = make_defective(rng)
        ev = evaluate(net, rng)
        self.assertLess(ev["defect_region"], 0.5)   # 缺陷象限確實壞了
        self.assertGreater(ev["outside"], 0.85)     # 其餘區域大致正常

    def test_apply_edits_counts_and_ignores_bad(self):
        net = MLP.random_init([2, 3, 1], random.Random(1))
        edits = [{"layer": 1, "row": 0, "col": 1, "value": 9.0},
                 {"layer": 1, "row": 2, "col": -1, "value": -3.0},
                 {"layer": 9, "row": 0, "col": 0, "value": 1.0},   # 越界
                 {"layer": 1}]                                       # 缺欄位
        new, changed = apply_edits(net, edits)
        self.assertEqual(changed, 2)
        self.assertEqual(new.weights[0][0][1], 9.0)
        self.assertEqual(new.biases[0][2], -3.0)
        self.assertNotEqual(net.weights[0][0][1], 9.0)  # 原網路不被改

    def test_run_two_round_protocol(self):
        calls = []

        def fn(prompt):
            calls.append(prompt)
            return {"queries": [[0.5, 0.5], [0.1, 0.2], [0.3, 0.3]]} if "queries" in prompt else {"edits": []}

        rows = run(ScriptedLLM(fn), budgets=[0, 2], seeds=1, rng=random.Random(2))
        self.assertEqual(sum("查詢網路" in c or "你可以先查詢" in c for c in calls), 1)  # budget=0 不查詢
        llm_rows = [r for r in rows if r["arm"] == "llm"]
        self.assertEqual([r.get("queries_used") for r in llm_rows], [0, 2])  # 超過預算被截斷
        self.assertEqual(len(summarize(rows)), 5)  # no_edit + (llm, random) x 2 budgets

    def test_run_resumes_without_recalling_llm(self):
        edit = {"layer": 3, "row": 0, "col": -1, "value": 1.0}
        saved = []
        first = run(ScriptedLLM(lambda p: {"edits": [edit], "reason": "r"}), [0], 2, random.Random(2),
                    on_row=saved.append)
        self.assertEqual(saved, first)  # 每一列都逐筆送出

        def boom(prompt):
            raise AssertionError("已完成的部分不該再呼叫 LLM")

        extra = []
        again = run(ScriptedLLM(boom), [0], 2, random.Random(2), done=saved, on_row=extra.append)
        self.assertEqual(again, first)
        self.assertEqual(extra, [])
        # 只完成第一個 seed 時,第二個 seed 的三列都要補存
        partial = [r for r in saved if r["seed"] == 0]
        run(ScriptedLLM(lambda p: {"edits": [edit], "reason": "r"}), [0], 2, random.Random(2),
            done=partial, on_row=extra.append)
        self.assertEqual(extra, [r for r in first if r["seed"] == 1])


if __name__ == "__main__":
    unittest.main()
