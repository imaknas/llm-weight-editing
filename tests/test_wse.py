import random
import unittest

from wse.llm import ScriptedLLM
from wse.mlp import MLP
from wse.read_exp import Case, build_cases, build_none_controls, build_prompt, run_cases, summarize
from wse.tasks import TASKS


class MLPTest(unittest.TestCase):
    def test_permute_hidden_preserves_function(self):
        rng = random.Random(1)
        net = MLP.random_init([2, 6, 6, 1], rng)
        p = net.permute_hidden(rng)
        self.assertNotEqual(net.weights, p.weights)
        for _ in range(20):
            x = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
            self.assertAlmostEqual(net.predict(x), p.predict(x), places=9)

    def test_shuffle_keeps_values_changes_function(self):
        rng = random.Random(2)
        net = MLP.random_init([2, 6, 6, 1], rng)
        s = net.shuffle_weights(rng)
        flat = lambda m: sorted(w for W in m.weights for r in W for w in r)
        self.assertEqual(flat(net), flat(s))
        self.assertEqual(net.num_params(), s.num_params())

    def test_training_learns_diagonal(self):
        rng = random.Random(3)
        t = TASKS["diagonal"]
        net = MLP.random_init([2, 6, 1], rng)
        net.train(t.sample(rng, 150), 40, 0.05, rng)
        self.assertGreater(net.accuracy(t.sample(rng, 300)), 0.9)

    def test_dict_roundtrip(self):
        net = MLP.random_init([2, 4, 1], random.Random(4))
        self.assertEqual(MLP.from_dict(net.to_dict()).weights, net.weights)


class ReadExpTest(unittest.TestCase):
    def test_run_and_summarize_with_scripted_llm(self):
        net = MLP.random_init([2, 4, 1], random.Random(5))
        cases = [Case("trained", "xor", "xor", net, 0.9), Case("random", "none", "none", net, 0.0)]
        rows = run_cases(ScriptedLLM(lambda prompt: {"task": "xor", "reason": "x"}), cases)
        self.assertEqual(summarize(rows), {"trained": "1/1", "random": "0/1"})

    def test_prompt_contains_weights_and_options(self):
        net = MLP.random_init([2, 4, 1], random.Random(6))
        p = build_prompt(net)
        for name in TASKS:
            self.assertIn(name, p)
        self.assertIn("W1", p)



class NoneControlTest(unittest.TestCase):
    def test_controls_leave_no_zero_bias_tell(self):
        cases = build_none_controls(random.Random(0), 2, epochs=3, n_train=20, sizes=[2, 4, 1])
        self.assertEqual([c.kind for c in cases], ["random", "random", "randlabel", "randlabel"])
        for c in cases:
            self.assertEqual(c.expected, "none")
            self.assertTrue(all(b != 0.0 for layer in c.net.biases for b in layer))

    def test_task_cases_do_not_depend_on_controls(self):
        # 對照組排在最後才取亂數,改動對照組不會改到前面已跑過的任務網路
        kw = dict(seeds_per_task=1, epochs=2, n_train=10, n_eval=10, sizes=[2, 3, 1])
        cases = build_cases(random.Random(0), **kw)
        n_task = 3 * len(TASKS)
        self.assertEqual({c.kind for c in cases[:n_task]}, {"trained", "permuted", "shuffled"})
        self.assertEqual({c.kind for c in cases[n_task:]}, {"random", "randlabel"})

class BrunnaCompleteTest(unittest.TestCase):
    def _llm(self, returncode, stdout, stderr=""):
        import subprocess
        from wse.llm import BrunnaCompleteLLM
        return BrunnaCompleteLLM("m", runner=lambda *a, **k: subprocess.CompletedProcess(a, returncode, stdout, stderr))

    def test_failure_reason_comes_from_llm_error(self):
        from wse.llm import LLMError
        with self.assertRaises(LLMError) as cm:
            self._llm(1, '{"is_error": true, "llm_error": "超過 600 秒"}').complete("p")
        self.assertIn("超過 600 秒", str(cm.exception))
        with self.assertRaises(LLMError) as cm:
            self._llm(2, "", "hub-run: 找不到工作").complete("p")
        self.assertIn("找不到工作", str(cm.exception))

    def test_success_accumulates_reported_cost(self):
        llm = self._llm(0, '{"structured_output": {"a": 1}, "total_cost_usd": 0.5}')
        self.assertEqual(llm.complete("p", {"type": "object"}), {"a": 1})
        llm.complete("p", {"type": "object"})
        self.assertEqual((llm.calls, llm.total_cost_usd), (2, 1.0))


if __name__ == "__main__":
    unittest.main()


class RetryAndResumeTest(unittest.TestCase):
    def test_retry_then_succeed_and_log(self):
        from wse.llm import LLMError, RetryingLLM
        calls, logs = [], []

        class Flaky:
            def complete(self, prompt, schema=None):
                calls.append(1)
                if len(calls) < 3:
                    raise LLMError("暫時失敗")
                return {"task": "xor", "reason": ""}

        out = RetryingLLM(Flaky(), attempts=3, log=logs.append).complete("p")
        self.assertEqual(out["task"], "xor")
        self.assertEqual(len(logs), 2)

    def test_retry_gives_up_visibly(self):
        from wse.llm import LLMError, RetryingLLM

        class Dead:
            def complete(self, prompt, schema=None):
                raise LLMError("壞了")

        with self.assertRaises(LLMError):
            RetryingLLM(Dead(), attempts=2, log=lambda m: None).complete("p")

    def test_run_cases_resumes_and_streams(self):
        net = MLP.random_init([2, 4, 1], random.Random(7))
        cases = [Case("trained", "xor", "xor", net, 0.9), Case("random", "none", "none", net, 0.0)]
        seen = []
        done = [{"i": 0, "kind": "trained", "task": "xor", "expected": "xor", "answer": "xor",
                 "correct": True, "reason": "", "own_accuracy": 0.9}]
        rows = run_cases(ScriptedLLM(lambda p: {"task": "none", "reason": ""}), cases, done, seen.append)
        self.assertEqual([r["i"] for r in seen], [1])      # 只新跑未完成的
        self.assertEqual([r["i"] for r in rows], [0, 1])
