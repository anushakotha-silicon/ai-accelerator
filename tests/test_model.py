"""Sanity checks on the analytical model.  python3 -m unittest discover tests"""
import unittest
from dataclasses import replace

from model.engine import best_decode_step, decode_step, prefill
from model.hw import N3, Chip, Serving
from model.scenarios import POLICIES, run_agents, serve_chat
from model.workloads import LLAMA_8B, LLAMA_70B, MIXTRAL_8X7B, AgentProfile, ChatProfile

CHIP = Chip()
SRV = Serving(spec_k=4)


class TestWorkloads(unittest.TestCase):
    def test_param_counts_match_published(self):
        self.assertAlmostEqual(LLAMA_8B.total_params / 1e9, 8.03, delta=0.05)
        self.assertAlmostEqual(LLAMA_70B.total_params / 1e9, 70.6, delta=0.3)
        self.assertAlmostEqual(MIXTRAL_8X7B.total_params / 1e9, 46.7, delta=0.3)
        self.assertAlmostEqual(MIXTRAL_8X7B.active_params_per_token / 1e9, 12.9, delta=0.3)

    def test_kv_bytes(self):
        self.assertEqual(LLAMA_8B.kv_bytes_per_token(1.0), 64 * 1024)
        self.assertEqual(LLAMA_70B.kv_bytes_per_token(1.0), 160 * 1024)

    def test_moe_touches_all_experts_at_large_batch(self):
        self.assertAlmostEqual(MIXTRAL_8X7B.experts_touched(1), 2.0)
        self.assertAlmostEqual(MIXTRAL_8X7B.experts_touched(512), 8.0, places=3)


class TestEngine(unittest.TestCase):
    def test_peak(self):
        self.assertAlmostEqual(CHIP.peak_tflops("fp8"), 1258.3, delta=1)

    def test_decode_batch1_memory_bound_and_hbm_dominated(self):
        c = decode_step(LLAMA_8B, CHIP, Serving(), 1, 1024)
        self.assertEqual(c.bound, "memory")
        self.assertGreater(c.energy["hbm"], 50 * c.energy["mac"])

    def test_prefill_compute_bound(self):
        self.assertEqual(prefill(LLAMA_70B, CHIP, SRV, 4096).bound, "array")

    def test_batching_amortises_weights(self):
        e1 = decode_step(LLAMA_8B, CHIP, Serving(), 1, 1024).total_j
        e64 = decode_step(LLAMA_8B, CHIP, Serving(), 64, 1024).total_j / 64
        self.assertLess(e64, e1 / 10)

    def test_adaptive_spec_backs_off_at_large_batch(self):
        s_small, _ = best_decode_step(LLAMA_8B, CHIP, replace(SRV, spec_accept=0.35), 1, 1024)
        s_large, _ = best_decode_step(LLAMA_8B, CHIP, replace(SRV, spec_accept=0.35), 256, 1024)
        self.assertGreater(s_small.spec_k, s_large.spec_k)

    def test_fp4_beats_bf16(self):
        fp4 = serve_chat(LLAMA_8B, CHIP, SRV, ChatProfile(), 64).tokens_per_j
        bf16 = serve_chat(LLAMA_8B, CHIP, replace(SRV, weight_dtype="bf16", compute_dtype="bf16",
                                                   kv_dtype="bf16"), ChatProfile(), 64).tokens_per_j
        self.assertGreater(fp4, 2 * bf16)

    def test_power_within_tdp(self):
        for b in (1, 64, 256):
            self.assertLess(serve_chat(LLAMA_70B, CHIP, SRV, ChatProfile(), b).avg_power_w, CHIP.tdp_w)


class TestAgents(unittest.TestCase):
    def test_kv_reuse_beats_recompute(self):
        for m in (LLAMA_8B, LLAMA_70B):
            rec = run_agents(m, CHIP, SRV, AgentProfile(), "recompute")
            hbm = run_agents(m, CHIP, SRV, AgentProfile(), "hbm-cache")
            self.assertLess(hbm.j_per_turn * 3, rec.j_per_turn)

    def test_tiered_kv_helps_long_sessions(self):
        prof = AgentProfile(turns=40, tool_output_tokens=3000, tool_latency_s=20.0, max_turn_latency_s=15.0)
        hbm = run_agents(LLAMA_8B, CHIP, SRV, prof, "hbm-cache")
        tier = run_agents(LLAMA_8B, CHIP, SRV, prof, "tiered-kv")
        self.assertGreater(tier.turns_per_s, 1.3 * hbm.turns_per_s)

    def test_results_respect_slo(self):
        prof = AgentProfile()
        for p in POLICIES:
            r = run_agents(LLAMA_8B, CHIP, SRV, prof, p)
            self.assertLessEqual(r.turn_latency_s, prof.max_turn_latency_s)
            self.assertLessEqual(r.utilization, 1.0)


class TestNodeAndPower(unittest.TestCase):
    def test_area_calibration_and_n3_reinvestment(self):
        self.assertAlmostEqual(Chip().die_area_mm2, 442.0, delta=0.5)
        # 48 N3 tiles fit in the N5 32-tile die area (within 2%)
        self.assertAlmostEqual(Chip(node=N3, tiles=48).die_area_mm2 / Chip().die_area_mm2, 1.0, delta=0.02)

    def test_n3_port_is_perf_neutral_and_cheaper(self):
        a = serve_chat(LLAMA_70B, Chip(), SRV, ChatProfile(), 64)
        b = serve_chat(LLAMA_70B, Chip(node=N3), SRV, ChatProfile(), 64)
        self.assertLessEqual(b.tpot_ms, 1.02 * a.tpot_ms)   # scheduler contract: within 2%
        self.assertGreater(b.tokens_per_j, 1.15 * a.tokens_per_j)

    def test_dvfs_lowers_voltage_in_memory_bound_decode_without_slowdown(self):
        chip = Chip(node=N3, dvfs="efficiency")
        c = decode_step(LLAMA_70B, chip, Serving(), 64, 1280)
        nominal = decode_step(LLAMA_70B, Chip(node=N3), Serving(), 64, 1280)
        self.assertLess(c.opp[0], 0.75)
        self.assertLessEqual(c.time_s, 1.02 * nominal.time_s)
        self.assertLess(c.total_j, nominal.total_j)

    def test_joint_scheduler_never_trades_more_than_2pct_latency(self):
        for chip in (Chip(), Chip(node=N3, dvfs="efficiency", power_gating=True)):
            for b in (8, 128):
                r = serve_chat(LLAMA_8B, chip, SRV, ChatProfile(), b)
                fastest = min(serve_chat(LLAMA_8B, replace(chip, dvfs="off"), replace(SRV, spec_k=k), ChatProfile(), b).tpot_ms
                              for k in range(SRV.spec_k + 1))
                self.assertLessEqual(r.tpot_ms, 1.021 * fastest)

    def test_performance_policy_respects_tdp(self):
        chip = Chip(node=N3, tiles=64, dvfs="performance")
        self.assertLessEqual(prefill(LLAMA_70B, chip, SRV, 4096).power_w, chip.tdp_w)


if __name__ == "__main__":
    unittest.main()
