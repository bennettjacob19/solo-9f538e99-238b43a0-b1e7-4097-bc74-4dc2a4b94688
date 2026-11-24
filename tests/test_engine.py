import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import (  # noqa: E402
    KIND_BAD_BLOCK,
    KIND_QUOTA,
    MemorySink,
    StagingArea,
    baseline,
    simulate_recovery,
)
from app.planner import plan_recovery, preflight  # noqa: E402
from app.storage import BLOCK_SIZE, discover_sets, load_set  # noqa: E402

from helpers import ORDERED_LAYOUT, SCRAMBLED_LAYOUT, build_set, make_block  # noqa: E402


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        build_set(self.tmp, "ordered", ORDERED_LAYOUT)
        build_set(self.tmp, "scrambled", SCRAMBLED_LAYOUT)
        build_set(self.tmp, "bad", SCRAMBLED_LAYOUT, corrupt=11)

    def test_ordered_baseline_and_success(self):
        tape_set = load_set(self.tmp, "ordered")
        est = baseline(tape_set)
        self.assertEqual(est["blocks_read"], 12)
        self.assertEqual(est["staging_peak"], 0)
        self.assertEqual(est["tape_switches"], 1)
        self.assertEqual(est["rewinds"], 0)

        sink = MemorySink()
        result = simulate_recovery(tape_set, 12, 0, sink=sink)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metrics.blocks_read, 12)
        self.assertEqual(result.metrics.staging_peak, 0)
        self.assertEqual(result.metrics.rewinds, 0)
        self.assertEqual(len(sink.chunks), 12)
        self.assertEqual(sink.chunks[0], make_block(0))
        self.assertEqual(sink.chunks[11], make_block(11))
        self.assertEqual(
            b"".join(sink.chunks), b"".join(make_block(i) for i in range(12))
        )

    def test_scrambled_baseline_metrics(self):
        tape_set = load_set(self.tmp, "scrambled")
        est = baseline(tape_set)
        sink = MemorySink()
        result = simulate_recovery(tape_set, 100, 12, sink=sink)
        self.assertTrue(result.success, result.error)
        # 贪心参考估算与宽额度下的实读方案在该布局上指标相同（12/10）。
        self.assertEqual(est["blocks_read"], result.metrics.blocks_read)
        self.assertEqual(est["staging_peak"], result.metrics.staging_peak)
        self.assertEqual(result.metrics.blocks_read, 12)
        self.assertEqual(result.metrics.staging_peak, 10)
        self.assertEqual(result.metrics.tape_switches, 1)
        self.assertEqual(result.metrics.rewinds, 0)
        self.assertEqual(
            b"".join(sink.chunks), b"".join(make_block(i) for i in range(12))
        )

    def test_bug_123_layout_exact_quota_succeeds(self):
        # 题面回归：一盘磁带按 1、2、0 存放，读带 4 / 暂存 1。
        # 旧的“未来块必须全部暂存”会误报暂存不足；正确策略是暂存块 2、
        # 丢弃块 1，写出块 0 后回卷重读块 1，读带恰好 4、暂存峰值 1。
        build_set(self.tmp, "bug123", [(1, 0, 0), (2, 0, 1), (0, 0, 2)])
        tape_set = load_set(self.tmp, "bug123")

        sink = MemorySink()
        result = simulate_recovery(tape_set, 4, 1, sink=sink)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metrics.blocks_read, 4)
        self.assertEqual(result.metrics.staging_peak, 1)
        self.assertEqual(result.metrics.rewinds, 1)
        self.assertEqual(sink.chunks, [make_block(0), make_block(1), make_block(2)])

    def test_bug123_tight_quotas(self):
        build_set(self.tmp, "bug123b", [(1, 0, 0), (2, 0, 1), (0, 0, 2)])
        tape_set = load_set(self.tmp, "bug123b")
        # 刚好够用的两档额度：读 3 需暂存 2；暂存 1 需读 4。
        self.assertTrue(simulate_recovery(tape_set, 3, 2).success)
        self.assertTrue(simulate_recovery(tape_set, 4, 1).success)
        # 少任何一项都无解。
        for read_quota, staging_quota in ((3, 1), (2, 2), (4, 0)):
            result = simulate_recovery(tape_set, read_quota, staging_quota)
            self.assertFalse(result.success, (read_quota, staging_quota))
            self.assertEqual(result.failure_kind, KIND_QUOTA)
            # 预演失败：一次带都没读。
            self.assertEqual(result.metrics.blocks_read, 0)
            self.assertEqual(result.metrics.staging_peak, 0)
            self.assertIn("未读带", result.error)

    def test_read_quota_boundary_and_shortfall(self):
        tape_set = load_set(self.tmp, "ordered")
        ok = simulate_recovery(tape_set, 12, 0)
        self.assertTrue(ok.success)
        short = simulate_recovery(tape_set, 11, 0)
        self.assertFalse(short.success)
        self.assertEqual(short.failure_kind, KIND_QUOTA)
        self.assertIsInstance(short.error, str)
        self.assertIn("额度", short.error)
        # 清单预演判定无解，不允许实读。
        self.assertEqual(short.metrics.blocks_read, 0)

        tape_set2 = load_set(self.tmp, "scrambled")
        need = baseline(tape_set2)["blocks_read"]
        # 暂存放宽后，读带少 1 仍无解；卡线给够则成功。
        self.assertFalse(simulate_recovery(tape_set2, need - 1, 12).success)
        self.assertTrue(simulate_recovery(tape_set2, need, 12).success)

    def test_tight_staging_trades_reads_for_rewinds(self):
        # 乱序样例贪心暂存峰值为 10；只给 9 不再直接失败，而是回卷重读换空间。
        tape_set = load_set(self.tmp, "scrambled")
        tight = simulate_recovery(tape_set, 100, 9)
        self.assertTrue(tight.success, tight.error)
        self.assertLessEqual(tight.metrics.staging_peak, 9)
        self.assertGreater(tight.metrics.blocks_read, 12)
        # 零暂存也能靠反复换带/回卷完成。
        zero = simulate_recovery(tape_set, 100, 0)
        self.assertTrue(zero.success, zero.error)
        self.assertEqual(zero.metrics.staging_peak, 0)
        self.assertGreater(zero.metrics.blocks_read, tight.metrics.blocks_read)
        # 参考值 10 档仍走贪心读法。
        ok = simulate_recovery(tape_set, 100, 10)
        self.assertTrue(ok.success)
        self.assertEqual(ok.metrics.staging_peak, 10)
        self.assertEqual(ok.metrics.blocks_read, 12)

    def test_bad_block_fails_and_reports_index(self):
        tape_set = load_set(self.tmp, "bad")
        result = simulate_recovery(tape_set, 100, 12)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_kind, KIND_BAD_BLOCK)
        self.assertEqual(result.metrics.bad_block, 11)
        self.assertIn("坏块", result.error)
        # 坏块是实读发现的：有实际读带量，且块 0 已先写出。
        self.assertGreater(result.metrics.blocks_read, 0)
        self.assertGreaterEqual(result.blocks_written, 1)

    def test_bad_block_distinct_from_quota_failure(self):
        # 同一带卷：额度不足走预演分类（零读带）；额度充足时才实读出坏块。
        bad = load_set(self.tmp, "bad")
        quota_fail = simulate_recovery(bad, 1, 0)
        self.assertFalse(quota_fail.success)
        self.assertEqual(quota_fail.failure_kind, KIND_QUOTA)
        self.assertEqual(quota_fail.metrics.bad_block, None)
        self.assertEqual(quota_fail.metrics.blocks_read, 0)

        bad_fail = simulate_recovery(bad, 100, 12)
        self.assertFalse(bad_fail.success)
        self.assertEqual(bad_fail.failure_kind, KIND_BAD_BLOCK)
        self.assertEqual(bad_fail.metrics.bad_block, 11)
        self.assertGreater(bad_fail.metrics.blocks_read, 0)

    def test_disk_staging_counts_and_cleans_up(self):
        # 磁盘暂存与内存暂存等价：块数同样计入暂存峰值，结束后目录清空。
        tape_set = load_set(self.tmp, "scrambled")
        staging_dir = os.path.join(self.tmp, "staging")
        os.makedirs(staging_dir)
        sink = MemorySink()
        result = simulate_recovery(
            tape_set, 12, 10, sink=sink, staging_dir=staging_dir
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metrics.staging_peak, 10)
        self.assertEqual(
            b"".join(sink.chunks), b"".join(make_block(i) for i in range(12))
        )
        # 暂存文件逐块删除，空目录被引擎兜底清理，无任何残留。
        self.assertFalse(os.path.exists(staging_dir))

    def test_skip_frames_count_as_reads(self):
        # 块 0 在位置 3：顺读必须经过 0、1、2 三帧，读带量为 4。
        layout = [(1, 0, 0), (2, 0, 1), (3, 0, 2), (0, 0, 3)]
        build_set(self.tmp, "skip", layout)
        tape_set = load_set(self.tmp, "skip")
        result = simulate_recovery(tape_set, 10, 3)
        self.assertTrue(result.success)
        self.assertEqual(result.metrics.blocks_read, 4)

    def test_repeated_reads_count(self):
        # 两带各一帧，块 0 在 tape1：挂 tape1 读 1 块即可完成块 0；
        # 块 1 在 tape0，换带再读 1 块。随后无更多块。
        layout = [(1, 0, 0), (0, 1, 0)]
        build_set(self.tmp, "multi", layout)
        tape_set = load_set(self.tmp, "multi")
        result = simulate_recovery(tape_set, 10, 1)
        self.assertTrue(result.success)
        self.assertEqual(result.metrics.blocks_read, 2)
        self.assertEqual(result.metrics.tape_switches, 1)

    def test_rewind_reread_counts(self):
        # 两帧同带：位置 0 放块 1，位置 1 放块 0；零暂存只能丢弃块 1，
        # 写出块 0 后回卷重读，总读带量 3（重读的帧重复计数）。
        layout = [(1, 0, 0), (0, 0, 1)]
        build_set(self.tmp, "rewind", layout)
        tape_set = load_set(self.tmp, "rewind")
        result = simulate_recovery(tape_set, 10, 0)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metrics.rewinds, 1)
        self.assertEqual(result.metrics.blocks_read, 3)
        self.assertEqual(result.metrics.staging_peak, 0)
        # 给 1 块暂存则不必回卷。
        keep = simulate_recovery(tape_set, 10, 1)
        self.assertTrue(keep.success)
        self.assertEqual(keep.metrics.blocks_read, 2)
        self.assertEqual(keep.metrics.rewinds, 0)
        self.assertEqual(keep.metrics.staging_peak, 1)

    def test_rewind_skips_frames_but_still_counts_reads(self):
        # 布局 0,1,3,2、零暂存：写 0、1 后块 2 在位置 3 被磁头越过，
        # 回卷顺读时位置 0/1 是已写块（SKIP，不暂存不写出），位置 2 块 3
        # 丢弃过，位置 3 才写块 2；SKIP 的两帧照样计入读带量。
        layout = [(0, 0, 0), (1, 0, 1), (3, 0, 2), (2, 0, 3)]
        build_set(self.tmp, "rewindskip", layout)
        tape_set = load_set(self.tmp, "rewindskip")
        plan = plan_recovery(tape_set, 100, 0)
        rewind_passes = [p for p in plan.passes if p.kind == "rewind"]
        self.assertTrue(
            any(
                any(a.action == "skip" for a in p.actions)
                for p in rewind_passes
            )
        )
        sink = MemorySink()
        result = simulate_recovery(tape_set, 100, 0, sink=sink)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metrics.blocks_read, plan.reads)
        self.assertEqual(result.metrics.blocks_read, 7)
        self.assertEqual(sink.chunks, [make_block(i) for i in range(4)])

    def test_staging_area_memory_and_disk_equivalence(self):
        mem = StagingArea()
        mem.put(3, b"abc")
        self.assertEqual(len(mem), 1)
        self.assertIn(3, mem)
        self.assertEqual(mem.get(3), b"abc")
        mem.discard(3)
        self.assertEqual(len(mem), 0)

        disk_dir = os.path.join(self.tmp, "disk-stage")
        os.makedirs(disk_dir)
        disk = StagingArea(disk_dir)
        disk.put(4, b"xyz")
        self.assertTrue(disk.on_disk)
        self.assertEqual(len(disk), 1)
        self.assertEqual(disk.get(4), b"xyz")
        self.assertTrue(os.listdir(disk_dir))
        disk.cleanup()
        self.assertFalse(os.path.exists(disk_dir))

    def test_quota_error_types(self):
        tape_set = load_set(self.tmp, "ordered")
        with self.assertRaises(ValueError):
            simulate_recovery(tape_set, 0, 0)
        with self.assertRaises(ValueError):
            simulate_recovery(tape_set, 1, -1)
        with self.assertRaises(ValueError):
            preflight(tape_set, 0, 0)

    def test_preflight_does_not_touch_tape_bytes(self):
        # 预演只凭清单：把带文件截断后，无可行解仍按额度失败而不是读带报错。
        tape_set = load_set(self.tmp, "ordered")
        path = os.path.join(self.tmp, "ordered", "tape0.bin")
        os.truncate(path, 10)
        result = simulate_recovery(tape_set, 1, 0)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_kind, KIND_QUOTA)
        self.assertEqual(result.metrics.blocks_read, 0)

    def test_discover_ignores_unknown_dirs(self):
        os.makedirs(os.path.join(self.tmp, "not-a-set"))
        found = {s.name for s in discover_sets(self.tmp)}
        self.assertEqual(found, {"ordered", "scrambled", "bad"})


class BadManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_duplicate_block_rejected(self):
        build_set(self.tmp, "dup", [(0, 0, 0), (0, 0, 1)])
        with self.assertRaises(Exception):
            load_set(self.tmp, "dup")

    def test_too_many_blocks_rejected(self):
        layout = [(i, 0, i) for i in range(13)]
        build_set(self.tmp, "big", layout)
        with self.assertRaisesRegex(Exception, "最多 12"):
            load_set(self.tmp, "big")

    def test_truncated_tape_rejected(self):
        build_set(self.tmp, "trunc", [(0, 0, 0), (1, 0, 1)])
        path = os.path.join(self.tmp, "trunc", "tape0.bin")
        os.truncate(path, BLOCK_SIZE + 10)
        with self.assertRaisesRegex(Exception, "大小"):
            load_set(self.tmp, "trunc")

    def test_path_traversal_rejected(self):
        with self.assertRaises(Exception):
            load_set(self.tmp, "../escape")


if __name__ == "__main__":
    unittest.main(verbosity=2)
