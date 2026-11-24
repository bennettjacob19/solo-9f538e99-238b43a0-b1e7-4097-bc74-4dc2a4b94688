import hashlib
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.web import create_app  # noqa: E402
from app.storage import BLOCK_SIZE  # noqa: E402

from helpers import ORDERED_LAYOUT, SCRAMBLED_LAYOUT, build_set, make_block  # noqa: E402


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tapes = tempfile.mkdtemp()
        self.output = tempfile.mkdtemp()
        build_set(self.tapes, "ordered", ORDERED_LAYOUT)
        build_set(self.tapes, "scrambled", SCRAMBLED_LAYOUT)
        build_set(self.tapes, "bad", SCRAMBLED_LAYOUT, corrupt=11)
        build_set(self.tapes, "bug123", [(1, 0, 0), (2, 0, 1), (0, 0, 2)])
        self.app = create_app(
            tape_root=self.tapes, output_root=self.output, read_delay=0.0
        )
        self.client = self.app.test_client()

    def _wait_done(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.client.get("/status").get_json()
            if not data["running"]:
                return data
            time.sleep(0.02)
        self.fail("恢复任务超时未完成")

    def test_index_lists_sets_and_marks_estimate_reference(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        for name in ("ordered", "scrambled", "bad", "bug123"):
            self.assertIn(name, body)
        # 单一读法的估算必须标注为参考，不能当作可行性结论。
        self.assertIn("参考", body)

    def test_recover_success_publishes_artifact(self):
        resp = self.client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 12, "staging_quota": 0},
        )
        self.assertEqual(resp.status_code, 302)
        status = self._wait_done()
        self.assertTrue(status["result"]["success"])
        self.assertEqual(status["result"]["metrics"]["blocks_read"], 12)

        path = os.path.join(self.output, "ordered", "image.bin")
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.getsize(path), 12 * BLOCK_SIZE)
        with open(path, "rb") as fh:
            image = fh.read()
        expected = b"".join(make_block(i) for i in range(12))
        self.assertEqual(image, expected)
        # 成品内每个块的摘要与清单一致（逐块核对）
        for i in range(12):
            chunk = image[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            self.assertEqual(
                hashlib.sha256(chunk).hexdigest(),
                hashlib.sha256(make_block(i)).hexdigest(),
            )

        dl = self.client.get("/download/ordered")
        self.assertEqual(dl.status_code, 200)
        self.assertEqual(dl.data, expected)

    def test_bug123_exact_quota_delivers_full_image(self):
        # 题面回归：1、2、0 布局，读带 4 / 暂存 1，必须交付完整镜像。
        self.client.post(
            "/recover",
            data={"set_name": "bug123", "read_quota": 4, "staging_quota": 1},
        )
        status = self._wait_done()
        self.assertTrue(status["result"]["success"], status["error"])
        metrics = status["result"]["metrics"]
        self.assertEqual(metrics["blocks_read"], 4)
        self.assertEqual(metrics["staging_peak"], 1)
        self.assertEqual(metrics["rewinds"], 1)
        path = os.path.join(self.output, "bug123", "image.bin")
        self.assertEqual(os.path.getsize(path), 3 * BLOCK_SIZE)
        with open(path, "rb") as fh:
            self.assertEqual(
                fh.read(), b"".join(make_block(i) for i in range(3))
            )
        # 磁盘暂存无残留。
        self.assertFalse(
            os.path.exists(os.path.join(self.output, "bug123", "staging"))
        )

    def test_quota_preflight_failure_reads_nothing_and_keeps_artifact(self):
        # 先成功一次，交付 v1 成品。
        self.client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 12, "staging_quota": 0},
        )
        self._wait_done()
        path = os.path.join(self.output, "ordered", "image.bin")
        with open(path, "rb") as fh:
            before = fh.read()
        mtime_before = os.path.getmtime(path)

        # 额度不足的第二次恢复：清单预演判定，实读量必须为 0，成品原样保留。
        time.sleep(0.01)
        self.client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 5, "staging_quota": 0},
        )
        status = self._wait_done()
        result = status["result"]
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_kind"], "quota")
        self.assertEqual(result["metrics"]["blocks_read"], 0)
        self.assertIn("未读带", result["error"])
        self.assertTrue(os.path.isfile(path))
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), before)
        self.assertEqual(os.path.getmtime(path), mtime_before)
        # 没有遗留半成品文件与暂存目录
        self.assertFalse(os.path.exists(path + ".part"))
        self.assertFalse(
            os.path.exists(os.path.join(self.output, "ordered", "staging"))
        )

    def test_bad_tape_failure_is_classified_and_keeps_artifact(self):
        self.client.post(
            "/recover",
            data={"set_name": "bad", "read_quota": 100, "staging_quota": 12},
        )
        status = self._wait_done()
        result = status["result"]
        self.assertFalse(result["success"])
        # 坏块属于实读发现，与额度不足的预演分类区分开。
        self.assertEqual(result["failure_kind"], "bad_block")
        self.assertEqual(result["metrics"]["bad_block"], 11)
        self.assertGreater(result["metrics"]["blocks_read"], 0)
        self.assertFalse(
            os.path.isfile(os.path.join(self.output, "bad", "image.bin"))
        )

    def test_tight_staging_quota_succeeds_via_reread(self):
        # 乱序样例参考暂存峰值 10；只给 5 也应通过回卷/换带重读完成。
        self.client.post(
            "/recover",
            data={"set_name": "scrambled", "read_quota": 100, "staging_quota": 5},
        )
        status = self._wait_done()
        self.assertTrue(status["result"]["success"], status["error"])
        metrics = status["result"]["metrics"]
        self.assertLessEqual(metrics["staging_peak"], 5)
        self.assertGreater(metrics["blocks_read"], 12)

    def test_download_available_while_recovery_running(self):
        # 先交付一个成品。
        self.client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 12, "staging_quota": 0},
        )
        self._wait_done()
        with open(os.path.join(self.output, "ordered", "image.bin"), "rb") as fh:
            expected = fh.read()

        # 用一个带延迟的新应用实例，确保恢复期间仍能下载。
        slow = create_app(
            tape_root=self.tapes, output_root=self.output, read_delay=0.2
        )
        client = slow.test_client()
        client.post(
            "/recover",
            data={"set_name": "scrambled", "read_quota": 100, "staging_quota": 12},
        )
        running_seen = False
        for _ in range(50):
            st = client.get("/status").get_json()
            if st["running"]:
                running_seen = True
                dl = client.get("/download/ordered")
                self.assertEqual(dl.status_code, 200)
                self.assertEqual(dl.data, expected)
                break
            time.sleep(0.02)
        self.assertTrue(running_seen, "未能观察到运行中的任务")
        # 等它结束，避免线程泄漏到下一个用例
        for _ in range(100):
            if not client.get("/status").get_json()["running"]:
                break
            time.sleep(0.02)

    def test_live_metrics_reported_during_run(self):
        slow = create_app(
            tape_root=self.tapes, output_root=self.output, read_delay=0.15
        )
        client = slow.test_client()
        client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 12, "staging_quota": 0},
        )
        observed = False
        for _ in range(50):
            st = client.get("/status").get_json()
            if st["running"] and st.get("live_metrics"):
                observed = True
                self.assertGreaterEqual(st["live_metrics"]["blocks_read"], 1)
                break
            time.sleep(0.02)
        self.assertTrue(observed, "运行期间未观察到实时读带量")
        for _ in range(100):
            if not client.get("/status").get_json()["running"]:
                break
            time.sleep(0.02)

    def test_concurrent_job_rejected(self):
        slow = create_app(
            tape_root=self.tapes, output_root=self.output, read_delay=0.2
        )
        client = slow.test_client()
        client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 12, "staging_quota": 0},
        )
        resp = client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": 12, "staging_quota": 0},
        )
        self.assertEqual(resp.status_code, 409)
        for _ in range(100):
            if not client.get("/status").get_json()["running"]:
                break
            time.sleep(0.02)

    def test_invalid_inputs(self):
        resp = self.client.post(
            "/recover",
            data={"set_name": "ordered", "read_quota": "x", "staging_quota": 0},
        )
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post(
            "/recover",
            data={"set_name": "missing", "read_quota": 10, "staging_quota": 1},
        )
        self.assertEqual(resp.status_code, 400)
        resp = self.client.get("/download/nope")
        self.assertEqual(resp.status_code, 404)

    def test_scrambled_boundary_quotas(self):
        # 读带卡线（贪心读法需 12）：少 1 失败且零实读，给够则成功。
        self.client.post(
            "/recover",
            data={"set_name": "scrambled", "read_quota": 11, "staging_quota": 12},
        )
        st = self._wait_done()
        self.assertFalse(st["result"]["success"])
        self.assertEqual(st["result"]["failure_kind"], "quota")
        self.assertEqual(st["result"]["metrics"]["blocks_read"], 0)

        # 参考暂存峰值 10：给 9 不再失败，改用重读策略成功。
        self.client.post(
            "/recover",
            data={"set_name": "scrambled", "read_quota": 100, "staging_quota": 9},
        )
        st = self._wait_done()
        self.assertTrue(st["result"]["success"], st["error"])
        self.assertLessEqual(st["result"]["metrics"]["staging_peak"], 9)

        # 卡线 12/10 仍是最省读带的贪心方案。
        self.client.post(
            "/recover",
            data={"set_name": "scrambled", "read_quota": 12, "staging_quota": 10},
        )
        st = self._wait_done()
        self.assertTrue(st["result"]["success"])
        self.assertEqual(st["result"]["metrics"]["blocks_read"], 12)
        self.assertEqual(st["result"]["metrics"]["staging_peak"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
