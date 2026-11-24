"""恢复演练引擎。

磁带机械语义（按演练规则实现）：

* 每盘磁带只能从头顺读；换带或回卷之后，下一次读取都从第 0 帧重新开始。
* 顺读路径上跳过的帧同样计入读带量；同一帧重复读到也重复计数。
* 提前读到、暂时不能按块号写出的块进入暂存区；内存与磁盘承载等价，
  统一按块数计入暂存额度（暂存块数峰值受暂存额度约束）。
* 镜像只允许按块号 0、1、2…… 追加，不能回读、不能改前面的块。
* 每个块在**读到时（入暂存/写出前）**与**由暂存写出前**各核对一次
  SHA-256；坏块立即失败。
* 额度可行性在**开带前只凭清单**判定（见 :mod:`app.planner`）：无可行解
  时一次带都不读，直接报额度不足；实读阶段才可能发现坏块。
* 失败时不产出新镜像（由上层删除半成品并保留上一次成品）。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .planner import (
    DISCARD,
    MOUNT,
    REWIND,
    SKIP,
    STAGE,
    SWITCH,
    WRITE,
    Plan,
    QuotaExceeded as PlannerQuotaExceeded,
    preflight,
)
from .storage import BLOCK_SIZE, BlockEntry, TapeSet, read_frame, sha256_hex


class RecoveryError(RuntimeError):
    """恢复失败：坏块或额度不足。"""


class QuotaExceeded(RecoveryError, PlannerQuotaExceeded):
    """额度不足：同时属于恢复失败与清单预演失败两个层级。"""


class BadBlock(RecoveryError):
    pass


# 失败分类（页面据此区分“额度不足（预演）”和“实读发现坏块”）。
KIND_QUOTA = "quota"
KIND_BAD_BLOCK = "bad_block"
KIND_OTHER = "other"


class ImageSink(Protocol):
    """镜像输出端：只允许按块号顺序追加。"""

    def write_block(self, index: int, data: bytes) -> None: ...

    def close(self) -> None: ...


class FileImageSink:
    """写临时镜像文件；只有全部块写完并核对通过后才由上层原子发布。"""

    def __init__(self, tmp_path: str) -> None:
        self._tmp_path = tmp_path
        self._fh = open(tmp_path, "wb")  # noqa: SIM115 - 生命周期由 close 管理
        self._next_index = 0
        self.bytes_written = 0

    def write_block(self, index: int, data: bytes) -> None:
        if index != self._next_index:
            raise RecoveryError(
                f"镜像只允许顺序追加：期望块 {self._next_index}，收到块 {index}"
            )
        if len(data) != BLOCK_SIZE:
            raise RecoveryError(f"块 {index} 不是 {BLOCK_SIZE} 字节")
        self._fh.write(data)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._next_index += 1
        self.bytes_written += len(data)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @property
    def next_index(self) -> int:
        return self._next_index


class MemorySink:
    """测试用内存输出端。"""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write_block(self, index: int, data: bytes) -> None:
        if index != len(self.chunks):
            raise RecoveryError(
                f"镜像只允许顺序追加：期望块 {len(self.chunks)}，收到块 {index}"
            )
        self.chunks.append(data)

    def close(self) -> None:
        return None


class StagingArea:
    """暂存区：内存与磁盘承载等价，统一按块数计数。

    给 ``staging_dir`` 时提前块落盘（每块一个临时文件），否则保留在内存中；
    两种模式的块数口径完全一致。磁盘目录随 :meth:`cleanup` 整体清理。
    """

    def __init__(self, staging_dir: str | None = None) -> None:
        self._on_disk = staging_dir is not None
        self._dir = staging_dir
        self._memory: dict[int, bytes] = {}
        self._files: dict[int, str] = {}

    @property
    def on_disk(self) -> bool:
        return self._on_disk

    def put(self, index: int, data: bytes) -> None:
        if self._on_disk:
            fd, path = tempfile.mkstemp(prefix=f"block{index}-", dir=self._dir)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            self._files[index] = path
        else:
            self._memory[index] = data

    def get(self, index: int) -> bytes:
        if self._on_disk:
            with open(self._files[index], "rb") as fh:
                return fh.read()
        return self._memory[index]

    def __contains__(self, index: object) -> bool:
        return index in self._memory or index in self._files

    def __len__(self) -> int:
        return len(self._memory) + len(self._files)

    def discard(self, index: int) -> None:
        if self._on_disk:
            path = self._files.pop(index)
            if os.path.exists(path):
                os.remove(path)
        else:
            self._memory.pop(index, None)

    def cleanup(self) -> None:
        if not self._on_disk or not self._dir:
            self._memory.clear()
            return
        for index in list(self._files):
            self.discard(index)
        try:
            os.rmdir(self._dir)
        except OSError:
            pass


@dataclass
class RecoveryMetrics:
    blocks_read: int = 0           # 实际读带块数（含跳过、含重读）
    staging_peak: int = 0          # 暂存块数峰值（内存/磁盘统一计数）
    tape_mounts: int = 0           # 挂带次数（含第一次挂带）
    tape_switches: int = 0         # 在两盘带之间切换的次数
    rewinds: int = 0               # 同一盘带回卷次数
    bad_block: int | None = None   # 命中的坏块号（若有）
    head_position: int = 0         # 当前磁头：本盘带上“下一个待读帧”的序号
    failure_kind: str | None = None  # quota / bad_block / other
    events: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.events.append(message)


@dataclass
class RecoveryResult:
    success: bool
    set_name: str
    metrics: RecoveryMetrics
    error: str | None = None
    failure_kind: str | None = None
    artifact_path: str | None = None
    blocks_written: int = 0
    plan: Plan | None = None


def _baseline_metrics(tape_set: TapeSet) -> RecoveryMetrics:
    """贪心读法（提前读到的块一律暂存）的确定性指标，仅作页面前沿参考值。

    这是“单一读法”的估算，不是额度可行性的判据：可行解由规划器在
    :func:`app.planner.plan_recovery` 中枚举暂存/丢弃策略求出。
    """

    metrics = RecoveryMetrics()
    current_tape: int | None = None
    head_position = 0
    staged: dict[int, bytes] = {}
    written = 0
    n = tape_set.block_count

    for target in range(n):
        if target in staged:
            staged.pop(target, None)
            written += 1
            continue
        entry = tape_set.entry(target)
        if current_tape is None:
            current_tape = entry.tape
            head_position = 0
            metrics.tape_mounts += 1
            metrics.log(f"挂带 tape{entry.tape}")
        elif entry.tape != current_tape:
            current_tape = entry.tape
            head_position = 0
            metrics.tape_mounts += 1
            metrics.tape_switches += 1
            metrics.log(f"换带 tape{entry.tape}，从头顺读")
        elif entry.position < head_position:
            metrics.rewinds += 1
            head_position = 0
            metrics.log(f"回卷 tape{entry.tape}，从头顺读")
        # 目标帧在磁头之后：磁头自然前移，无需回卷。
        metrics.blocks_read += entry.position + 1 - head_position
        for position in range(head_position, entry.position + 1):
            frame_entry = _entry_at(tape_set, entry.tape, position)
            if (
                frame_entry.index >= written
                and frame_entry.index != target
                and frame_entry.index not in staged
            ):
                staged[frame_entry.index] = b""
        head_position = entry.position + 1
        metrics.staging_peak = max(metrics.staging_peak, len(staged))
        staged.pop(target, None)
        written += 1
    return metrics


def _entry_at(tape_set: TapeSet, tape: int, position: int) -> BlockEntry:
    for entry in tape_set.blocks:
        if entry.tape == tape and entry.position == position:
            return entry
    raise RecoveryError(f"清单缺少 tape{tape} 位置 {position} 的记录")


def baseline(tape_set: TapeSet) -> dict[str, int]:
    """返回贪心读法的读带量与暂存峰值（页面标注为“参考”，非可行性判据）。"""

    metrics = _baseline_metrics(tape_set)
    return {
        "blocks_read": metrics.blocks_read,
        "staging_peak": metrics.staging_peak,
        "tape_mounts": metrics.tape_mounts,
        "tape_switches": metrics.tape_switches,
        "rewinds": metrics.rewinds,
    }


def _record_pass_boundary(
    metrics: RecoveryMetrics, kind: str, tape: int
) -> None:
    if kind == MOUNT:
        metrics.tape_mounts += 1
        metrics.log(f"挂带 tape{tape}，从头顺读")
    elif kind == SWITCH:
        metrics.tape_mounts += 1
        metrics.tape_switches += 1
        metrics.log(f"换带 tape{tape}，从头顺读")
    elif kind == REWIND:
        metrics.rewinds += 1
        metrics.log(f"回卷 tape{tape}，从头顺读")
    # CONTINUE：磁头自然前移，无机械动作。


def simulate_recovery(
    tape_set: TapeSet,
    read_quota: int,
    staging_quota: int,
    sink: ImageSink | None = None,
    *,
    staging_dir: str | None = None,
    read_delay: float = 0.0,
    sleep: Callable[[float], None] | None = None,
    progress: Callable[[RecoveryMetrics], None] | None = None,
) -> RecoveryResult:
    """按给定额度执行恢复演练。

    流程：

    1. 开带前只凭清单做额度预演（:func:`preflight`），无可行解直接失败，
       一次带都不读；
    2. 按规划出的 pass 顺序实读：跳过/重读的帧都计读带量，提前块进入
       内存或磁盘暂存（统一计入暂存峰值）；
    3. 每帧读到时核对 SHA-256，暂存块写出前再核对一次；坏块立即失败。

    sink 为 None 时使用内存输出端（供测试）；``staging_dir`` 给定时暂存
    落盘，否则留在内存，两者计数口径一致。
    """

    if read_quota <= 0 or staging_quota < 0:
        raise ValueError("读带额度必须为正整数，暂存额度必须为非负整数")

    own_sink = sink is None
    if sink is None:
        sink = MemorySink()
    metrics = RecoveryMetrics()
    n = tape_set.block_count
    written = 0
    plan: Plan | None = None

    def drain_staged(staging: StagingArea) -> None:
        nonlocal written
        while written in staging:
            data = staging.get(written)
            entry = tape_set.entry(written)
            # 写出前第二次核对 SHA-256（第一次在读到入暂存时）。
            if sha256_hex(data) != entry.sha256:
                raise BadBlock(
                    f"块 {written} 写出前摘要核对失败（暂存内容损坏）"
                )
            sink.write_block(written, data)
            staging.discard(written)
            metrics.log(f"块 {written} 由暂存区追加写入镜像")
            written += 1

    staging = StagingArea(staging_dir)
    try:
        # 第一步：清单预演，只判断额度可行性，不读任何磁带数据。
        try:
            plan = preflight(tape_set, read_quota, staging_quota)
        except PlannerQuotaExceeded as exc:
            metrics.failure_kind = KIND_QUOTA
            raise QuotaExceeded(str(exc)) from exc
        metrics.log(
            f"清单预演通过：方案需读带 {plan.reads} 块、暂存峰值 "
            f"{plan.staging_peak} 块（额度上限 {read_quota}/{staging_quota}）"
        )

        # 第二步：按方案实读。
        for planned_pass in plan.passes:
            _record_pass_boundary(metrics, planned_pass.kind, planned_pass.tape)
            for action in planned_pass.actions:
                if metrics.blocks_read >= read_quota:
                    metrics.failure_kind = KIND_QUOTA
                    raise QuotaExceeded(
                        f"读带额度不足：已读 {metrics.blocks_read} 块，"
                        f"上限 {read_quota} 块"
                    )
                data = read_frame(tape_set, action.tape, action.position)
                metrics.blocks_read += 1
                metrics.head_position = action.position + 1
                if read_delay and sleep is not None:
                    sleep(read_delay)

                entry = tape_set.entry(action.index)
                # 读到时核对 SHA-256：无论跳过、丢弃、暂存还是直接写出。
                if sha256_hex(data) != entry.sha256:
                    metrics.bad_block = action.index
                    metrics.failure_kind = KIND_BAD_BLOCK
                    raise BadBlock(
                        f"坏块：镜像块 {action.index}（tape{action.tape} "
                        f"位置 {action.position}）SHA-256 核对失败"
                    )

                if action.action == WRITE:
                    sink.write_block(action.index, data)
                    metrics.log(f"块 {action.index} 读带后直接追加写入镜像")
                    written += 1
                    drain_staged(staging)
                elif action.action == STAGE:
                    # 双保险：实读暂存峰值不能突破额度。
                    if len(staging) >= staging_quota:
                        metrics.failure_kind = KIND_QUOTA
                        raise QuotaExceeded(
                            f"暂存额度不足：暂存峰值将超过上限 "
                            f"{staging_quota} 块"
                        )
                    staging.put(action.index, data)
                    metrics.staging_peak = max(
                        metrics.staging_peak, len(staging)
                    )
                    metrics.log(
                        f"提前读到块 {action.index}，"
                        f"{'落盘' if staging.on_disk else '内存'}暂存"
                        f"（当前暂存 {len(staging)} 块）"
                    )
                elif action.action == DISCARD:
                    metrics.log(
                        f"提前读到块 {action.index}，按规划丢弃，"
                        "之后回卷/换带重读"
                    )
                elif action.action == SKIP:
                    metrics.log(
                        f"顺读经过块 {action.index}（已写出或已暂存），跳过"
                    )
                else:  # pragma: no cover - 规划器动作集合封闭
                    raise RecoveryError(f"未知规划动作: {action.action}")
                if progress is not None:
                    progress(metrics)

        if written != n:
            raise RecoveryError(f"恢复不完整：应写 {n} 块，实写 {written} 块")
        metrics.log(
            f"恢复完成：共读带 {metrics.blocks_read} 块，"
            f"暂存峰值 {metrics.staging_peak} 块"
        )
        return RecoveryResult(
            success=True,
            set_name=tape_set.name,
            metrics=metrics,
            blocks_written=written,
            plan=plan,
        )
    except RecoveryError as exc:
        metrics.log(f"恢复失败：{exc}")
        if metrics.failure_kind is None:
            metrics.failure_kind = KIND_OTHER
        return RecoveryResult(
            success=False,
            set_name=tape_set.name,
            metrics=metrics,
            error=str(exc),
            failure_kind=metrics.failure_kind,
            blocks_written=written,
            plan=plan,
        )
    finally:
        staging.cleanup()
        if not own_sink:
            sink.close()
