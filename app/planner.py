"""清单驱动的恢复规划器：开带前只凭 manifest 判断额度是否可行。

规划器 **不读取任何磁带数据**，只用清单里每块的（块号、磁带、帧位置）
推演机械动作与额度占用，输出一个可执行的顺读方案（若干个 pass）：

* 磁带只能从头顺读；换带或回卷后磁头回到第 0 帧，同一盘带上磁头只向前移。
* 顺读路径上跳过的帧照样计读带量；回卷/换带后重读的帧重复计数。
* 提前读到、暂时不能按块号写出的块，可以暂存（占暂存额度），也可以
  **丢弃**——等它成为目标块后再回卷/换带重读（占读带额度）。
* 镜像只按块号 0、1、2…… 顺序追加；已暂存的连续块在目标块写出后批量排空。

状态空间很小（最多 12 块、两盘带）。DFS 枚举每个 pass 内“暂存/丢弃”的
选择，并为每个记忆化状态保留（读带量, 暂存峰值）的**帕累托前沿**——
前沿与具体额度无关，因此一次推演即可回答任意双额度组合的可行性，
额度只在顶层按方案指标过滤。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .storage import TapeSet

# 单帧动作
STAGE = "stage"        # 提前读到：保留进暂存区
DISCARD = "discard"    # 提前读到：丢弃，之后回卷/换带重读
SKIP = "skip"          # 读到已写出或已暂存的块：不保留（读带量仍计数）
WRITE = "write"        # 正是当前目标块：核对后直接追加写镜像

# pass 类型
MOUNT = "mount"        # 第一次挂带
SWITCH = "switch"      # 换带：磁头回到带头
REWIND = "rewind"      # 同带回卷：磁头回到带头
CONTINUE = "continue"  # 同带磁头自然前移


class PlannerError(RuntimeError):
    """规划阶段失败。"""


class QuotaExceeded(PlannerError):
    """清单预演判定双额度下无可行解（尚未读带）。"""


@dataclass(frozen=True)
class FrameAction:
    """一次读帧动作：读 tape 上 position 帧（内容为镜像块 index）后如何处置。"""

    tape: int
    position: int
    index: int
    action: str


@dataclass(frozen=True)
class PlannedPass:
    """一次顺读会话：从 start 帧顺序读到 end 帧（含），磁头最终停在 end+1。"""

    tape: int
    start: int
    end: int
    kind: str
    target: int
    actions: tuple[FrameAction, ...]


@dataclass(frozen=True)
class Plan:
    """完整恢复方案及其确定性额度指标。"""

    passes: tuple[PlannedPass, ...]
    reads: int
    staging_peak: int
    mounts: int
    switches: int
    rewinds: int


# (读带量, 暂存峰值, 后续 pass) 三元组列表的类型别名。
_Frontier = list[tuple[int, int, tuple[PlannedPass, ...]]]


class _RecoveryPlanner:
    """无额度约束的全策略推演：每个状态只算一次，结果跨额度复用。"""

    def __init__(self, tape_set: TapeSet) -> None:
        self.tape_set = tape_set
        self.n = tape_set.block_count
        self._at: dict[tuple[int, int], Any] = {
            (e.tape, e.position): e for e in tape_set.blocks
        }
        self._memo: dict[tuple, _Frontier] = {}

    @staticmethod
    def _drain(written: int, staged: set[int]) -> int:
        """目标块写出后，把暂存区里连续的后续块按块号排空（不产生读带量）。"""
        while written in staged:
            staged.discard(written)
            written += 1
        return written

    def solve(
        self,
        written: int,
        staged_frozen: frozenset[int],
        current_tape: int | None,
        head: int,
    ) -> _Frontier:
        """返回从当前状态出发全部不被支配的（读带量, 暂存峰值, 后续 pass）。"""

        staged = set(staged_frozen)
        written = self._drain(written, staged)
        if written == self.n:
            return [(0, 0, ())]

        key = (written, tuple(sorted(staged)), current_tape, head)
        cached = self._memo.get(key)
        if cached is not None:
            return cached

        target = self.tape_set.entry(written)
        if current_tape is None:
            tape, start, kind = target.tape, 0, MOUNT
        elif current_tape == target.tape:
            if target.position >= head:
                # 目标帧在磁头前方：磁头自然前移，无需回卷。
                tape, start, kind = current_tape, head, CONTINUE
            else:
                # 目标帧已被磁头越过：回卷后从头顺读。
                tape, start, kind = current_tape, 0, REWIND
        else:
            # 目标在另一盘带上：换带后从带头顺读。
            tape, start, kind = target.tape, 0, SWITCH
        end = target.position

        frames = [self._at[(tape, pos)] for pos in range(start, end + 1)]
        # 末帧必然是当前目标块；前面的帧只能是跳过项或可选择暂存/丢弃的未来块。
        fixed_actions: list[FrameAction | None] = []
        for entry in frames[:-1]:
            if entry.index < written or entry.index in staged:
                fixed_actions.append(
                    FrameAction(tape, entry.position, entry.index, SKIP)
                )
            else:
                fixed_actions.append(None)

        acc: _Frontier = []
        entry_occupancy = len(staged)

        def enumerate_frames(
            idx: int,
            occupancy: int,
            peak: int,
            staged_now: set[int],
            actions: list[FrameAction],
        ) -> None:
            if idx == len(frames) - 1:
                # 读到目标帧：直接追加写出，再排空暂存区里的连续块。
                all_actions = tuple(
                    actions + [FrameAction(tape, end, written, WRITE)]
                )
                next_staged = set(staged_now)
                next_written = self._drain(written + 1, next_staged)
                for suffix_reads, suffix_peak, suffix_passes in self.solve(
                    next_written, frozenset(next_staged), tape, end + 1
                ):
                    total_reads = (end - start + 1) + suffix_reads
                    # 峰值取本 pass 内峰值与后续 pass 峰值的较大者。
                    total_peak = max(peak, suffix_peak)
                    planned = PlannedPass(
                        tape=tape,
                        start=start,
                        end=end,
                        kind=kind,
                        target=written,
                        actions=all_actions,
                    )
                    acc.append(
                        (total_reads, total_peak, (planned,) + suffix_passes)
                    )
                return

            entry = frames[idx]
            fixed = fixed_actions[idx]
            if fixed is not None:
                enumerate_frames(
                    idx + 1, occupancy, peak, staged_now, actions + [fixed]
                )
                return

            # 选择一：丢弃提前块，不占暂存，之后需要时回卷/换带重读。
            enumerate_frames(
                idx + 1,
                occupancy,
                peak,
                staged_now,
                actions + [FrameAction(tape, entry.position, entry.index, DISCARD)],
            )

            # 选择二：暂存提前块（峰值由推演自然记录，额度在顶层过滤）。
            with_block = set(staged_now)
            with_block.add(entry.index)
            enumerate_frames(
                idx + 1,
                occupancy + 1,
                max(peak, occupancy + 1),
                with_block,
                actions + [FrameAction(tape, entry.position, entry.index, STAGE)],
            )

        enumerate_frames(0, entry_occupancy, entry_occupancy, set(staged), [])
        frontier = _pareto(acc)
        self._memo[key] = frontier
        return frontier


def _pareto(solutions: _Frontier) -> _Frontier:
    """保留不被任何其他方案同时压住读带量与暂存峰值的方案。"""

    unique = list({(reads, peak, passes) for reads, peak, passes in solutions})
    result = []
    for cand in unique:
        dominated = False
        for other in unique:
            if other is cand:
                continue
            if (
                other[0] <= cand[0]
                and other[1] <= cand[1]
                and (other[0] < cand[0] or other[1] < cand[1])
            ):
                dominated = True
                break
        if not dominated:
            result.append(cand)
    return result


def _to_plan(best: tuple[int, int, tuple[PlannedPass, ...]]) -> Plan:
    reads, peak, passes = best
    mounts = sum(1 for pass_ in passes if pass_.kind == MOUNT)
    switches = sum(1 for pass_ in passes if pass_.kind == SWITCH)
    rewinds = sum(1 for pass_ in passes if pass_.kind == REWIND)
    # 每个动作对应一帧实读，读带量必须与逐 pass 推算一致。
    assert reads == sum(len(pass_.actions) for pass_ in passes)
    return Plan(
        passes=passes,
        reads=reads,
        staging_peak=peak,
        mounts=mounts,
        switches=switches,
        rewinds=rewinds,
    )


def explore(tape_set: TapeSet) -> _Frontier:
    """返回带卷全部非支配恢复方案（读带量, 暂存峰值, pass 序列），不读带。"""

    return _RecoveryPlanner(tape_set).solve(0, frozenset(), None, 0)


class FeasibilityCache:
    """同一带卷多组额度查询时复用一次全策略推演的帕累托前沿。

    演练规模（≤12 块）下一次推演为亚秒级；缓存让逐额度网格查询从
    “每次重算”降为对小列表的线性过滤。
    """

    def __init__(self, tape_set: TapeSet) -> None:
        self.tape_set = tape_set
        self._frontier: _Frontier | None = None

    @property
    def frontier(self) -> _Frontier:
        if self._frontier is None:
            self._frontier = explore(self.tape_set)
        return self._frontier

    def plan(
        self,
        read_quota: int,
        staging_quota: int,
        *,
        objective: tuple[str, str] = ("reads", "peak"),
    ) -> Plan | None:
        if read_quota <= 0 or staging_quota < 0:
            raise ValueError("读带额度必须为正整数，暂存额度必须为非负整数")
        feasible = [
            item
            for item in self.frontier
            if item[0] <= read_quota and item[1] <= staging_quota
        ]
        if not feasible:
            return None
        if objective == ("reads", "peak"):
            best = min(feasible, key=lambda item: (item[0], item[1]))
        elif objective == ("peak", "reads"):
            best = min(feasible, key=lambda item: (item[1], item[0]))
        else:
            raise ValueError(f"未知优化目标: {objective}")
        return _to_plan(best)


def plan_recovery(
    tape_set: TapeSet,
    read_quota: int,
    staging_quota: int,
    *,
    objective: tuple[str, str] = ("reads", "peak"),
) -> Plan | None:
    """清单预演：返回双额度内的最优方案；无解返回 None。全程不读带。

    objective=("reads","peak") 时先最小化读带量（再压低暂存峰值）；
    ("peak","reads") 反过来。同一带卷连续查询多组额度时请直接复用
    :class:`FeasibilityCache`，避免重复推演。
    """

    return FeasibilityCache(tape_set).plan(
        read_quota, staging_quota, objective=objective
    )


def preflight(tape_set: TapeSet, read_quota: int, staging_quota: int) -> Plan:
    """开带前的额度可行性判定；无解抛 QuotaExceeded（保证未读带）。"""

    if read_quota <= 0 or staging_quota < 0:
        raise ValueError("读带额度必须为正整数，暂存额度必须为非负整数")

    cache = FeasibilityCache(tape_set)
    plan = cache.plan(read_quota, staging_quota)
    if plan is not None:
        return plan

    frontier = cache.frontier
    # 理论最省读带方案，以及当前暂存额度下的最省读带方案。
    unlimited = _to_plan(min(frontier, key=lambda item: (item[0], item[1])))
    under_staging_items = [item for item in frontier if item[1] <= staging_quota]
    messages = [
        f"清单预演判定读带 {read_quota} 块 / 暂存 {staging_quota} 块的额度组合"
        "无可行解（本次未读带）"
    ]
    if under_staging_items:
        under_staging = _to_plan(
            min(under_staging_items, key=lambda item: item[0])
        )
        if under_staging.reads > read_quota:
            messages.append(
                f"暂存额度维持 {staging_quota} 块时，最少需要读带 "
                f"{under_staging.reads} 块"
            )
    messages.append(
        f"该带卷理论最少读带 {unlimited.reads} 块（对应暂存峰值 "
        f"{unlimited.staging_peak} 块）；放宽暂存额度可减少回卷重读"
    )
    raise QuotaExceeded("；".join(messages))
