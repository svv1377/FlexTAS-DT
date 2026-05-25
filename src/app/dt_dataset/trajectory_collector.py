"""
轨迹收集器
==========
提供两种轨迹收集方式：
1. TrajectoryCollector — 在 NetEnv 交互过程中实时收集
2. SchedulerTrajectoryReplay — 从调度器最终结果反向回放生成 per-hop 轨迹
"""

import copy
import logging
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from src.env.env import NetEnv, SchedulingError
from src.network.net import Flow, Link, Network, Net
from src.app.scheduler import BaseScheduler, ScheduleRes
from src.lib.operation import Operation, check_operation_isolation

logger = logging.getLogger(__name__)


# ============================================================================
# 轨迹数据结构
# ============================================================================

class Trajectory:
    """单条 episode 的完整轨迹（per-hop 序列）。"""

    __slots__ = (
        "states",       # List[Dict]: 每步的 observation dict
        "actions",      # List[int]: 每步的 action (0 或 1)
        "rewards",      # List[float]: 每步的 reward
        "dones",        # List[bool]: 每步的 done
        "success",      # bool: episode 是否成功
        "num_flows",    # int: 已成功调度的流数
        "metadata",     # Dict: 额外元信息（拓扑、调度器类型等）
    )

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.success = False
        self.num_flows = 0
        self.metadata = {}

    def add_step(self, state: Dict, action: int, reward: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

    def is_valid(self) -> bool:
        """检查轨迹长度一致性。"""
        return len(self.states) == len(self.actions) == len(self.rewards) == len(self.dones)

    def to_dict(self) -> Dict:
        """转换为可序列化的 dict。"""
        return {
            "states": self.states,
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=bool),
            "success": self.success,
            "num_flows": self.num_flows,
            "metadata": self.metadata,
        }


# ============================================================================
# 方式 1：实时轨迹收集器（用于 PPO env 交互 / DT online fine-tune）
# ============================================================================

class TrajectoryCollector:
    """
    在 NetEnv 交互过程中实时收集 (state, action, reward, done) 轨迹。

    使用方式：
        collector = TrajectoryCollector(env)
        obs, _ = env.reset()
        collector.start_episode()
        while True:
            action = model.predict(obs)
            collector.record_state(obs)   # 在 step 之前记录 state
            obs, reward, done, truncated, info = env.step(action)
            collector.record_step(action, reward, done)
            if done:
                trajectory = collector.finish_episode(info)
                break
    """

    def __init__(self, env: NetEnv):
        self.env = env
        self._current_traj: Optional[Trajectory] = None

    def start_episode(self):
        """开始一个新的 episode。"""
        self._current_traj = Trajectory()

    def record_state(self, obs: Dict):
        """在 env.step() 之前，记录当前 observation。"""
        if self._current_traj is None:
            raise RuntimeError("Call start_episode() before record_state()")
        # deep copy 防止 obs 被后续 step 修改
        self._current_traj.states.append(copy.deepcopy(obs))

    def record_step(self, action: int, reward: float, done: bool):
        """在 env.step() 之后，记录 action、reward、done。"""
        if self._current_traj is None:
            raise RuntimeError("Call start_episode() before record_step()")
        self._current_traj.actions.append(int(action))
        self._current_traj.rewards.append(float(reward))
        self._current_traj.dones.append(bool(done))

    def finish_episode(self, info: Dict) -> Trajectory:
        """结束当前 episode，返回完整轨迹。"""
        traj = self._current_traj
        self._current_traj = None
        traj.success = info.get("success", False)
        traj.num_flows = self.env.flow_index
        traj.metadata["num_flows_total"] = self.env.num_flows
        return traj


# ============================================================================
# 方式 2：调度器结果回放（核心：将调度结果反推为 per-hop 轨迹）
# ============================================================================

class SchedulerTrajectoryReplay:
    """
    从调度器的最终结果 (ScheduleRes) 反向推导 per-hop 的 (state, action, reward) 轨迹。

    核心思路：
        调度器（SMT / Tabu / TimeTabling）输出每条 link 上的 operation 列表，
        但 DT 需要 per-hop 决策序列。本类通过"重放环境"的方式，
        在每一步根据调度结果判断 action（gating/no-gating），
        并调用 StateEncoder 生成 state、计算 reward。

    使用方式：
        replay = SchedulerTrajectoryReplay(network, scheduler.get_res())
        trajectory = replay.replay()
    """

    def __init__(self, network: Network, schedule_res: ScheduleRes):
        """
        Args:
            network: 原始网络（含 graph + flows）
            schedule_res: 调度结果，格式 {Link: [(Flow, Operation), ...]}
        """
        self.network = network
        self.schedule_res = schedule_res
        self.graph = network.graph

        # 复制 flows（保持与调度器相同的顺序）
        self.flows = list(network.flows)

        # 构建快速查找索引：(flow_id, link_id) → gating bool
        self._schedule_index: Dict[Tuple[str, str], bool] = {}
        for link, operations in schedule_res.items():
            for flow, operation in operations:
                gating = operation.gating_time is not None
                self._schedule_index[(flow.flow_id, link.link_id)] = gating

    def _fresh_env_without_shuffle(self) -> NetEnv:
        """Create a clean NetEnv while preserving the scheduler flow order."""
        env = NetEnv(self.network)
        env.flows = list(self.flows)
        env.flow_index = 0
        env.temp_operations = []
        env.links_operations.clear()
        env.links_gcl.clear()
        env.reward = 0
        env.last_action = None
        return env

    def replay(self) -> Trajectory:
        """
        重放调度过程，生成 per-hop 轨迹。

        注意：不经过 env.reset()/env.step() 的完整路径，而是直接
        操作 env 内部状态来逐步收集 state/action/reward。
        这是因为调度器的 decisions 可能与 env 的 constraint check
        有微小差异（env 在 step 中有额外的 jitter/gcl/period 检查），
        而调度器已验证了这些约束。

        Returns:
            Trajectory: 包含完整的 (state, action, reward, done) 序列
        """
        # 创建新 env，使用 network 的原始 flow 顺序（不 shuffle）
        env = self._fresh_env_without_shuffle()

        traj = Trajectory()
        traj.metadata = {
            "topo": "unknown",
            "num_flows": len(self.flows),
            "scheduler_type": "unknown",
        }

        # --- 按 flow 顺序、per-hop 逐步回放 ---
        for flow_idx, flow in enumerate(self.flows):
            env.flow_index = flow_idx
            path = flow.path

            for hop_idx, link_id in enumerate(path):
                link = env.link_dict[link_id]

                # 1. 生成当前 state
                state = env._generate_state()

                # 2. 从调度结果反推 action
                key = (flow.flow_id, link_id)
                if key not in self._schedule_index:
                    logger.warning(f"Missing schedule entry for flow={flow.flow_id}, "
                                   f"link={link_id}. Marking trajectory as failed.")
                    traj.add_step(state, action=0, reward=0.0, done=True)
                    traj.success = False
                    traj.num_flows = flow_idx
                    return traj

                gating = self._schedule_index[key]
                action = 1 if gating else 0

                # 3. 计算 reward（模拟 env.step 中的 reward 计算）
                reward = self._compute_reward(env, flow, link, hop_idx, path,
                                              gating)

                # 4. 更新 env 内部状态（模拟 step 成功的路径）
                self._advance_env_state(env, flow, link, path, hop_idx, gating)

                # 5. 记录
                done = (flow_idx == len(self.flows) - 1 and hop_idx == len(path) - 1)
                traj.add_step(state, action, reward, done)

        traj.success = True
        traj.num_flows = len(self.flows)
        return traj

    def validate_trajectory(
        self,
        traj: Trajectory,
        reward_tol: float = 1e-5,
        state_atol: float = 1e-5,
        state_rtol: float = 1e-5,
        compare_states: bool = True,
    ) -> Tuple[bool, str]:
        """
        Replay the inferred action sequence through the real NetEnv.step().

        This rejects trajectories whose approximate replay state/reward/done
        sequence diverges from the environment that DTScheduler will use at
        inference time.
        """
        if not traj.is_valid():
            return False, "trajectory fields have inconsistent lengths"
        if len(traj) == 0:
            return False, "empty trajectory"

        env = self._fresh_env_without_shuffle()
        done = False
        info = {"success": False}

        for step_idx, (expected_state, action, expected_reward, expected_done) in enumerate(
            zip(traj.states, traj.actions, traj.rewards, traj.dones)
        ):
            if done:
                return False, f"trajectory has extra steps after done at step {step_idx}"

            try:
                actual_state = env._generate_state()
            except Exception as exc:
                return False, f"failed to generate env state at step {step_idx}: {exc}"

            if compare_states and not self._states_close(
                actual_state, expected_state, atol=state_atol, rtol=state_rtol
            ):
                return False, f"state mismatch at step {step_idx}"

            try:
                _, reward, done, _, info = env.step(int(action))
            except Exception as exc:
                return False, f"env.step raised at step {step_idx}: {exc}"

            if abs(float(reward) - float(expected_reward)) > reward_tol:
                return (
                    False,
                    f"reward mismatch at step {step_idx}: "
                    f"expected {expected_reward}, got {reward}",
                )

            if bool(done) != bool(expected_done):
                return (
                    False,
                    f"done mismatch at step {step_idx}: "
                    f"expected {expected_done}, got {done}",
                )

        if not done:
            return False, "action sequence ended before env reached done"

        success = bool(info.get("success", False))
        if success != bool(traj.success):
            return False, f"success mismatch: expected {traj.success}, got {success}"

        return True, ""

    @staticmethod
    def _states_close(lhs: Dict, rhs: Dict, atol: float, rtol: float) -> bool:
        if lhs.keys() != rhs.keys():
            return False

        for key in lhs:
            lhs_arr = np.asarray(lhs[key])
            rhs_arr = np.asarray(rhs[key])
            if lhs_arr.shape != rhs_arr.shape:
                return False
            if np.issubdtype(lhs_arr.dtype, np.integer) or np.issubdtype(rhs_arr.dtype, np.integer):
                if not np.array_equal(lhs_arr, rhs_arr):
                    return False
            elif not np.allclose(lhs_arr, rhs_arr, atol=atol, rtol=rtol):
                return False

        return True

    def _compute_reward(
        self,
        env: NetEnv,
        flow: Flow,
        link: Link,
        hop_idx: int,
        path: List[str],
        gating: bool,
    ) -> float:
        """模拟 env.step() 中的 reward 计算逻辑。"""
        trans_time = link.transmission_time(flow.payload)

        if gating:
            wait_time = 0
        else:
            wait_time = link.interference_time()

        # GCL 惩罚
        gcl_added = 0
        if gating:
            try:
                old_gcl = env.links_gcl[link].gcl_length
                env.add_gating(link, flow.period)
                new_gcl = env.links_gcl[link].gcl_length
                gcl_added = new_gcl - old_gcl
            except RuntimeError:
                pass  # 调度器已验证过，不应发生

        reward_gcl = 0 - env.alpha * gcl_added / link.gcl_capacity if link.gcl_capacity != 0 else 0
        reward_time = 0 - env.beta * wait_time / flow.e2e_delay
        reward = 1 + reward_gcl + reward_time

        # 里程碑奖励（与 env 保持一致）
        if hop_idx == len(path) - 1:
            flow_index_after = env.flow_index + 1  # 当前 flow 完成后
            if flow_index_after % max(1, int(env.num_flows * 0.1)) == 0:
                reward += env.gamma * ((flow_index_after / env.num_flows) ** 2)

        return reward

    def _advance_env_state(
        self,
        env: NetEnv,
        flow: Flow,
        link: Link,
        path: List[str],
        hop_idx: int,
        gating: bool,
    ):
        """推进 env 内部状态（模拟一次成功的 step 后的状态变更）。"""
        # 构造一个简单的 operation（仅用于记录状态，具体时间不重要）
        trans_time = link.transmission_time(flow.payload)

        # 简化处理：用近似值推进 env 状态
        # 获取上一个 hop 的 temp_operations 来计算 enqueue 时间
        earliest_enqueue = 0
        latest_enqueue = 0
        if len(env.temp_operations) > 0:
            last_link_op, last_oper = env.temp_operations[-1]
            last_gating = env.last_action if env.last_action is not None else False
            if last_gating:
                earliest_deq = last_oper.gating_time
                latest_deq = last_oper.gating_time
            else:
                earliest_deq = last_oper.earliest_time
                latest_deq = last_oper.latest_time

            earliest_enqueue = (earliest_deq + trans_time + Net.DELAY_PROP
                                - Net.SYNC_PRECISION + Net.DELAY_PROC_MIN)
            latest_enqueue = (latest_deq + trans_time + Net.DELAY_PROP
                              + Net.SYNC_PRECISION + Net.DELAY_PROC_MAX)

        if gating:
            wait_time = 0
        else:
            wait_time = link.interference_time()

        latest_dequeue = latest_enqueue + wait_time
        end_time = latest_dequeue + trans_time

        operation = Operation(earliest_enqueue, None, latest_dequeue, end_time)
        if gating:
            operation.gating_time = latest_dequeue

        env.temp_operations.append((link, operation))

        # 检查是否是 flow 的最后一跳
        if hop_idx == len(path) - 1:
            for lk, oper in env.temp_operations:
                env.links_operations[lk].append((flow, oper))
            env.temp_operations = []
            env.flow_index += 1

        env.last_action = gating
