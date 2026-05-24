"""
DT 数据集类
===========
存储和加载 DT 离线训练数据。
核心功能：RTG（Return-to-Go）计算、数据序列化/反序列化、数据集合并。
"""

import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 单条 DT 轨迹
# ============================================================================

@dataclass
class DTTrajectory:
    """
    一条可供 DT 训练的轨迹，格式对齐 DT 论文的输入要求。

    DT 输入格式：每步 3 个 token — (RTG_t, s_t, a_{t-1})
    训练目标：预测 a_t

    存储字段：
        states:      List[np.ndarray]  每步的 observation（各自扁平化后的向量）
        actions:     np.ndarray (T,)   每步的 action (0/1)
        rewards:     np.ndarray (T,)   每步的即时 reward
        rtgs:        np.ndarray (T,)   Return-to-Go (从 t 时刻起的累计 reward)
        dones:       np.ndarray (T,)   每步的 done 标志
        timesteps:   np.ndarray (T,)   时间步索引 [0, 1, ..., T-1]
        mask:        np.ndarray (T,)   padding mask (全 1，无 padding 时为全 1)
        success:     bool              episode 是否成功
        metadata:    Dict              额外元信息
    """

    states: List[np.ndarray] = field(default_factory=list)
    raw_states: List[Dict] = field(default_factory=list)  # 原始 dict observation（用于 GIN 编码）
    actions: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    rtgs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    dones: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    timesteps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    success: bool = False
    metadata: Dict = field(default_factory=dict)

    def __len__(self):
        return len(self.actions)

    def compute_rtg(self, gamma: float = 1.0):
        """
        从已有 rewards 反向计算 RTG（Return-to-Go）。

        RTG_t = r_t + r_{t+1} + ... + r_{T-1}
        若 gamma < 1，则 RTG_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...

        Args:
            gamma: 折扣因子（默认 1.0，DT 论文中使用无折扣 RTG）
        """
        T = len(self.rewards)
        if T == 0:
            return

        self.rtgs = np.zeros(T, dtype=np.float32)
        running_rtg = 0.0
        for t in reversed(range(T)):
            running_rtg = self.rewards[t] + gamma * running_rtg
            self.rtgs[t] = float(running_rtg)

        self.timesteps = np.arange(T, dtype=np.int64)
        self.mask = np.ones(T, dtype=bool)

    @staticmethod
    def from_trajectory(traj, gamma: float = 1.0) -> "DTTrajectory":
        """
        从 TrajectoryCollector 产出的 Trajectory 转换为 DTTrajectory。

        Args:
            traj: Trajectory 对象（来自 trajectory_collector.py）
            gamma: RTG 折扣因子
        """
        dt_traj = DTTrajectory()
        dt_traj.states = [DTTrajectory._state_to_array(s) for s in traj.states]
        dt_traj.raw_states = traj.states  # 保留原始 dict observation
        dt_traj.actions = np.array(traj.actions, dtype=np.int64)
        dt_traj.rewards = np.array(traj.rewards, dtype=np.float32)
        dt_traj.dones = np.array(traj.dones, dtype=bool)
        dt_traj.success = traj.success
        dt_traj.metadata = traj.metadata.copy()
        dt_traj.compute_rtg(gamma=gamma)
        return dt_traj

    @staticmethod
    def _state_to_array(state: Dict) -> np.ndarray:
        """
        将 observation dict 扁平化为 1D numpy array。

        observation dict 结构（来自 _StateEncoder）：
            flow_feature:       (F_f,) float32
            link_feature:       (F_l,) float32
            adjacency_matrix:   (2, E) int64
            features_matrix:    (N, F_n) float32
            remain_hops:        (R,) float32

        Returns:
            1D numpy array 包含所有 state 信息拼接
        """
        parts = []

        for key in ["flow_feature", "link_feature", "adjacency_matrix",
                     "features_matrix", "remain_hops"]:
            arr = state[key]
            if isinstance(arr, np.ndarray):
                parts.append(arr.ravel().astype(np.float32))
            elif isinstance(arr, torch.Tensor):
                parts.append(arr.detach().cpu().numpy().ravel().astype(np.float32))
            else:
                parts.append(np.array(arr, dtype=np.float32).ravel())

        return np.concatenate(parts)


# ============================================================================
# 数据集容器
# ============================================================================

class DTDataset:
    """
    DT 离线数据集，管理多条 DTTrajectory。

    功能：
        - 添加轨迹
        - 计算统计量（mean/std of states, RTG target）
        - 保存/加载到磁盘
        - 按 source 分类统计
    """

    def __init__(self, name: str = "dt_dataset"):
        self.name = name
        self.trajectories: List[DTTrajectory] = []

        # 预计算的统计量
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        self.rtg_mean: float = 0.0
        self.rtg_std: float = 0.0

    def add(self, traj: DTTrajectory):
        """添加一条轨迹。"""
        if len(traj) > 0:
            self.trajectories.append(traj)

    def add_from_collector(self, traj, gamma: float = 1.0):
        """从 Trajectory 添加。"""
        dt_traj = DTTrajectory.from_trajectory(traj, gamma=gamma)
        self.add(dt_traj)

    def __len__(self):
        return len(self.trajectories)

    def total_steps(self) -> int:
        """总时间步数。"""
        return sum(len(t) for t in self.trajectories)

    def compute_statistics(self):
        """
        计算所有轨迹的 state mean/std 和 RTG mean/std。
        用于 DT 训练时的 state 归一化和 RTG 条件缩放。
        """
        all_states = []
        all_rtgs = []

        for traj in self.trajectories:
            if len(traj.states) > 0:
                all_states.extend(traj.states)
            if len(traj.rtgs) > 0:
                all_rtgs.append(traj.rtgs[0])  # 只取初始 RTG 作为目标分布

        if len(all_states) > 0:
            stacked = np.stack(all_states, axis=0)
            self.state_mean = stacked.mean(axis=0)
            self.state_std = stacked.std(axis=0)
            # 避免除零
            self.state_std[self.state_std < 1e-6] = 1.0

        if len(all_rtgs) > 0:
            all_rtgs = np.array(all_rtgs)
            self.rtg_mean = float(all_rtgs.mean())
            self.rtg_std = float(all_rtgs.std())
            if self.rtg_std < 1e-6:
                self.rtg_std = 1.0

        logger.info(f"Computed statistics: {len(self.trajectories)} trajectories, "
                    f"{self.total_steps()} steps, "
                    f"RTG mean={self.rtg_mean:.3f}, RTG std={self.rtg_std:.3f}")

    def summary(self) -> Dict:
        """返回数据集摘要。"""
        if len(self.trajectories) == 0:
            return {"num_trajectories": 0}

        successes = sum(1 for t in self.trajectories if t.success)
        lens = [len(t) for t in self.trajectories]
        sources = defaultdict(int)
        for t in self.trajectories:
            src = t.metadata.get("scheduler_type", "unknown")
            sources[src] += 1

        return {
            "name": self.name,
            "num_trajectories": len(self.trajectories),
            "total_steps": self.total_steps(),
            "success_rate": successes / len(self.trajectories),
            "avg_episode_len": np.mean(lens),
            "max_episode_len": max(lens),
            "min_episode_len": min(lens),
            "sources": dict(sources),
            "rtg_mean": self.rtg_mean,
            "rtg_std": self.rtg_std,
        }

    def save(self, filepath: str):
        """
        保存数据集到磁盘。

        Args:
            filepath: 保存路径（不含扩展名，会自动加 .pt 和 _stats.pkl）
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # 保存轨迹数据
        trajectories_data = []
        for traj in self.trajectories:
            trajectories_data.append({
                "states": [torch.from_numpy(s) for s in traj.states],
                "raw_states": traj.raw_states,  # 保留原始 dict（pickle 兼容）
                "actions": torch.from_numpy(traj.actions),
                "rewards": torch.from_numpy(traj.rewards),
                "rtgs": torch.from_numpy(traj.rtgs),
                "dones": torch.from_numpy(traj.dones),
                "timesteps": torch.from_numpy(traj.timesteps),
                "mask": torch.from_numpy(traj.mask),
                "success": traj.success,
                "metadata": traj.metadata,
            })

        torch.save(trajectories_data, f"{filepath}.pt")
        logger.info(f"Saved {len(trajectories_data)} trajectories to {filepath}.pt")

        # 保存统计量
        stats = {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "rtg_mean": self.rtg_mean,
            "rtg_std": self.rtg_std,
            "summary": self.summary(),
        }
        with open(f"{filepath}_stats.pkl", "wb") as f:
            pickle.dump(stats, f)
        logger.info(f"Saved statistics to {filepath}_stats.pkl")

    def load(self, filepath: str):
        """
        从磁盘加载数据集。

        Args:
            filepath: 加载路径（不含扩展名）
        """
        # 加载轨迹
        trajectories_data = torch.load(f"{filepath}.pt", weights_only=False)
        self.trajectories = []
        for data in trajectories_data:
            traj = DTTrajectory()
            traj.states = [s.numpy() for s in data["states"]]
            traj.raw_states = data.get("raw_states", [])  # 兼容旧格式（无 raw_states）
            traj.actions = data["actions"].numpy()
            traj.rewards = data["rewards"].numpy()
            traj.rtgs = data["rtgs"].numpy()
            traj.dones = data["dones"].numpy()
            traj.timesteps = data["timesteps"].numpy()
            traj.mask = data["mask"].numpy()
            traj.success = data["success"]
            traj.metadata = data["metadata"]
            self.trajectories.append(traj)

        logger.info(f"Loaded {len(self.trajectories)} trajectories from {filepath}.pt")

        # 加载统计量
        stats_path = f"{filepath}_stats.pkl"
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
            self.state_mean = stats.get("state_mean")
            self.state_std = stats.get("state_std")
            self.rtg_mean = stats.get("rtg_mean", 0.0)
            self.rtg_std = stats.get("rtg_std", 1.0)
            logger.info(f"Loaded statistics from {stats_path}")

    def merge(self, other: "DTDataset"):
        """合并另一个数据集。"""
        self.trajectories.extend(other.trajectories)
        logger.info(f"Merged dataset: now {len(self.trajectories)} trajectories")


# ============================================================================
# 便捷函数
# ============================================================================

def build_dataset(
    trajectories: List,
    name: str = "dt_dataset",
    gamma: float = 1.0,
    compute_stats: bool = True,
) -> DTDataset:
    """
    从 Trajectory 列表快速构建 DTDataset。

    Args:
        trajectories: Trajectory 对象列表
        name: 数据集名称
        gamma: RTG 折扣因子
        compute_stats: 是否计算统计量
    """
    dataset = DTDataset(name=name)
    for traj in trajectories:
        dataset.add_from_collector(traj, gamma=gamma)
    if compute_stats and len(dataset) > 0:
        dataset.compute_statistics()
    return dataset
