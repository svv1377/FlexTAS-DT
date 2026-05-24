"""
DT 离线数据集模块
=================
提供轨迹收集、调度结果→per-hop 轨迹转换、数据集存储/加载等功能。
用于为 Decision Transformer 构建离线训练数据。

目录结构：
    src/app/dt_dataset/
    ├── __init__.py
    ├── trajectory_collector.py  # 轨迹收集器（env 交互 + 调度器结果回放）
    ├── dt_dataset.py            # 数据集类（存储/加载/RTG 计算）
    └── build_dataset.py         # 主构建脚本
"""

from src.app.dt_dataset.trajectory_collector import (
    TrajectoryCollector,
    SchedulerTrajectoryReplay,
)
from src.app.dt_dataset.dt_dataset import (
    DTTrajectory,
    DTDataset,
    build_dataset,
)

__all__ = [
    "TrajectoryCollector",
    "SchedulerTrajectoryReplay",
    "DTTrajectory",
    "DTDataset",
    "build_dataset",
]
