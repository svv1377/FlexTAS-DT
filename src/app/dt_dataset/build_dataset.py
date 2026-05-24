#!/usr/bin/env python3
"""
离线数据集构建脚本
====================
按 Week 1-2 路线图，从多种调度器生成 DT 离线训练数据。

用法：
    # 生成 Tabu（时间表）轨迹
    PYTHONPATH=. python src/app/dt_dataset/build_dataset.py \
        --topo CEV --num_flows 50 --num_episodes 500 \
        --scheduler tabu_all_gate,tabu_no_gate,tabu_random_gate \
        --output out/dt_data/tabu

    # 生成 SMT 轨迹
    PYTHONPATH=. python src/app/dt_dataset/build_dataset.py \
        --topo CEV --num_flows 50 --num_episodes 100 \
        --scheduler smt --output out/dt_data/smt --timeout 300

    # 生成全部数据并合并
    PYTHONPATH=. python src/app/dt_dataset/build_dataset.py \
        --topo CEV --num_flows 50 --num_episodes 500 \
        --scheduler tabu_all_gate,tabu_no_gate,tabu_random_gate,smt \
        --output out/dt_data/combined --timeout 60

    # 多拓扑生成
    PYTHONPATH=. python src/app/dt_dataset/build_dataset.py \
        --topo CEV,RRG,ERG,BAG --num_flows 50 --num_episodes 200 \
        --scheduler tabu_all_gate,smt --output out/dt_data/multi_topo
"""

import argparse
import copy
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np

from definitions import OUT_DIR, ROOT_DIR
from src.app.dt_dataset.trajectory_collector import (
    Trajectory,
    TrajectoryCollector,
    SchedulerTrajectoryReplay,
)
from src.app.dt_dataset.dt_dataset import DTDataset, DTTrajectory
from src.app.no_wait_tabu_scheduler import TimeTablingScheduler, GatingStrategy
from src.app.smt_scheduler import SmtScheduler
from src.app.scheduler import BaseScheduler, ScheduleRes
from src.env.env import NetEnv
from src.lib.log_config import log_config
from src.lib.timing_decorator import timing_decorator
from src.network.net import (
    FlowGenerator,
    Network,
    generate_graph,
    generate_flows,
)

# ============================================================================
# 配置
# ============================================================================

DATA_DIR = os.path.join(OUT_DIR, "dt_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 支持的调度器
SCHEDULER_REGISTRY = {
    "tabu_all_gate": {
        "class": TimeTablingScheduler,
        "kwargs": {"gating_strategy": GatingStrategy.AllGate},
        "label": "tabu_all_gate",
    },
    "tabu_no_gate": {
        "class": TimeTablingScheduler,
        "kwargs": {"gating_strategy": GatingStrategy.NoGate},
        "label": "tabu_no_gate",
    },
    "tabu_random_gate": {
        "class": TimeTablingScheduler,
        "kwargs": {"gating_strategy": GatingStrategy.RandomGate},
        "label": "tabu_random_gate",
    },
    "smt": {
        "class": SmtScheduler,
        "kwargs": {},
        "label": "smt",
    },
}

logger = logging.getLogger(__name__)


# ============================================================================
# 核心函数：从调度器生成轨迹
# ============================================================================

def generate_network(topo: str, num_flows: int, seed: int,
                     link_rate: int = 100) -> Network:
    """生成一个随机网络实例。"""
    graph = generate_graph(topo, link_rate)
    flow_generator = FlowGenerator(graph, seed=seed)
    flows = flow_generator(num_flows)
    return Network(graph, flows)


def run_scheduler_and_get_result(
    network: Network,
    scheduler_name: str,
    timeout_s: int = 60,
) -> Tuple[bool, Optional[ScheduleRes], str]:
    """
    运行调度器并获取结果。

    Returns:
        (success, schedule_res, error_msg)
    """
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. "
                         f"Available: {list(SCHEDULER_REGISTRY.keys())}")

    cfg = SCHEDULER_REGISTRY[scheduler_name]
    scheduler_cls = cfg["class"]
    kwargs = cfg["kwargs"].copy()
    kwargs["timeout_s"] = timeout_s

    scheduler = scheduler_cls(network, **kwargs)

    try:
        success = scheduler.schedule()
        if success:
            return True, scheduler.get_res(), ""
        else:
            return False, None, "scheduler returned False"
    except Exception as e:
        return False, None, str(e)


def scheduler_to_trajectory(
    network: Network,
    schedule_res: ScheduleRes,
    scheduler_name: str,
    topo: str,
) -> Optional[Trajectory]:
    """
    将调度结果转换为 per-hop 轨迹。

    Args:
        network: 网络实例
        schedule_res: 调度结果
        scheduler_name: 调度器名称
        topo: 拓扑名称

    Returns:
        Trajectory 或 None（转换失败时）
    """
    try:
        replay = SchedulerTrajectoryReplay(network, schedule_res)
        traj = replay.replay()
        traj.metadata["scheduler_type"] = scheduler_name
        traj.metadata["topo"] = topo
        return traj
    except Exception as e:
        logger.warning(f"Failed to convert schedule result to trajectory: {e}")
        return None


def generate_trajectories_from_scheduler(
    topo: str,
    num_flows: int,
    num_episodes: int,
    scheduler_name: str,
    timeout_s: int = 60,
    link_rate: int = 100,
    seed_start: int = 0,
) -> List[Trajectory]:
    """
    使用指定调度器批量生成轨迹。

    对于每个 episode：
        1. 生成一个新的随机网络实例
        2. 运行调度器
        3. 将调度结果转换为轨迹
    """
    trajectories = []
    success_count = 0
    fail_count = 0

    logger.info(f"[{scheduler_name}] Starting: target={num_episodes} episodes, "
                f"topo={topo}, flows={num_flows}")

    for i in range(num_episodes):
        seed = seed_start + i
        network = generate_network(topo, num_flows, seed, link_rate)

        # 运行调度器
        ok, schedule_res, err_msg = run_scheduler_and_get_result(
            network, scheduler_name, timeout_s
        )

        if not ok or schedule_res is None:
            fail_count += 1
            if i % 50 == 0:
                logger.info(f"[{scheduler_name}] Episode {i}: FAIL ({err_msg})")
            continue

        # 转换为轨迹
        traj = scheduler_to_trajectory(
            network, schedule_res, scheduler_name, topo
        )

        if traj is not None:
            trajectories.append(traj)
            success_count += 1
        else:
            fail_count += 1

        if (i + 1) % 50 == 0:
            logger.info(f"[{scheduler_name}] Progress: {i+1}/{num_episodes} "
                        f"(success={success_count}, fail={fail_count})")

    logger.info(f"[{scheduler_name}] Done: {success_count} trajectories generated, "
                f"{fail_count} failed")
    return trajectories


# ============================================================================
# PPO 轨迹收集（从已训练模型）
# ============================================================================

def generate_trajectories_from_env(
    topo: str,
    num_flows: int,
    num_episodes: int,
    scheduler_name: str = "env_random",
    link_rate: int = 100,
    seed_start: int = 0,
) -> List[Trajectory]:
    """
    通过直接运行 NetEnv（随机策略）生成轨迹。
    用于收集"背景数据"——包含成功和失败的 episode。

    注意：此函数使用随机策略。若有训练好的 PPO 模型，
    可替换为 model.predict()。
    """
    trajectories = []

    logger.info(f"[{scheduler_name}] Starting env-based collection: "
                f"target={num_episodes} episodes, topo={topo}, flows={num_flows}")

    for i in range(num_episodes):
        seed = seed_start + i
        network = generate_network(topo, num_flows, seed, link_rate)

        env = NetEnv(network)

        # 使用 TrajectoryCollector（但关闭 shuffle，保持确定性）
        collector = TrajectoryCollector(env)
        obs, _ = env.reset()
        collector.start_episode()

        done = False
        while not done:
            collector.record_state(obs)
            # 随机策略（可替换为模型推理）
            action = random.randint(0, 1)
            obs, reward, done, truncated, info = env.step(action)
            collector.record_step(action, reward, done)

        traj = collector.finish_episode(info)
        traj.metadata["scheduler_type"] = scheduler_name
        traj.metadata["topo"] = topo
        trajectories.append(traj)

        if (i + 1) % 50 == 0:
            logger.info(f"[{scheduler_name}] Progress: {i+1}/{num_episodes}")

    successes = sum(1 for t in trajectories if t.success)
    logger.info(f"[{scheduler_name}] Done: {len(trajectories)} trajectories "
                f"(success={successes})")
    return trajectories


# ============================================================================
# 数据集构建主流程
# ============================================================================

@timing_decorator(logging.info)
def build_offline_dataset(
    topo: str,
    num_flows: int,
    num_episodes: int,
    scheduler_names: List[str],
    output_path: str,
    timeout_s: int = 60,
    link_rate: int = 100,
    gamma: float = 1.0,
    include_env_random: bool = False,
    env_episodes: int = 200,
) -> DTDataset:
    """
    构建完整的离线数据集。

    Args:
        topo: 拓扑名称 (CEV/RRG/ERG/BAG)
        num_flows: 流数量
        num_episodes: 每个调度器的目标轨迹数
        scheduler_names: 调度器列表
        output_path: 输出路径
        timeout_s: 调度器超时（秒）
        link_rate: 链路速率
        gamma: RTG 折扣因子
        include_env_random: 是否包含环境随机轨迹
        env_episodes: 环境随机轨迹数

    Returns:
        DTDataset
    """
    all_trajectories = []

    # 阶段 1：Tabu（时间表）调度器 —— 主力数据源
    for sched_name in scheduler_names:
        if sched_name == "smt":
            continue  # SMT 在阶段 2 单独处理

        logger.info(f"{'='*60}")
        logger.info(f"Stage 1: Generating Tabu trajectories [{sched_name}]")
        logger.info(f"{'='*60}")

        trajs = generate_trajectories_from_scheduler(
            topo=topo,
            num_flows=num_flows,
            num_episodes=num_episodes,
            scheduler_name=sched_name,
            timeout_s=timeout_s,
            link_rate=link_rate,
            seed_start=hash(sched_name) % 10000,
        )
        all_trajectories.extend(trajs)
        logger.info(f"Collected {len(trajs)} trajectories from {sched_name}")

    # 阶段 2：SMT 求解器 —— 高质量数据
    if "smt" in scheduler_names:
        logger.info(f"{'='*60}")
        logger.info(f"Stage 2: Generating SMT trajectories")
        logger.info(f"{'='*60}")

        smt_episodes = min(num_episodes, 200)  # SMT 数量限制
        logger.info(f"SMT target: {smt_episodes} (limited due to solver cost)")

        trajs = generate_trajectories_from_scheduler(
            topo=topo,
            num_flows=num_flows,
            num_episodes=smt_episodes,
            scheduler_name="smt",
            timeout_s=timeout_s,
            link_rate=link_rate,
            seed_start=9000,
        )
        all_trajectories.extend(trajs)
        logger.info(f"Collected {len(trajs)} trajectories from SMT")

    # 阶段 3（可选）：环境随机轨迹 —— 背景数据
    if include_env_random:
        logger.info(f"{'='*60}")
        logger.info(f"Stage 3: Generating env-random trajectories")
        logger.info(f"{'='*60}")

        trajs = generate_trajectories_from_env(
            topo=topo,
            num_flows=num_flows,
            num_episodes=env_episodes,
            scheduler_name="env_random",
            link_rate=link_rate,
            seed_start=10000,
        )
        all_trajectories.extend(trajs)
        logger.info(f"Collected {len(trajs)} trajectories from env-random")

    # 构建 DTDataset
    logger.info(f"{'='*60}")
    logger.info(f"Building DTDataset from {len(all_trajectories)} trajectories")
    logger.info(f"{'='*60}")

    dataset = DTDataset(name=f"dt_{topo}_{num_flows}flows")
    for traj in all_trajectories:
        dataset.add_from_collector(traj, gamma=gamma)

    # 计算统计量
    if len(dataset) > 0:
        dataset.compute_statistics()

    # 保存
    dataset.save(output_path)

    # 打印摘要
    summary = dataset.summary()
    logger.info(f"Dataset summary: {summary}")

    return dataset


# ============================================================================
# 多拓扑构建
# ============================================================================

def build_multi_topo_dataset(
    topos: List[str],
    num_flows: int,
    num_episodes: int,
    scheduler_names: List[str],
    output_base: str,
    timeout_s: int = 60,
    link_rate: int = 100,
    gamma: float = 1.0,
) -> DTDataset:
    """
    为多个拓扑构建数据集并合并。

    Returns:
        合并后的 DTDataset
    """
    merged = DTDataset(name="dt_multi_topo")

    for topo in topos:
        output_path = os.path.join(output_base, f"dt_{topo}_{num_flows}flows")
        dataset = build_offline_dataset(
            topo=topo,
            num_flows=num_flows,
            num_episodes=num_episodes,
            scheduler_names=scheduler_names,
            output_path=output_path,
            timeout_s=timeout_s,
            link_rate=link_rate,
            gamma=gamma,
        )
        merged.merge(dataset)

    # 重新计算合并后的统计量
    merged.compute_statistics()
    merged.name = f"dt_multi_topo_{'+'.join(topos)}_{num_flows}flows"

    merged_path = os.path.join(output_base, merged.name)
    merged.save(merged_path)

    logger.info(f"Merged dataset: {merged.summary()}")
    return merged


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="构建 Decision Transformer 离线训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Tabu 轨迹
  %(prog)s --topo CEV --num_flows 50 --num_episodes 500 \\
      --scheduler tabu_all_gate,tabu_no_gate --output out/dt_data/tabu

  # SMT + Tabu 混合
  %(prog)s --topo CEV --num_flows 50 --num_episodes 200 \\
      --scheduler tabu_all_gate,smt --output out/dt_data/combined --timeout 120

  # 多拓扑
  %(prog)s --topo CEV,RRG,ERG,BAG --num_flows 50 --num_episodes 200 \\
      --scheduler tabu_all_gate,smt --output out/dt_data/multi --timeout 60
        """,
    )

    parser.add_argument("--topo", type=str, default="CEV",
                        help="拓扑名称，多个用逗号分隔 (default: CEV)")
    parser.add_argument("--num_flows", type=int, default=50,
                        help="流数量 (default: 50)")
    parser.add_argument("--num_episodes", type=int, default=500,
                        help="每个调度器的目标 episode 数 (default: 500)")
    parser.add_argument("--scheduler", type=str, default="tabu_all_gate",
                        help="调度器列表，逗号分隔。可选: tabu_all_gate, tabu_no_gate, "
                             "tabu_random_gate, smt (default: tabu_all_gate)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（不含扩展名）(default: out/dt_data/<name>)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="调度器超时秒数 (default: 60)")
    parser.add_argument("--link_rate", type=int, default=100,
                        help="链路速率 (default: 100)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="RTG 折扣因子 (default: 1.0)")
    parser.add_argument("--include_env_random", action="store_true",
                        help="是否包含环境随机策略轨迹")
    parser.add_argument("--env_episodes", type=int, default=200,
                        help="环境随机轨迹数 (default: 200)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志级别 (default: INFO)")

    args = parser.parse_args()

    # 日志
    log_config(os.path.join(OUT_DIR, "dt_build_dataset.log"),
               getattr(logging, args.log_level.upper()))
    logger.info(f"Arguments: {args}")

    # 解析调度器列表
    scheduler_names = [s.strip() for s in args.scheduler.split(",")]
    for name in scheduler_names:
        if name not in SCHEDULER_REGISTRY:
            logger.error(f"Unknown scheduler: {name}. "
                         f"Available: {list(SCHEDULER_REGISTRY.keys())}")
            sys.exit(1)

    # 解析拓扑列表
    topos = [t.strip() for t in args.topo.split(",")]

    # 输出路径
    if args.output is None:
        output_base = os.path.join(DATA_DIR,
                                   f"dt_{'+'.join(topos)}_{args.num_flows}flows")
    else:
        output_base = args.output

    # 构建数据集
    if len(topos) == 1:
        dataset = build_offline_dataset(
            topo=topos[0],
            num_flows=args.num_flows,
            num_episodes=args.num_episodes,
            scheduler_names=scheduler_names,
            output_path=output_base,
            timeout_s=args.timeout,
            link_rate=args.link_rate,
            gamma=args.gamma,
            include_env_random=args.include_env_random,
            env_episodes=args.env_episodes,
        )
    else:
        dataset = build_multi_topo_dataset(
            topos=topos,
            num_flows=args.num_flows,
            num_episodes=args.num_episodes,
            scheduler_names=scheduler_names,
            output_base=output_base,
            timeout_s=args.timeout,
            link_rate=args.link_rate,
            gamma=args.gamma,
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset building complete!")
    logger.info(f"Summary: {dataset.summary()}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
