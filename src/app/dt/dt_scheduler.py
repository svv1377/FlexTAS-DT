"""
DT 调度器
=========
基于 GIN-DT 模型的调度器，与现有 BaseScheduler 接口兼容。

使用方式：
    scheduler = DTScheduler(network, model_path="out/dt_models/gin_dt_best.pt")
    ok = scheduler.schedule()
    if ok:
        res = scheduler.get_res()
"""

import logging
import math
import time
from typing import Optional

import numpy as np
import torch

from src.agent.encoder import FeaturesExtractor
from src.app.dt.dt_model import GINDTransformer
from src.app.scheduler import BaseScheduler, ScheduleRes
from src.env.env import NetEnv, SchedulingError
from src.network.net import Network, Net

logger = logging.getLogger(__name__)


class DTScheduler(BaseScheduler):
    """
    基于 GIN-DT 的调度器。

    自回归推理流程：
        1. 给定目标 RTG
        2. 每步：编码 state → 预测 action → env.step() → 更新 RTG
        3. 直到 episode 结束（成功或失败）
    """

    def __init__(
        self,
        network: Network,
        model_path: str,
        embed_dim: int = None,
        num_heads: int = None,
        num_layers: int = None,
        target_rtg: Optional[float] = None,
        deterministic: bool = True,
        max_steps: int = 500,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(network, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.target_rtg = target_rtg
        self.deterministic = deterministic
        self.max_steps = max_steps

        # 先加载 checkpoint 获取配置
        ckpt_path = model_path if model_path.endswith(".pt") else f"{model_path}.pt"
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("model_config", {})

        state_dim = config.get("state_dim", 192)
        embed_dim = embed_dim or config.get("embed_dim", 128)
        num_heads = num_heads or config.get("num_heads", 4)
        num_layers = num_layers or config.get("num_layers", 4)
        max_seq_len = config.get("max_seq_len", 1024)

        # 创建模型（使用与训练时相同的配置）
        self.model = GINDTransformer(
            state_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            device=self.device,
        )

        # 加载权重
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logger.info(f"Model loaded from {model_path} "
                    f"(embed_dim={embed_dim}, heads={num_heads}, layers={num_layers})")

        # 创建 state encoder（需要从 env 获取 observation_space）
        self.state_encoder = None  # 延迟初始化
        self._res = None

    def _init_state_encoder(self, env: NetEnv):
        """延迟初始化 state encoder（需要 env 来确定 observation_space）。"""
        if self.state_encoder is None:
            self.state_encoder = FeaturesExtractor(env.observation_space)
            self.state_encoder.to(self.device)
            self.state_encoder.eval()

    def _get_target_rtg(self) -> float:
        """获取目标 RTG。若未设置，基于经验估计。"""
        if self.target_rtg is not None:
            return self.target_rtg

        # 基于数据集统计的经验估计
        # CEV: ~188, RRG: ~96, ERG: ~90, BAG: ~84
        # 默认使用较高值偏向高质量解
        num_flows = len(self.flows)
        return num_flows * 1.9  # ~1 reward per hop, ~3.7 hops average for CEV

    def schedule(self) -> bool:
        """
        运行 DT 调度。

        Returns:
            True 如果成功调度所有 flow
        """
        start_time = time.time()

        # 重建 Network 对象
        from src.network.net import Network
        network = Network(self.graph, self.flows)

        # 创建环境
        env = NetEnv.__new__(NetEnv)
        NetEnv.__init__(env, network)

        self._init_state_encoder(env)

        target_rtg = self._get_target_rtg()
        logger.info(f"DT scheduling: target_rtg={target_rtg:.1f}, "
                    f"num_flows={len(self.flows)}")

        # 自回归生成
        success, actions, rewards, states = self.model.generate_episode(
            env=env,
            target_rtg=target_rtg,
            state_encoder=self.state_encoder,
            deterministic=self.deterministic,
            max_steps=self.max_steps,
        )

        elapsed = time.time() - start_time

        if success:
            self._res = env.links_operations.copy()
            logger.info(f"DT scheduling SUCCESS: {len(actions)} steps, "
                        f"{elapsed:.2f}s, final_rtg={target_rtg - sum(rewards):.1f}")
        else:
            logger.info(f"DT scheduling FAILED: {len(actions)} steps, "
                        f"{elapsed:.2f}s, flows_scheduled={env.flow_index}/{len(self.flows)}")

        return success

    def get_res(self) -> ScheduleRes:
        """获取调度结果。"""
        if self._res is None:
            raise RuntimeError("schedule() must be called and succeed before get_res()")
        return self._res


class DTMultiRTGScheduler(BaseScheduler):
    """
    多 RTG 尝试调度器：从高到低尝试多个目标 RTG 值，取第一个成功的。

    这利用了 DT 的 RTG 条件能力：高 RTG → 高质量但可能失败，
    低 RTG → 更容易成功但质量较低。
    """

    def __init__(
        self,
        network: Network,
        model_path: str,
        rtg_levels: Optional[list] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(network, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_path = model_path
        self.rtg_levels = rtg_levels
        self._res = None

    def schedule(self) -> bool:
        """从高到低尝试多个 RTG 值。"""
        num_flows = len(self.flows)

        if self.rtg_levels is None:
            # 默认 RTG 级别：从高到低
            base = num_flows * 1.9
            self.rtg_levels = [base * r for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]]

        for level_idx, rtg in enumerate(self.rtg_levels):
            logger.info(f"Trying RTG level {level_idx+1}/{len(self.rtg_levels)}: {rtg:.1f}")

            from src.network.net import Network
            scheduler = DTScheduler(
                Network(self.graph, self.flows),
                model_path=self.model_path,
                target_rtg=rtg,
                deterministic=True,
                device=self.device,
                timeout_s=self.timeout_s,
            )

            ok = scheduler.schedule()
            if ok:
                self._res = scheduler.get_res()
                logger.info(f"Success at RTG level {level_idx+1}: {rtg:.1f}")
                return True

        logger.info("All RTG levels failed")
        return False

    def get_res(self) -> ScheduleRes:
        if self._res is None:
            raise RuntimeError("schedule() must be called and succeed before get_res()")
        return self._res
