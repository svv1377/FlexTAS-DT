"""
DT 训练器
=========
从 DTDataset 加载离线数据，训练 GIN-DT 模型。

训练流程：
1. 加载数据集
2. 预编码所有 state（FeaturesExtractor 批量处理）
3. Mini-batch 训练（teacher forcing）
4. 定期验证
"""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from src.agent.encoder import FeaturesExtractor
from src.app.dt.dt_model import GINDTransformer
from src.app.dt_dataset.dt_dataset import DTDataset, DTTrajectory

logger = logging.getLogger(__name__)


# ============================================================================
# DT 训练数据集（PyTorch Dataset）
# ============================================================================

class DTTrainingDataset(Dataset):
    """
    将 DTDataset 转换为 PyTorch Dataset，支持 padding 和 batching。

    每条 trajectory 产出：
        - rtgs:       (T,)  float32
        - states:     (T, state_dim)  float32 — 预编码的状态
        - actions:    (T,)  int64
        - mask:       (T,)  bool — 有效 token mask
        - timesteps:  (T,)  int64
    """

    def __init__(
        self,
        dt_dataset: DTDataset,
        state_encoder: FeaturesExtractor,
        device: torch.device,
        context_len: int = 600,
        state_dim: int = 192,
        pre_encode: bool = True,
        normalize_rtgs: bool = True,
    ):
        self.dt_dataset = dt_dataset
        self.state_encoder = state_encoder
        self.device = device
        self.context_len = context_len
        self.state_dim = state_dim
        self.normalize_rtgs = normalize_rtgs
        self.rtg_mean = float(dt_dataset.rtg_mean)
        self.rtg_std = float(dt_dataset.rtg_std) if float(dt_dataset.rtg_std) > 1e-6 else 1.0

        self.trajectories = dt_dataset.trajectories

        # 如果 raw_states 不可用（旧数据集），检测 flat state 维度并创建投影层
        self._flat_state_dim = None
        self._state_projection = None
        if self.trajectories and not self.trajectories[0].raw_states:
            sample_flat = self.trajectories[0].states[0]
            self._flat_state_dim = sample_flat.shape[0]
            self._state_projection = nn.Linear(self._flat_state_dim, state_dim).to(device)
            logger.info(f"No raw_states found — using Linear({self._flat_state_dim}, {state_dim}) projection")

        # 预编码所有 state
        if pre_encode:
            logger.info(f"Pre-encoding states for {len(self.trajectories)} trajectories...")
            self._pre_encode_states()
            logger.info(f"Pre-encoding done: {len(self.encoded_states)} state tensors")

    def _pre_encode_states(self):
        """批量预编码所有 state dict → 192-dim 向量。"""
        self.encoded_states = []  # List of (T, state_dim) tensors
        self.encoded_rtgs = []
        self.encoded_actions = []
        self.encoded_masks = []

        self.state_encoder.eval()
        self.state_encoder.to(self.device)

        for traj in self.trajectories:
            T = len(traj.actions)
            if T == 0:
                continue

            # 使用 raw_states（dict）进行 GIN 编码；若无 raw_states，回退到 flat states
            if traj.raw_states:
                state_vecs = []
                for state_dict in traj.raw_states:
                    obs_tensor = {}
                    for k, v in state_dict.items():
                        if isinstance(v, np.ndarray):
                            t = torch.from_numpy(v).unsqueeze(0).float().to(self.device)
                        elif isinstance(v, torch.Tensor):
                            t = v.unsqueeze(0).float().to(self.device)
                        else:
                            t = torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                        obs_tensor[k] = t

                    # Fix adjacency_matrix dtype
                    if 'adjacency_matrix' in obs_tensor:
                        obs_tensor['adjacency_matrix'] = obs_tensor['adjacency_matrix'].to(torch.int64)

                    with torch.no_grad():
                        encoded = self.state_encoder(obs_tensor)  # (1, 192)
                    state_vecs.append(encoded.squeeze(0).cpu())

                self.encoded_states.append(torch.stack(state_vecs))  # (T, 192)
            else:
                # 回退：使用 flat states → 投影到 state_dim
                state_arr = np.stack(traj.states[:T])
                state_tensor = torch.from_numpy(state_arr).float()
                if self._state_projection is not None:
                    with torch.no_grad():
                        state_tensor = self._state_projection(state_tensor)
                self.encoded_states.append(state_tensor)

            rtgs = torch.from_numpy(traj.rtgs[:T]).float()
            if self.normalize_rtgs:
                rtgs = (rtgs - self.rtg_mean) / self.rtg_std
            self.encoded_rtgs.append(rtgs)
            self.encoded_actions.append(torch.from_numpy(traj.actions[:T]).long())
            self.encoded_masks.append(torch.from_numpy(traj.mask[:T]).bool())

    def __len__(self):
        return len(self.encoded_states)

    def __getitem__(self, idx):
        states = self.encoded_states[idx]
        rtgs = self.encoded_rtgs[idx]
        actions = self.encoded_actions[idx]
        mask = self.encoded_masks[idx]

        T = states.shape[0]

        # 截断：context_len 是 token 数上限，每步 3 token
        max_steps = self.context_len // 3
        if T > max_steps:
            states = states[:max_steps]
            rtgs = rtgs[:max_steps]
            actions = actions[:max_steps]
            mask = mask[:max_steps]

        return {
            "states": states,
            "rtgs": rtgs,
            "actions": actions,
            "mask": mask,
            "length": min(T, max_steps),
        }


def collate_fn(batch):
    """Padding + batching collate function。"""
    max_len = max(item["length"] for item in batch)
    B = len(batch)
    state_dim = batch[0]["states"].shape[-1]

    states = torch.zeros(B, max_len, state_dim)
    rtgs = torch.zeros(B, max_len)
    actions = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        states[i, :L] = item["states"][:L]
        rtgs[i, :L] = item["rtgs"][:L]
        actions[i, :L] = item["actions"][:L]
        mask[i, :L] = item["mask"][:L]

    return {
        "states": states,
        "rtgs": rtgs,
        "actions": actions,
        "mask": mask,
        "lengths": torch.tensor([item["length"] for item in batch]),
    }


# ============================================================================
# 训练器
# ============================================================================

class DTTrainer:
    """
    GIN-DT 训练器。

    使用方式：
        trainer = DTTrainer(model, state_encoder, device)
        trainer.train(train_dataset, val_dataset, num_epochs=50)
    """

    def __init__(
        self,
        model: GINDTransformer,
        state_encoder: FeaturesExtractor,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        grad_clip: float = 1.0,
        rtg_mean: float = 0.0,
        rtg_std: float = 1.0,
        normalize_rtgs: bool = True,
    ):
        self.model = model.to(device)
        self.state_encoder = state_encoder.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.rtg_mean = float(rtg_mean)
        self.rtg_std = float(rtg_std) if float(rtg_std) > 1e-6 else 1.0
        self.normalize_rtgs = normalize_rtgs

        # Learning rate scheduler (cosine with warmup)
        self.scheduler = None  # Will be set in train()

        # Logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def _get_lr_scheduler(self, total_steps: int):
        """Cosine LR scheduler with linear warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        train_dataset: DTTrainingDataset,
        val_dataset: Optional[DTTrainingDataset] = None,
        batch_size: int = 16,
        num_epochs: int = 50,
        eval_every: int = 5,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        训练循环。

        Args:
            train_dataset: 训练集
            val_dataset:   验证集（可选）
            batch_size:    批次大小
            num_epochs:    训练 epoch 数
            eval_every:    每 N 个 epoch 验证一次
            save_path:     模型保存路径

        Returns:
            训练历史 dict
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=0,  # GIN requires single process
        )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=0,
            )
        else:
            val_loader = None

        total_steps = len(train_loader) * num_epochs
        self.scheduler = self._get_lr_scheduler(total_steps)

        logger.info(f"Training: {num_epochs} epochs, {len(train_loader)} batches/epoch, "
                    f"{total_steps} total steps")

        history = defaultdict(list)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()

            for batch_idx, batch in enumerate(train_loader):
                states = batch["states"].to(self.device)
                rtgs = batch["rtgs"].to(self.device)
                actions = batch["actions"].to(self.device)
                mask = batch["mask"].to(self.device)

                # 跳过太短的序列（至少需要 2 步才能计算 loss）
                if states.shape[1] < 2:
                    continue

                self.optimizer.zero_grad()

                _, loss = self.model(rtgs, states, actions, attention_mask=mask)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 500 == 0:
                    logger.info(f"  Step {global_step}: loss={loss.item():.4f}, "
                                f"lr={self.scheduler.get_last_lr()[0]:.2e}")

            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            self.train_losses.append(avg_epoch_loss)
            epoch_time = time.time() - epoch_start

            log_msg = f"Epoch {epoch+1}/{num_epochs} | train_loss={avg_epoch_loss:.4f} | time={epoch_time:.1f}s"

            # Validation
            if val_loader is not None and (epoch + 1) % eval_every == 0:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                log_msg += f" | val_loss={val_loss:.4f}"

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if save_path:
                        self.save(save_path + "_best")
                        log_msg += " [BEST]"

            logger.info(log_msg)
            history['train_loss'].append(avg_epoch_loss)

        # Final save
        if save_path:
            self.save(save_path)
            logger.info(f"Model saved to {save_path}")

        return history

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """在验证集上计算平均 loss。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            states = batch["states"].to(self.device)
            rtgs = batch["rtgs"].to(self.device)
            actions = batch["actions"].to(self.device)
            mask = batch["mask"].to(self.device)

            if states.shape[1] < 2:
                continue

            _, loss = self.model(rtgs, states, actions, attention_mask=mask)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / max(1, num_batches)

    def save(self, path: str):
        """保存模型 checkpoint（包含配置）。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "state_encoder_state_dict": self.state_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "rtg_mean": self.rtg_mean,
            "rtg_std": self.rtg_std,
            "normalize_rtgs": self.normalize_rtgs,
            "model_config": {
                "state_dim": self.model.state_dim,
                "embed_dim": self.model.embed_dim,
                "num_heads": self.model.transformer.blocks[0].attn.num_heads,
                "num_layers": len(self.model.transformer.blocks),
                "max_seq_len": self.model.max_seq_len,
            },
        }
        torch.save(checkpoint, f"{path}.pt")
        logger.info(f"Checkpoint saved to {path}.pt")

    def load(self, path: str):
        """加载模型 checkpoint。"""
        checkpoint = torch.load(f"{path}.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "state_encoder_state_dict" in checkpoint:
            self.state_encoder.load_state_dict(checkpoint["state_encoder_state_dict"])
        else:
            logger.warning(
                "Checkpoint has no state_encoder_state_dict; keeping current "
                "FeaturesExtractor weights."
            )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.rtg_mean = float(checkpoint.get("rtg_mean", self.rtg_mean))
        self.rtg_std = float(checkpoint.get("rtg_std", self.rtg_std))
        if self.rtg_std < 1e-6:
            self.rtg_std = 1.0
        self.normalize_rtgs = bool(checkpoint.get("normalize_rtgs", self.normalize_rtgs))
        logger.info(f"Checkpoint loaded from {path}.pt (best_val_loss={self.best_val_loss:.4f})")


# ============================================================================
# 便捷训练函数
# ============================================================================

def create_state_encoder_for_env(device: torch.device) -> FeaturesExtractor:
    """
    创建一个与 NetEnv 兼容的 FeaturesExtractor。

    由于 FeaturesExtractor 需要 observation_space 来初始化，
    我们通过临时创建 NetEnv 来获取 observation_space。
    """
    from src.env.env import NetEnv
    from src.network.net import generate_graph, FlowGenerator, Network

    # 创建一个临时 env 来获取 observation_space
    graph = generate_graph("CEV", 100)
    flow_gen = FlowGenerator(graph, seed=0)
    flows = flow_gen(10)
    network = Network(graph, flows)
    env = NetEnv(network)

    state_encoder = FeaturesExtractor(env.observation_space)
    state_encoder.to(device)
    return state_encoder


def train_gin_dt(
    dataset_path: str,
    output_path: str,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    context_len: int = 600,
    val_split: float = 0.1,
    device: Optional[torch.device] = None,
) -> Tuple[GINDTransformer, DTTrainer]:
    """
    一站式训练函数。

    Args:
        dataset_path: DTDataset 路径（不含扩展名）
        output_path:  模型保存路径
        ...          其他超参数

    Returns:
        (model, trainer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    # 加载数据集
    dataset = DTDataset()
    dataset.load(dataset_path)
    logger.info(f"Loaded dataset: {len(dataset)} trajectories")

    # 创建 state encoder
    state_encoder = create_state_encoder_for_env(device)

    # 创建训练数据集
    full_train = DTTrainingDataset(
        dataset, state_encoder, device,
        context_len=context_len,
        pre_encode=True,
    )

    # 划分训练/验证集
    n_total = len(full_train)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_subset = torch.utils.data.Subset(full_train, range(n_train))
    val_subset = torch.utils.data.Subset(full_train, range(n_train, n_total))

    logger.info(f"Train: {n_train} trajectories, Val: {n_val} trajectories")

    # Wrap subsets to work with collate_fn
    # We need custom Dataset wrappers since Subset doesn't directly support our collate
    train_dataset = SubsetWrapper(full_train, range(n_train))
    val_dataset = SubsetWrapper(full_train, range(n_train, n_total))

    # 创建模型
    model = GINDTransformer(
        state_dim=192,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        max_seq_len=context_len,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # 训练
    trainer = DTTrainer(
        model, state_encoder, device,
        learning_rate=learning_rate,
        rtg_mean=dataset.rtg_mean,
        rtg_std=dataset.rtg_std,
        normalize_rtgs=True,
    )

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_path=output_path,
    )

    return model, trainer


class SubsetWrapper(Dataset):
    """Dataset wrapper for torch.utils.data.Subset compatibility with our collate_fn."""

    def __init__(self, dataset: DTTrainingDataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
