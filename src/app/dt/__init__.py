"""
GIN-DT 模块
===========
Decision Transformer for FlexTAS GCL scheduling.

模块结构：
    dt_model.py      — GIN-DT 核心模型（Transformer + token embeddings + action head）
    dt_trainer.py    — 训练器（data loading, training loop, checkpointing）
    dt_scheduler.py  — 调度器（自回归推理，与 BaseScheduler 兼容）
"""

from src.app.dt.dt_model import GINDTransformer, CausalTransformer, TransformerBlock
from src.app.dt.dt_trainer import DTTrainer, DTTrainingDataset, train_gin_dt
from src.app.dt.dt_scheduler import DTScheduler, DTMultiRTGScheduler

__all__ = [
    "GINDTransformer",
    "CausalTransformer",
    "TransformerBlock",
    "DTTrainer",
    "DTTrainingDataset",
    "train_gin_dt",
    "DTScheduler",
    "DTMultiRTGScheduler",
]
