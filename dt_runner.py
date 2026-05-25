#!/usr/bin/env python3
"""
GIN-DT 训练运行脚本
=====================
用法：
    python dt_runner.py

    或指定参数：
    python dt_runner.py --dataset out/dt_data/... --output out/dt_models/... --epochs 50
"""

import os
import sys

# 必须在所有其他导入之前设置，解决 macOS 上 PyTorch + torch_geometric 的 OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import torch


def main():
    parser = argparse.ArgumentParser(description="GIN-DT Training Runner")
    parser.add_argument("--dataset", type=str,
                        default="out/dt_data/dt_multi_50flows/dt_multi_topo_CEV+RRG+ERG+BAG_50flows",
                        help="DTDataset 路径（不含扩展名）")
    parser.add_argument("--output", type=str,
                        default="out/dt_models/gin_dt",
                        help="模型输出路径")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Transformer 隐藏维度")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Transformer 层数")
    parser.add_argument("--context_len", type=int, default=600,
                        help="最大上下文长度")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cpu/cuda)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志级别")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("dt_runner")

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output}")

    # 延迟导入，确保环境变量已设置
    from src.app.dt.dt_trainer import train_gin_dt

    model, trainer = train_gin_dt(
        dataset_path=args.dataset,
        output_path=args.output,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        context_len=args.context_len,
        val_split=args.val_split,
        device=device,
    )

    logger.info(f"Training complete. Best val loss: {trainer.best_val_loss:.4f}")
    best_path = f"{args.output}_best.pt"
    final_path = f"{args.output}.pt"
    if os.path.exists(best_path):
        logger.info(f"Best model saved to: {best_path}")
    logger.info(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
