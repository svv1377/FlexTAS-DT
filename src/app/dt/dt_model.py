"""
GIN-DT 核心模型
===============
混合架构：GIN 图编码器 + Decision Transformer

- FeaturesExtractor (GIN+MLP) 将 dict observation 编码为 192-dim 状态嵌入
- GPT-2 style Causal Transformer 处理 (RTG, state, action) token 序列
- 输出 2-class logits 预测 gating/no-gating
- 推理时叠加 action mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.agent.encoder import FeaturesExtractor


# ============================================================================
# GPT-2 style Causal Transformer
# ============================================================================

class CausalSelfAttention(nn.Module):
    """单头 → 多头因果自注意力。"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, D) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v  # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class TransformerBlock(nn.Module):
    """Transformer block: Attention + MLP，带 Pre-LN。"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    """GPT-2 style causal transformer stack。"""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return x


# ============================================================================
# GIN-DT: 完整模型
# ============================================================================

class GINDTransformer(nn.Module):
    """
    GIN-DT 混合模型。

    输入：
        - rtgs:          (B, T)     Return-to-Go 序列
        - states_dict:   Dict       原始 dict observation（每步一个）
        - actions:       (B, T)     动作序列 (0/1)
        - action_masks:  (B, T, 2)  可选，推理时的 action mask
        - return_logits: bool       是否返回 logits（推理模式）

    输出：
        - action_logits: (B, T, 2)  每步的 action logits
        - action_preds:  (B, T)     每步的预测 action

    训练时：
        Input:  [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, aₜ]
        Target: 在每个 state token 位置预测对应的 aₜ
        Loss:   CrossEntropy at state token positions

    推理时：
        自回归：每步生成一个 action, 用 env.step() 获取新 state 和 reward
    """

    def __init__(
        self,
        state_dim: int = 192,       # FeaturesExtractor 输出维度
        embed_dim: int = 128,       # Transformer 隐藏维度
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        device: torch.device = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # ---- Token Embeddings ----
        self.rtg_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.action_embed = nn.Embedding(2, embed_dim)

        # ---- Positional Embedding ----
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # ---- Layer Norm for embeddings ----
        self.embed_ln = nn.LayerNorm(embed_dim)

        # ---- Causal Transformer ----
        self.transformer = CausalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ---- Prediction Head ----
        self.pred_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2),
        )

        self.dropout = nn.Dropout(dropout)

        if device is not None:
            self.to(device)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # ================================================================
    # 训练前向（teacher forcing）
    # ================================================================

    def forward(
        self,
        rtgs: torch.Tensor,            # (B, T)
        states_encoded: torch.Tensor,  # (B, T, state_dim) — 预编码的状态
        actions: torch.Tensor,         # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练模式前向传播。

        Args:
            rtgs:            (B, T) RTG 序列
            states_encoded:  (B, T, state_dim) 预编码的状态向量
            actions:         (B, T) 动作序列
            attention_mask:  (B, T) 有效 token mask

        Returns:
            logits:  (B, T, 2) 每步的 action logits
            loss:    标量 cross-entropy loss
        """
        B, T = rtgs.shape
        T = min(T, self.max_seq_len // 3)  # 安全截断
        rtgs = rtgs[:, :T]
        states_encoded = states_encoded[:, :T]
        actions = actions[:, :T]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :T]

        # ---- Build token sequence ----
        # 每步 3 个 token: [RTG, state, action]
        total_tokens = 3 * T

        # Embed each type
        rtg_tokens = self.rtg_embed(rtgs.unsqueeze(-1))       # (B, T, D)
        state_tokens = self.state_embed(states_encoded)        # (B, T, D)
        action_tokens = self.action_embed(actions.long())      # (B, T, D)

        # Interleave: [R₁, s₁, a₁, R₂, s₂, a₂, ...]
        # stack → (B, T, 3, D) → reshape → (B, 3*T, D)
        tokens = torch.stack([rtg_tokens, state_tokens, action_tokens], dim=2)
        tokens = tokens.view(B, total_tokens, self.embed_dim)

        # ---- Positional embedding ----
        pos = self.pos_embed[:, :total_tokens, :]
        tokens = tokens + pos
        tokens = self.embed_ln(tokens)
        tokens = self.dropout(tokens)

        # ---- Build attention mask from per-token mask ----
        if attention_mask is not None:
            # Expand (B, T) → (B, 3*T)
            attn_mask_3t = attention_mask.unsqueeze(-1).repeat(1, 1, 3).view(B, total_tokens)
            # causal transformer already uses causal mask internally,
            # but we can use key_padding_mask for padding tokens
            # For simplicity, we pass it through by zeroing out padded tokens
            tokens = tokens * attn_mask_3t.unsqueeze(-1).float()

        # ---- Transformer ----
        hidden = self.transformer(tokens)  # (B, 3*T, D)

        # ---- Extract state token positions (every 3rd token, offset 1) ----
        # Positions: 1, 4, 7, ... (0-indexed). The current action token is after
        # the state token, so causal attention cannot leak a_t into this prediction.
        action_hidden = hidden[:, 1::3, :]  # (B, T, D)

        # ---- Predict actions ----
        logits = self.pred_head(action_hidden)  # (B, T, 2)

        # ---- Compute loss (only on valid tokens) ----
        if attention_mask is not None:
            target_actions = actions  # (B, T)
            valid_mask = attention_mask  # (B, T)

            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    logits[valid_mask],
                    target_actions[valid_mask].long(),
                )
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return logits, loss
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                actions.reshape(-1).long(),
            )
            return logits, loss

    # ================================================================
    # 推理前向（自回归单步）
    # ================================================================

    def get_action(
        self,
        rtg: torch.Tensor,              # (B, 1) 当前 RTG
        state_encoded: torch.Tensor,    # (B, state_dim) 当前状态编码
        prev_action: Optional[torch.Tensor] = None,  # (B,) 上一步 action
        action_mask: Optional[torch.Tensor] = None,  # (B, 2) bool
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        自回归单步推理。

        Args:
            rtg:            当前 RTG 值
            state_encoded:  当前状态（已由 FeaturesExtractor 编码）
            prev_action:    上一步 action（第一步时为 None，用 0 代替）
            action_mask:    当前步的有效动作 mask
            deterministic:  True → argmax, False → 采样

        Returns:
            action: (B,) 预测的动作
        """
        if prev_action is not None:
            # Kept for API compatibility. A single current state is not enough to
            # place the previous action in the DT sequence without its state/RTG.
            # Full autoregressive decoding should use get_action_from_history().
            pass
        return self.get_action_from_history(
            rtgs=rtg,
            states_encoded=state_encoded.unsqueeze(1),
            actions=None,
            action_mask=action_mask,
            deterministic=deterministic,
        )

    def get_action_from_history(
        self,
        rtgs: torch.Tensor,                  # (B, T)
        states_encoded: torch.Tensor,        # (B, T, state_dim)
        actions: Optional[torch.Tensor] = None,  # (B, T-1) previous actions
        action_mask: Optional[torch.Tensor] = None,  # (B, 2) bool
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        自回归推理：使用完整历史 [R_1,s_1,a_1,...,R_t,s_t] 预测当前 a_t。

        actions 只包含已经执行过的历史动作，长度应为 T-1。当前动作 token
        不放入输入序列，避免推理时不可获得的信息泄漏。
        """
        B, T = rtgs.shape
        device = rtgs.device

        if actions is None:
            actions = torch.empty(B, 0, dtype=torch.long, device=device)
        else:
            actions = actions.to(device=device, dtype=torch.long)

        if actions.shape[1] != max(0, T - 1):
            raise ValueError(
                f"actions length must be T-1 for autoregressive inference, "
                f"got actions={actions.shape[1]}, T={T}"
            )

        max_steps = max(1, (self.max_seq_len + 1) // 3)
        if T > max_steps:
            rtgs = rtgs[:, -max_steps:]
            states_encoded = states_encoded[:, -max_steps:]
            if max_steps > 1:
                actions = actions[:, -(max_steps - 1):]
            else:
                actions = actions[:, :0]
            T = max_steps

        rtg_tokens = self.rtg_embed(rtgs.unsqueeze(-1))       # (B, T, D)
        state_tokens = self.state_embed(states_encoded)       # (B, T, D)
        action_tokens = self.action_embed(actions)            # (B, T-1, D)

        tokens_by_step = []
        for t in range(T):
            tokens_by_step.append(rtg_tokens[:, t:t + 1, :])
            tokens_by_step.append(state_tokens[:, t:t + 1, :])
            if t < T - 1:
                tokens_by_step.append(action_tokens[:, t:t + 1, :])

        tokens = torch.cat(tokens_by_step, dim=1)  # (B, 3*T-1, D)
        total_tokens = tokens.shape[1]
        if total_tokens > self.max_seq_len:
            raise ValueError(
                f"token sequence length {total_tokens} exceeds max_seq_len={self.max_seq_len}"
            )

        pos = self.pos_embed[:, :total_tokens, :]
        tokens = tokens + pos
        tokens = self.embed_ln(tokens)
        tokens = self.dropout(tokens)

        hidden = self.transformer(tokens)            # (B, 3*T-1, D)
        current_state_pos = total_tokens - 1
        action_hidden = hidden[:, current_state_pos, :]  # (B, D)
        logits = self.pred_head(action_hidden)       # (B, 2)

        # Action masking
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        return action

    def get_action_with_kv_cache(
        self,
        rtg: torch.Tensor,
        state_encoded: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ):
        """
        带 KV-cache 的单步推理（用于加速自回归生成）。

        当前实现回退到 get_action()，未来可优化。
        """
        return self.get_action(rtg, state_encoded, prev_action, action_mask, deterministic)

    # ================================================================
    # 完整 episode 推理
    # ================================================================

    @torch.no_grad()
    def generate_episode(
        self,
        env,                           # NetEnv 实例
        target_rtg: float,             # 原始奖励尺度下的目标 RTG
        state_encoder,                 # 预初始化的 FeaturesExtractor
        deterministic: bool = True,
        max_steps: int = 500,
        normalize_rtgs: bool = False,
        rtg_mean: float = 0.0,
        rtg_std: float = 1.0,
    ) -> Tuple[bool, list, list, list]:
        """
        在给定环境中自回归生成完整 episode。

        Args:
            env:             NetEnv 实例
            target_rtg:     原始奖励尺度下的目标 RTG 值
            state_encoder:  FeaturesExtractor 实例（用于编码 state dict → 192-dim）
            deterministic:  是否确定性推理
            max_steps:      最大步数

        Returns:
            (success, actions_list, rewards_list, states_list)
        """
        device = next(self.parameters()).device
        obs, _ = env.reset()

        if rtg_std < 1e-6:
            rtg_std = 1.0

        def model_rtg(raw_rtg: float) -> float:
            if not normalize_rtgs:
                return raw_rtg
            return (raw_rtg - rtg_mean) / rtg_std

        current_rtg = target_rtg
        done = False
        all_actions = []
        all_rewards = []
        all_states = []
        rtg_history = []
        state_history = []

        step = 0
        while not done and step < max_steps:
            # Encode state
            obs_tensor = {
                k: torch.from_numpy(v).unsqueeze(0).float().to(device)
                if not isinstance(v, torch.Tensor) else v.unsqueeze(0).float().to(device)
                for k, v in obs.items()
            }
            # Fix adjacency_matrix dtype
            if 'adjacency_matrix' in obs_tensor:
                obs_tensor['adjacency_matrix'] = obs_tensor['adjacency_matrix'].to(torch.int64)

            state_encoded = state_encoder(obs_tensor)  # (1, 192)
            rtg_history.append(float(model_rtg(current_rtg)))
            state_history.append(state_encoded.squeeze(0))

            # Get action mask
            masks = env.action_masks()  # (2,) bool
            action_mask = torch.from_numpy(masks).unsqueeze(0).to(device)

            # Predict action
            rtg_tensor = torch.tensor([rtg_history], dtype=torch.float32, device=device)
            states_tensor = torch.stack(state_history, dim=0).unsqueeze(0).to(device)
            if all_actions:
                actions_tensor = torch.tensor([all_actions], dtype=torch.long, device=device)
            else:
                actions_tensor = None

            action = self.get_action_from_history(
                rtgs=rtg_tensor,
                states_encoded=states_tensor,
                actions=actions_tensor,
                action_mask=action_mask,
                deterministic=deterministic,
            )  # (1,)

            action_val = int(action.item())

            # Step environment
            obs, reward, done, truncated, info = env.step(action_val)
            all_actions.append(action_val)
            all_rewards.append(reward)
            all_states.append(obs)

            # Update RTG
            current_rtg = current_rtg - reward

            step += 1

        success = info.get('success', False) if done else False
        return success, all_actions, all_rewards, all_states
