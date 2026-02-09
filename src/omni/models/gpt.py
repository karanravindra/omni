from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # Query heads per KV head

        assert self.head_dim * n_heads == d_model
        assert n_heads % n_kv_heads == 0

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, 2 * n_kv_heads * self.head_dim)
        self.out_linear = nn.Linear(d_model, d_model)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads to match number of Q heads"""
        B, n_kv_heads, T, head_dim = x.size()
        if self.n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
            .expand(B, n_kv_heads, self.n_rep, T, head_dim)
            .reshape(B, n_kv_heads * self.n_rep, T, head_dim)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: (B, T, D)
        """
        B, T, _ = x.size()

        # Project Q, K, V
        q = self.q_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_linear(x).view(B, T, self.n_kv_heads, 2 * self.head_dim)
        k, v = kv.chunk(2, dim=-1)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = v.transpose(1, 2)

        # Repeat K/V to match Q heads
        k = self._repeat_kv(k)  # (B, n_heads, T, head_dim)
        v = self._repeat_kv(v)

        # Attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_linear(attn), (k, v)


class MoEGate(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k: int = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        Returns: top_k expert indices and weights
        """
        logits = self.gate(x)  # (B, T, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_indices, top_k_weights


class MLP(nn.Module):
    def __init__(self, d_model: int, m: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * m)
        self.linear2 = nn.Linear(d_model * m, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, m: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.gate = MoEGate(d_model, num_experts, top_k)

        # Batched expert weights for GPU-friendly computation
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, d_model * m))
        self.b1 = nn.Parameter(torch.zeros(num_experts, d_model * m))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model * m, d_model))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))

        for i in range(num_experts):
            nn.init.normal_(self.w1[i], std=0.02)
            nn.init.normal_(self.w2[i], std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.size()

        # Get routing decisions
        expert_indices, expert_weights = self.gate(x)  # (B, T, top_k)

        # Flatten
        x_flat = x.view(-1, D)  # (N, D) where N = B*T
        N = x_flat.size(0)

        # Gather selected expert parameters: run all N tokens through top_k experts
        flat_indices = expert_indices.view(N * self.top_k)  # (N*top_k,)

        # Gather expert weights for selected experts
        w1 = self.w1[flat_indices]  # (N*top_k, D, D*m)
        b1 = self.b1[flat_indices]  # (N*top_k, D*m)
        w2 = self.w2[flat_indices]  # (N*top_k, D*m, D)
        b2 = self.b2[flat_indices]  # (N*top_k, D)

        # Repeat input for each top_k selection
        x_rep = x_flat.unsqueeze(1).expand(N, self.top_k, D).reshape(N * self.top_k, D)

        # Batched expert forward: x @ w1 + b1 -> gelu -> @ w2 + b2
        h = torch.bmm(x_rep.unsqueeze(1), w1).squeeze(1) + b1  # (N*top_k, D*m)
        h = F.gelu(h)
        expert_out = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2  # (N*top_k, D)

        # Reshape and apply routing weights
        expert_out = expert_out.view(N, self.top_k, D)  # (N, top_k, D)
        weights = expert_weights.view(N, self.top_k, 1)  # (N, top_k, 1)
        output = (expert_out * weights).sum(dim=1)  # (N, D)

        return output.view(B, T, D)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps
        return x / norm * self.scale


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        m: int,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)

        # Use MoE if specified, otherwise standard MLP
        if num_experts is not None and top_k is not None:
            self.mlp = MixtureOfExperts(d_model, m, num_experts, top_k)
        else:
            self.mlp = MLP(d_model, m)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        m: int,
        num_layers: int,
        max_seq_length: int,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, n_kv_heads, m, num_experts, top_k)
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T = x.size()
        x = self.token_embedding(x) + self.position_embedding[:, :T, :]
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm_f(x)
        return self.output_linear(x)


if __name__ == "__main__":
    from torchinfo import summary

    model = GPT(
        vocab_size=256,
        d_model=32,
        n_heads=4,
        n_kv_heads=1,
        m=2,
        num_layers=3,
        max_seq_length=64,
        num_experts=16,
        top_k=1,
    )

    summary(model, input_size=(2, 64), dtypes=[torch.long])
