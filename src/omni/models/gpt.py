from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # Query heads per KV head
        self.dropout = dropout

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
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
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
    def __init__(self, d_model: int, m: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * m)
        self.linear2 = nn.Linear(d_model * m, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class MixtureOfExperts(nn.Module):
    def __init__(
        self, d_model: int, m: int, num_experts: int, top_k: int, dropout: float
    ) -> None:
        super().__init__()
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.gate = MoEGate(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [MLP(d_model, m, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.size()

        # Get routing decisions
        expert_indices, expert_weights = self.gate(x)  # (B, T, top_k)
        N = B * T

        # Run ALL tokens through ALL experts (each is a single dense matmul)
        # Stack results: (num_experts, N, D)
        all_expert_outputs = torch.stack(
            [expert(x.view(N, D)) for expert in self.experts]
        )  # (num_experts, N, D)

        # Gather the top_k expert outputs per token
        # expert_indices: (B, T, top_k) -> (N, top_k)
        flat_indices = expert_indices.view(N, self.top_k)

        # Index into all_expert_outputs: for each token, pick its top_k experts
        # flat_indices (N, top_k) -> (top_k, N) for gather along expert dim
        flat_indices_t = flat_indices.t()  # (top_k, N)

        # Gather: for each top_k slot, select the right expert output per token
        selected = torch.stack(
            [
                all_expert_outputs[flat_indices_t[k], torch.arange(N, device=x.device)]
                for k in range(self.top_k)
            ]
        )  # (top_k, N, D)

        selected = selected.permute(1, 0, 2)  # (N, top_k, D)

        # Apply routing weights
        weights = expert_weights.view(N, self.top_k, 1)  # (N, top_k, 1)
        output = (selected * weights).sum(dim=1)  # (N, D)

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
        attn_dropout: float,
        mlp_dropout: float,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, attn_dropout
        )
        self.norm2 = RMSNorm(d_model)

        # Use MoE if specified, otherwise standard MLP
        if num_experts is not None and top_k is not None:
            self.mlp = MixtureOfExperts(d_model, m, num_experts, top_k, mlp_dropout)
        else:
            self.mlp = MLP(d_model, m, mlp_dropout)

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
        tie_weights: bool,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        m: int,
        num_layers: int,
        max_seq_length: int,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    m,
                    attn_dropout,
                    mlp_dropout,
                    num_experts,
                    top_k,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)

        if tie_weights:
            self.output_linear.weight = self.token_embedding.weight

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
        tie_weights=False,
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
