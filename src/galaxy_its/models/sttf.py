"""Spatio-Temporal Threat Forecaster (STTF).

    per window:  dense multi-head GAT over CAN-ID nodes (edge weights bias attention,
                 non-edges masked)            -> node embeddings -> mean/max readout
    over time:   Transformer encoder on the L window embeddings
    heads:       global attack probability per horizon; per-node probability per horizon
    uncertainty: Monte Carlo dropout at inference

The CAN-ID graph is small (tens of nodes), so a dense GAT batched over B*L graphs is simpler
and faster on one GPU than sparse message passing, and needs no compiled extensions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class STTFConfig:
    n_nodes: int
    n_feat: int
    n_horizons: int
    seq_len: int = 20
    d_model: int = 128
    gat_layers: int = 2
    gat_heads: int = 8
    tf_layers: int = 4
    tf_heads: int = 8
    dropout: float = 0.1
    use_graph: bool = True        # ablation (a): False -> nodes attend only to themselves


class DenseGATLayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.edge_w = nn.Parameter(torch.ones(heads))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h [G, N, d]; adj [G, N, N] (weights); mask [G, N, N] bool (True = edge allowed)
        G, N, _ = h.shape
        z = self.W(h).view(G, N, self.h, self.dh)
        s = (z * self.a_src).sum(-1)                         # [G, N, H]
        t = (z * self.a_dst).sum(-1)
        e = F.leaky_relu(s.unsqueeze(2) + t.unsqueeze(1), 0.2)   # [G, N, N, H]
        e = e + self.edge_w * adj.unsqueeze(-1)
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        att = self.drop(torch.softmax(e, dim=2))
        msg = torch.einsum("gijh,gjhd->gihd", att, z).reshape(G, N, -1)
        h = self.norm(h + self.drop(self.out(msg)))
        return self.norm2(h + self.drop(self.ff(h)))


class STTF(nn.Module):
    def __init__(self, cfg: STTFConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.node_emb = nn.Embedding(cfg.n_nodes, d)          # identity of each CAN ID
        self.inp = nn.Linear(cfg.n_feat, d)
        self.gat = nn.ModuleList(DenseGATLayer(d, cfg.gat_heads, cfg.dropout) for _ in range(cfg.gat_layers))
        self.readout = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, d))
        layer = nn.TransformerEncoderLayer(d, cfg.tf_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, cfg.tf_layers)
        self.head_global = nn.Sequential(nn.LayerNorm(d), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_horizons))
        self.head_node = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(cfg.dropout), nn.Linear(2 * d, cfg.n_horizons))

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x [B, L, N, F]; adj [B, L, N, N]
        B, L, N, _ = x.shape
        h = self.inp(x) + self.node_emb.weight                          # [B, L, N, d]
        h = h.reshape(B * L, N, -1)
        adj = adj.reshape(B * L, N, N)
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        mask = ((adj > 0) | eye) if self.cfg.use_graph else eye.expand(B * L, N, N)
        if not self.cfg.use_graph:
            adj = torch.zeros_like(adj)
        for layer in self.gat:
            h = layer(h, adj, mask)
        g = self.readout(torch.cat([h.mean(1), h.amax(1)], -1)).view(B, L, -1)
        ctx = self.temporal(g + self.pos[:, :L])[:, -1]                 # [B, d]
        last_nodes = h.view(B, L, N, -1)[:, -1]                          # [B, N, d]
        node_in = torch.cat([last_nodes, ctx[:, None].expand(-1, N, -1)], -1)
        return {
            "logit": self.head_global(ctx),                              # [B, K]
            "node_logit": self.head_node(node_in).transpose(1, 2),       # [B, K, N]
        }

    @torch.no_grad()
    def mc_predict(self, x, adj, passes: int = 50):
        """Monte Carlo dropout: returns mean and variance of global attack probability."""
        was = self.training
        self.train()
        probs = torch.stack([torch.sigmoid(self(x, adj)["logit"]) for _ in range(passes)])
        self.train(was)
        return probs.mean(0), probs.var(0)


def sttf_loss(out, batch, label_smoothing=0.05, node_weight=0.3, mono_weight=0.1, pos_weight=None):
    y, yn = batch["y"], batch["y_node"]
    valid = y >= 0
    ys = y * (1 - label_smoothing) + 0.5 * label_smoothing
    lg = F.binary_cross_entropy_with_logits(out["logit"], ys, reduction="none", pos_weight=pos_weight)
    loss_g = (lg * valid).sum() / valid.sum().clamp(min=1)

    nvalid = yn >= 0
    yns = yn * (1 - label_smoothing) + 0.5 * label_smoothing
    ln = F.binary_cross_entropy_with_logits(out["node_logit"], yns, reduction="none")
    loss_n = (ln * nvalid).sum() / nvalid.sum().clamp(min=1)

    # temporal consistency: labels are nested over horizons, so predictions should be monotone
    p = torch.sigmoid(out["logit"])
    loss_m = F.relu(p[:, :-1] - p[:, 1:]).mean() if p.shape[1] > 1 else p.new_zeros(())
    total = loss_g + node_weight * loss_n + mono_weight * loss_m
    return total, {"global": loss_g.item(), "node": loss_n.item(), "mono": loss_m.item()}
