"""Minimal Mamba in one PyTorch file, modified so B_t and C_t depend on BOTH
the current SSM input x_t (u_t) and the previous state h_(t-1).

Changes vs. the standard mamba-minimal:
    1. B and C come from ONE fused projection (`BC_proj`) that is split afterwards.
    2. The projection input is [u_t ; summary(h_(t-1))].
    3. Because the projection is linear, W [u; h] = Wx u + Wh h, so the u-part is
       computed for all timesteps in ONE matmul before the loop (big speedup).
    4. Wh is zero-initialised so training starts as input-only selective Mamba.
    5. h summary options: plain mean, or learned weighted pool, plus optional LayerNorm.
    6. from_pretrained maps the original x_proj rows into delta / BC_proj.

Note: since B_t depends on h_(t-1), the recurrence is no longer linear in h,
so the parallel scan cannot be used. The scan runs sequentially.

Glossary:
    b: batch, l: sequence length, d_in: d_model*expand, n: d_state
"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    # --- state-dependence options ---
    h_pool: str = 'mean'        # 'mean' or 'learned' (weighted pool over d_inner)
    h_norm: bool = True         # LayerNorm over d_state on the pooled state

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

        assert self.h_pool in ('mean', 'learned')


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, input_ids):
        """input_ids: (b, l) -> logits: (b, l, vocab_size)"""
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained Mamba weights from HuggingFace into this modified model.

        The original x_proj has shape (dt_rank + 2n, d_inner) with rows ordered
        [delta | B | C]. Here:
            rows [:dt_rank]            -> x_proj
            rows [dt_rank:dt_rank+n]   -> BC_proj.weight[:n,  :d_inner]   (B, x-part)
            rows [dt_rank+n:]          -> BC_proj.weight[n:2n, :d_inner]  (C, x-part)
        The h-part of BC_proj (and pooling / norm params) keeps its fresh
        initialisation (zero h-weights => behaves like the original model at start).

        Pretrained names: 'state-spaces/mamba-{130m,370m,790m,1.4b,2.8b,2.8b-slimpj}'
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size'],
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        n, r, d_in = args.d_state, args.dt_rank, args.d_inner

        for key in state_dict:
            new_key = key.replace('backbone.', '')
            if new_key.endswith('mixer.x_proj.weight'):
                prefix = new_key[:-len('x_proj.weight')]
                w = state_dict[key]                       # (r + 2n, d_in)
                new_state_dict[prefix + 'x_proj.weight'] = w[:r]
                # Build full BC weight: x-columns from pretrained, h-columns zero.
                bc = torch.zeros(2 * n, d_in + n, dtype=w.dtype)
                bc[:n, :d_in] = w[r:r + n]
                bc[n:, :d_in] = w[r + n:r + 2 * n]
                new_state_dict[prefix + 'BC_proj.weight'] = bc
            else:
                new_state_dict[new_key] = state_dict[key]

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        # Only the new h-dependence params should be missing.
        allowed = ('h_norm', 'h_pool_w')
        bad = [k for k in missing if not any(a in k for a in allowed)]
        assert not bad and not unexpected, f"missing={bad}, unexpected={unexpected}"

        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Norm -> Mamba -> Add."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A Mamba block with state-dependent B and C (single fused projection)."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # Delta from the current SSM input only.
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank, bias=False)

        # ONE projection producing both B_t and C_t, from [u_t ; summary(h_(t-1))].
        # Output is split into B_t and C_t afterwards.
        self.BC_proj = nn.Linear(
            args.d_inner + args.d_state,
            2 * args.d_state,
            bias=False,
        )
        # Zero-init the h-columns: start as input-only selective Mamba.
        with torch.no_grad():
            self.BC_proj.weight[:, args.d_inner:].zero_()

        # Optional pooling of h over d_inner, and normalisation over d_state.
        if args.h_pool == 'learned':
            self.h_pool_w = nn.Parameter(torch.full((args.d_inner,), 1.0 / args.d_inner))
        self.h_norm = nn.LayerNorm(args.d_state) if args.h_norm else nn.Identity()

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Input/output shape: (batch, sequence_length, d_model)."""
        _, l, _ = x.shape

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner],
            dim=-1,
        )

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, u):
        """Run the SSM with recurrently generated B_t and C_t."""
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        delta = self.x_proj(u)
        delta = F.softplus(self.dt_proj(delta))

        return self.selective_scan(u, delta, A, D)

    def pool_state(self, h):
        """Summarise state (b, d_in, n) -> (b, n)."""
        if self.args.h_pool == 'learned':
            s = einsum(h, self.h_pool_w.to(h.dtype), 'b d n, d -> b n')
        else:
            s = h.mean(dim=1)
        return self.h_norm(s)

    def selective_scan(self, u, delta, A, D):
        """Sequential scan with B_t, C_t = split(BC_proj([u_t ; pool(h_(t-1))])).

        u:     (b, l, d_in)
        delta: (b, l, d_in)
        A:     (d_in, n)
        D:     (d_in,)
        """
        b, l, d_in = u.shape
        n = A.shape[1]

        # Discretize A for every time step.
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))

        # Split the fused weight: W [u; h] = Wx u + Wh h.
        Wx = self.BC_proj.weight[:, :d_in]       # (2n, d_in)
        Wh = self.BC_proj.weight[:, d_in:]       # (2n, n)

        # u-part for ALL timesteps in one matmul (no recurrence dependence).
        ux = F.linear(u, Wx).to(deltaA.dtype)    # (b, l, 2n)
        Wh = Wh.to(deltaA.dtype)

        h = torch.zeros((b, d_in, n), device=u.device, dtype=deltaA.dtype)
        ys = []

        for i in range(l):
            # Only the small state-dependent part stays inside the loop.
            bc = ux[:, i] + F.linear(self.pool_state(h), Wh)   # (b, 2n)
            B_t, C_t = bc.chunk(2, dim=-1)                     # each (b, n)

            u_t = u[:, i, :]
            deltaB_u_t = einsum(
                delta[:, i, :], B_t, u_t,
                'b d_in, b n, b d_in -> b d_in n',
            )

            # h_t = deltaA_t * h_(t-1) + deltaB_u_t
            h = deltaA[:, i] * h + deltaB_u_t

            # y_t = C_t h_t + D u_t
            y = einsum(h, C_t, 'b d_in n, b n -> b d_in')
            y = y + u_t * D
            ys.append(y)

        return torch.stack(ys, dim=1).to(u.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


if __name__ == "__main__":
    # Quick sanity check.
    for pool in ('mean', 'learned'):
        args = ModelArgs(d_model=32, n_layer=2, vocab_size=100, h_pool=pool)
        model = Mamba(args)
        ids = torch.randint(0, 100, (2, 12))
        logits = model(ids)
        logits.sum().backward()
        print(pool, logits.shape, "grad on Wh:",
              model.layers[0].mixer.BC_proj.weight.grad[:, args.d_inner:].abs().sum().item())
