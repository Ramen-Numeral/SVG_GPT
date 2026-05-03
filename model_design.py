import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import global_config as cfg
import mup
from mup import MuSharedReadout


####======== the model design is mostly derived from karpathy's implementation


class GPT(nn.Module):
    """base transformer language model adapted for mup."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_mup = getattr(config, "use_mup", False)
        self.drop = nn.Dropout(config.dropout)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)

        # mup requires a special readout layer to scale logits down as width increases
        if self.use_mup:
            self.lm_head = mup.MuReadout(
                in_features=config.n_embd,
                out_features=config.vocab_size,
                bias=False,
                output_mult=getattr(config, "output_mult", 1.0)
            )
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _init_weights(self, module):
        """custom weight init handling mup zeroing rules."""
        if isinstance(module, mup.MuReadout):
            return

        if isinstance(module, nn.Linear):
            if self.use_mup:
                # use mup for weights only
                mup.init.normal_(module.weight, mean=0.0, std=0.02)
                # zeroing out query weights per mup documentation to keep attention stable
                if hasattr(module, 'is_qkv'):
                    with torch.no_grad():
                        module.weight[:self.config.n_embd, :].zero_()
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # no scale on bias
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            if self.use_mup:
                mup.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # crop sequence if it exceeds the block size
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            if targets is not None:
                targets = targets[:, -self.config.block_size:]
            T = idx.size(1)

        pos = torch.arange(T, device=idx.device)

        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """autoregressive sampling loop."""
        self.eval()

        for _ in range(max_new_tokens):
            # isolate the context window
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # append to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class GPTConfig:
    """configuration object for model hyperparameters."""

    def __init__(self, name, n_embd, n_layer, n_head, d_ff,
                 vocab_size, block_size, use_mup, dropout):
        self.name = name
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_mup = use_mup
        self.dropout = dropout

    def to_dict(self, exclude_name=False):
        """returns attributes as a dictionary."""
        data = self.__dict__.copy()
        if exclude_name:
            data.pop('name', None)
        return data


class LayerNorm(nn.Module):
    """custom layernorm with optional bias handling."""

    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        return F.layer_norm(
            x,
            x.shape[-1:],
            self.weight,
            self.bias
        )


# =========================
# causal self-attention
# =========================
class CausalSelfAttention(nn.Module):
    """vanilla multi-head masked attention with mup scaling."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        assert self.head_dim * self.n_head == self.n_embd, "invalid head split"

        # fused qkv projection
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd)

        # tag the layer so we know to zero out queries during init
        self.qkv.is_qkv = True
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.attn_dropout = config.dropout  # store the value
        self.resid_dropout = nn.Dropout(config.dropout)  # for the projection

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # key mup difference: scale attention by 1/d instead of 1/sqrt(d)
        scale = 1.0 / self.head_dim if self.config.use_mup else (1.0 / math.sqrt(self.head_dim))

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
            scale=scale
        )

        # re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


# =========================
# mlp
# =========================
class MLP(nn.Module):
    """standard feed forward network."""

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.n_embd, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


# =========================
# transformer block
# =========================
class Block(nn.Module):
    """pre-norm transformer block."""

    def __init__(self, config):
        super().__init__()

        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # residual connections around attention and mlp
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x