import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from .configuration_konkan import KonkanSmallConfig

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class KonkanBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate_up_proj = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.input_layernorm = RMSNorm(config.d_model)
        self.post_attention_layernorm = RMSNorm(config.d_model)
        self.act = SwiGLU()

    def forward(self, x, cos, sin, mask):
        residual = x
        x = self.input_layernorm(x)
        b, t, c = x.shape
        
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().reshape(b, t, c)
        
        x = residual + self.o_proj(y)
        x = x + self.down_proj(self.act(self.gate_up_proj(self.post_attention_layernorm(x))))
        return x

class KonkanGPT(PreTrainedModel):
    config_class = KonkanSmallConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.d_model // config.n_heads, config.max_len)
        self.layers = nn.ModuleList([KonkanBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.token_emb.weight = self.head.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def forward(self, input_ids, labels=None, attention_mask=None, token_type_ids=None, **kwargs):
        b, t = input_ids.shape
        cos, sin = self.rope(input_ids, t)
        
        # Causal mask logic
        mask = torch.tril(torch.ones(t, t, device=input_ids.device)).view(1, 1, t, t).bool()
        
        if attention_mask is not None:
            # Expand mask to match attention_mask from transformers
            mask = mask & attention_mask.view(b, 1, 1, t).bool()
        
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
            
        logits = self.head(self.norm(x))
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # We remove 'token_type_ids' and 'past_key_values' to keep inputs clean for our forward
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }