"""
Definition of a encoder-decoder transformer model for machine translation.
Note that positional encoding is provided in the separate file.
References:
1) nanoGPT: 
    https://github.com/karpathy/nanoGPT/tree/master
2) Pytorch implementation of "Attention is All You Need" for de-en translation task:
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config: dataclass) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
        x: (batch size, sequence length, embedding dimensionality)
        embedding dimensionality = number of heads * head size
        '''
        B, T, C = x.size()
        hs = C // self.n_head
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # -> (B, n_head, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # -> (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # -> (B, n_head, T, hs)
        out = F.scaled_dot_product_attention(q, k, v, 
                                             attn_mask=None, 
                                             dropout=self.dropout if self.training else 0, 
                                             is_causal=True) # -> (B, n_head, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out # -> (B, T, C)

class MultiHeadCrossAttention(nn.Module):
    '''
    Following "Attention Is All You Need" section 3.2.3 and reference 2), 
    queries are from the inputs to the decoder, and keys and values come from the encoder.
    '''
    def __init__(self, config: dataclass) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.enc_q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dec_kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, dec_x: torch.Tensor, enc_x: torch.Tensor) -> torch.Tensor:
        '''
        enc_x is the input from the encoder, dec_x is from the decoder.
        Both has size (batch size, sequence length, embedding dimensionality).
        embedding dimensionality = number of heads * head size
        '''
        B, T, C = enc_x.size() # enc_x.size() = dec_x.size()
        hs = C // self.n_head
        q = self.enc_q_proj(enc_x)
        k, v = self.dec_kv_proj(dec_x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # -> (B, n_head, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # -> (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # -> (B, n_head, T, hs)
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=None, 
                                             dropout=self.dropout if self.training else 0, 
                                             is_causal=True) # -> (B, n_head, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out # -> (B, T, C)

class MLP(nn.Module):

    def __init__(self, config: dataclass) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) 
        # The hidden layer size of 4 * n_embd looks arbitrary.
        match config.activation:
            case "gelu":
                self.activation = nn.GELU()
            case "relu":
                self.activation == F.relu()
            case _:
                self.activation == F.relu()
        self.out_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.activation(out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out # -> (B, T, C)
    
class EncoderBlock(nn.Module):

    def __init__(self, config: dataclass) -> None:
        super().__init__()
        self.ln_attn = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_mlp = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln_attn(x)) + x
        x = self.mlp(self.ln_mlp(x)) + x
        return x # -> (B, T, C)
    
class DecoderBlock(nn.Module):

    def __init__(self, config: dataclass) -> None:
        super().__init__()
        self.dec_ln_self_attn = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn = MultiHeadSelfAttention(config)
        self.dec_ln_cross_attn = LayerNorm(config.n_embd, bias=config.bias)
        self.enc_ln_cross_attn = LayerNorm(config.n_embd, bias=config.bias)
        self.cross_attn = MultiHeadCrossAttention(config)
        self.ln_mlp = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, dec_x: torch.Tensor, enc_x: torch.Tensor) -> torch.Tensor:
        dec_x = self.self_attn(self.dec_ln_self_attn(dec_x)) + dec_x
        dec_x = self.cross_attn(self.dec_ln_cross_attn(dec_x), self.enc_ln_cross_attn(enc_x)) + dec_x
        dec_x = self.mlp(self.ln_mlp(dec_x)) + dec_x
        return dec_x # -> (B, T, C)
    
@dataclass
class ModelConfig:
    block_size: int = 1024 # Maximum sequence length
    enc_vocab_size: int = 50304 # 64 * 786
    dec_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 # head size = n_embd / n_head = 64
    dropout: float = 0.0
    bias: bool = True
    activation: str = "relu"

class EncDecModel(nn.Module):

    def __init__(self, config: dataclass) -> None:
        super().__init__()
        assert config.enc_vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.enc_x = None # Keep the encoder's output for the inference.
        self.transformer = nn.ModuleDict(dict(
            enc_wte = nn.Embedding(config.enc_vocab_size, config.n_embd),
            dec_wte = nn.Embedding(config.dec_vocab_size, config.n_embd),
            enc_wpe = nn.Embedding(config.block_size, config.n_embd),
            dec_wpe = nn.Embedding(config.block_size, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.linear_head = nn.Linear(config.n_embd, config.dec_vocab_size, bias=False)
        self.transformer.dec_wte.weight = self.linear_head.weight # Weight tying introduced in https://arxiv.org/abs/1608.05859
        # Initialize weights.
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # Report number of parameters.
        print(f'Number of parameters: {self.get_num_params():,}')
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        n_params = sum(param.numel() for param in self.parameters())
        return n_params
    
    def forward(self,
                 dec_idx: torch.Tensor, 
                 enc_idx: torch.Tensor = None, 
                 targets: torch.Tensor = None) -> torch.Tensor | torch.Tensor:
        device = enc_idx.device
        b, enc_t = enc_idx.size() # batch size, sequence length
        assert enc_t <= self.config.block_size
        _b, dec_t = dec_idx.size()
        assert dec_t <= self.config.block_size
        assert b == _b

        if enc_idx is None:
            enc_pos = torch.arange(0, enc_t, dtype=torch.long, device=device) # -> (enc_t)
            enc_tok_emb = self.transformer.enc_wte(enc_idx) # -> (b, enc_t, C)
            enc_pos_emb = self.transformer.enc_wpe(enc_pos) # -> (b, enc_t, C)
            self.enc_x = enc_tok_emb + enc_pos_emb # -> (b, enc_t, C)
            self.enc_x = self.transformer.dropout(self.enc_x) # -> (b, enc_t, C)
            for enc_block in self.transformer.encoder:
                self.enc_x = enc_block(self.enc_x) # -> (b, enc_t, C)
        
        dec_pos = torch.arange(0, dec_t, dtype=torch.long, device=device) # -> (dec_t)
        dec_tok_emb = self.transformer.dec_wte(dec_idx) # -> (b, dec_t, C)
        dec_pos_emb = self.transformer.dec_wpe(dec_pos) # -> (b, dec_t, C)
        dec_x = dec_tok_emb + dec_pos_emb # -> (b, dec_t, C)
        dec_x = self.transformer.dropout(dec_x) # -> (b, dec_t, C)
        for dec_block in self.transformer.decoder:
            dec_x = dec_block(dec_x, self.enc_x) # -> (b, dec_t, C)
        
        dec_x = self.transformer.ln_f(dec_x) # -> (b, dec_t, C)
        if targets is not None: # Training phase
            logits = self.lm_head(dec_x) # -> (b, dec_t, dec_vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: # Inference phase
            logits = self.lm_head(dec_x[:, [-1], :])
            loss = None
        
        return logits, loss

    def configure_optimizers(self, 
                             weight_decay: float, 
                             learning_rate: float, 
                             betas: tuple[float, float],
                             device_type: str):
        param_dict = {name: param for name, param in self.named_parameters()}
        param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}
        # Following reference 1), 2+ dimensional parameters will be weight decayed, otherwise no.
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for name, param in param_dict.items() if param.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, flops_promised: int = 312e12) -> int:
        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        N = self.get_num_params()
        L = self.config.n_layer
        H = self.config.n_head
        Q = self.config.n_embd // self.config.n_head
        T = self.config.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.no_grad()
    def generate(self, 
                 dec_idx: torch.Tensor, 
                 enc_idx: torch.Tensor, 
                 max_new_tokens: int, 
                 temperature: float = 1.0, 
                 top_k: int = None) -> torch.Tensor:
        assert enc_idx.size(1) <= self.config.block_size
        for i in range(max_new_tokens):
            dec_idx_cond = dec_idx if dec_idx.size(1) <= self.config.block_size else dec_idx[:, -self.block_size:]
            if i == 0:
                logits, _ = self(dec_idx=dec_idx_cond, enc_idx=enc_idx)
            else:
                logits, _ = self(dec_idx=dec_idx_cond, enc_idx=None)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            dec_idx = torch.cat((dec_idx, idx_next), dim=1)
        return dec_idx

