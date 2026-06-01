#argument 1: Transformer language models undergo a “phase change” during training, 
# during which induction heads form and simultaneously in-context learning improves dramatically.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F #Whats this?

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int): 
        super().__init__() 
        assert d_model % n_heads == 0

       

        self.d_model = d_model #size of the token representation vector
        self.n_heads = n_heads #number of separate attention mechanisms
        self.d_head = d_model // n_heads #representation size per head


        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))
        
      

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.shape

        qkv = self.qkv(x) # (B, T, 3 * C) 
        q, k, v = qkv.chunk(3, dim=-1)
        

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

       

        scores = q @ k.transpose(-2, -1) #every qurey token compares against every key token
        #result: scores.shape = (B, n_heads, T, T) aka attention matrix
        scores = scores / math.sqrt(self.d_head)

        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        y = attn @ v #(B, n_heads, T, d_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.out(y)

        if return_attn:
            return y, attn
        return y

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, use_mlp: bool = False):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        #token representations
        # → LayerNorm
        # → CausalSelfAttention
        # → residual update
        self.use_mlp = use_mlp
        if use_mlp:
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

    def forward(self, x, return_attn: bool = False):
        #residual stream update logic
        if return_attn:
            attn_out, attn = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln1(x))
            attn = None
        
        if self.use_mlp:
            x = x + self.mlp(self.ln2(x))
        
        return x, attn 
    
class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        use_mlp: bool = False,
    ):
        super().__init__()
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, use_mlp=use_mlp)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        self.max_seq_len = max_seq_len

    
    def forward(self, tokens, targets=None, return_attn: bool = False):
        B, T = tokens.shape
        assert T <= self.max_seq_len

        positions = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        all_attn = []

        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn:
                all_attn.append(attn)
            
        x = self.ln_final(x)
        logits = self.unembed(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),#logits.view(-1, logits.size(-1)),
                targets.reshape(-1), #targets.view(-1), ... past commands created errors
            )
        
        if return_attn:
            return logits, loss, all_attn
        
        return logits, loss
