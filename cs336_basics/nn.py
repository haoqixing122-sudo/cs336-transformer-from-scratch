import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange


class Embedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((num_embedding, embedding_dim), **factory_kwargs)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids):
        return self.weight[token_ids]
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, context_length: int, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        context_length: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k
        
        # 1. 计算频率频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 形状: (d_k/2,)
        
        # 2. 创建位置序列 [0, 1, ..., context_length - 1]
        t = torch.arange(context_length, device=device).float()  # 形状: (context_length,)
        
        # 3. 计算所有位置的所有角度 (外积)
        # freqs_matrix 形状: (context_length, d_k/2)
        freqs_matrix = torch.outer(t, freqs)
        
        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (..., context_length, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output



class Linear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_feature, in_feature), **factory_kwargs)
        )
        std = (2 / (in_feature + out_feature)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return torch.einsum("...i,oi->...o", x, self.weight)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # 1. 为了数值稳定性，减去指定维度上的最大值
        # dim=-1 通常是 Transformer 中的隐藏层或词表维度
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        # 2. 计算指数
        exp_x = torch.exp(x_stable)
        
        # 3. 计算分母的各指数之和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 4. 计算最终结果
        return exp_x / sum_exp
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
):
    d_k = Q.size(-1)
    scores = torch.einsum("...nd,...md->...nm", Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask.to(dtype=torch.bool, device=scores.device), float("-inf"))
    probs= torch.softmax(scores,dim=-1)
    output= torch.einsum("...nm,...md->...nd",probs,V)
    return output
class Casualselfattetion(nn.Module):
    def __init__(self,d_modle:int,num_heads:int,bias:bool=False,
                 context_length=None,theta=None,device=None,dtype=None):
        super().__init__()
        assert d_modle%num_heads==0
        self.d_modle=d_modle
        self.num_heads=num_heads
        self.d_k=d_modle//num_heads
        self.q_proj=Linear(d_modle,d_modle,device=device,dtype=dtype)
        self.k_proj=Linear(d_modle,d_modle,device=device,dtype=dtype)
        self.v_proj=Linear(d_modle,d_modle,device=device,dtype=dtype)
        self.output_proj=Linear(d_modle,d_modle,device=device,dtype=dtype)
        if theta is not None and context_length is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, context_length, device=device)
        else:
            self.rope = None

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)-> torch.Tensor:
        b, s, d = x.shape
        q=rearrange(self.q_proj(x),"... s (h d)->... h s d",h=self.num_heads)
        k=rearrange(self.k_proj(x),"... s (h d)->... h s d",h=self.num_heads)
        v=rearrange(self.v_proj(x),"... s (h d)->... h s d",h=self.num_heads)
        if self.rope is not None:
            # 如果没传位置，且 RoPE 需要位置，则生成默认位置
            if token_positions is None:
                # 适配各种 Batch 维度，使用 expand 比 repeat 更高效
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s, device=x.device).expand(*batch_dims, s)
            
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask=torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))
        atten_out=scaled_dot_product_attention(q,k,v,mask=mask)
        atten_out=rearrange(atten_out,"... h s d->... s (h d)")
        return self.output_proj(atten_out)
def silu_fn(in_feature):
    return in_feature*torch.sigmoid(in_feature)
class SwiGlu(nn.Module):
    def __init__(self, diff:int,d_module:int,device=None,dtype=None) -> None:
        super().__init__()
        self.diff=diff
        self.d_moulde=d_module
        self.W1=Linear(d_module,diff,device,dtype)
        self.W3=Linear(d_module,diff,device,dtype)
        self.W2=Linear(diff,d_module,device,dtype)
    def forward(self,x:torch.Tensor):
        gate=silu_fn(self.W1(x))
        signal=self.W3(x)
        return self.W2(gate*signal)
class RMSnorm(nn.Module):
    def __init__(self,d_module:int,eps:float=1e-5,device=None,dtype=None ) -> None:
        super().__init__()
        self.d_module=d_module
        self.eps=eps
        factory_kwrags={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.ones(d_module,**factory_kwrags))

    def forward(self,x:torch.Tensor):
        in_dtype=x.dtype
        x_float=x.to(torch.float32)
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=x_float/torch.sqrt(ms+self.eps)
        result=rms*self.weight
        return result.to(in_dtype)
class TransformerBlock(nn.Module):
    def __init__(self,
                 d_module:int,
                 num_heads:int,
                 context_lenth:int,
                 theta:float,
                 d_ff:int,
                 device=None,
                 dtype=None
                   ) -> None:
        super().__init__()
     
        self.norm1=RMSnorm(d_module,device=device,dtype=dtype)
        self.norm2=RMSnorm(d_module,device=device,dtype=dtype)
        self.atten=Casualselfattetion(d_module,num_heads,context_length=context_lenth,theta=theta,device=device,dtype=dtype)
        self.ffn=SwiGlu(d_ff,d_module,device=device,dtype=dtype)
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None):
        u=x+self.atten(self.norm1(x),token_positions=token_positions)
        y=u+self.ffn(self.norm2(u))
        return y
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 context_length,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 rope_theta,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings=Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype
        )
        self.layers=nn.ModuleList([
            TransformerBlock(
                d_module=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                context_lenth=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.In_final=RMSnorm(d_model,
                              device=device,
                              dtype=dtype)
        self.Im_head=Linear(d_model,vocab_size,device=device,dtype=dtype)
    def forward(self,token_ids):
        b,s=token_ids.shape
        token_positions = (
            torch.arange(s, device=token_ids.device)
            .unsqueeze(0)
            .expand(b, s)
        )
        x=self.token_embeddings(token_ids)
        for layer in self.layers:
            x=layer(
                x,
                token_positions=token_positions)
        x=self.In_final(x)
        logits=self.Im_head(x)
        return logits
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if prompt_ids.ndim != 2 or prompt_ids.size(1) == 0:
            raise ValueError("prompt_ids must have shape [batch, sequence] with a nonempty sequence")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        was_training = self.training
        self.eval()
        try:
            generated = prompt_ids.clone()
            finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device)
            for _ in range(max_new_tokens):
                idx_cond = generated[:, -self.context_length:]
                logits = self(idx_cond)[:, -1, :]
                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_p < 1.0:
                        logits = self._top_p_filter(logits, top_p)
                    probs = softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                if eos_token_id is not None:
                    # Keep completed rows at EOS while the other rows continue.
                    next_token = next_token.masked_fill(finished.unsqueeze(-1), eos_token_id)
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and finished.all():
                    break
            return generated
        finally:
            self.train(was_training)

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))
