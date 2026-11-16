import torch
import torch.nn as nn
import torch.nn.functional as F


# num_kv_heads = num_heads -> MHA
# num_kv_heads == 1 -> MQA
# 1 < num_kv_heads < num_heads -> GQA
class Attention(nn.Module): 
    def __init__(self, num_heads, head_dim, num_kv_heads): 
        super().__init__() 

        # num_heads must be divisible by num_kv_heads 
        if num_heads % num_kv_heads != 0: 
            raise ValueError("num_heads must be divisible by num_kv_heads!") 

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads 
        self.head_dim = head_dim  

        self.scale = head_dim ** -0.5 
        self.groups = num_heads // num_kv_heads 

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor: 
        # query: [B, H, S, D] 
        # value / key: [B, Hkv, S, D] 
        B, H, S, D = query.shape 

        # [B, H, S, D] -> [B, Hkv, G, S, D]
        query = query.reshape(B, self.num_kv_heads, self.groups, S, D)
        # [B, Hkv, S, D] -> [B, Hkv, 1, S, D] 
        value = value.unsqueeze(2)
        # [B, Hkv, S, D] -> [B, Hkv, 1, S, D] 
        key = key.unsqueeze(2) 

        mask = torch.ones(S, S, dtype=torch.bool, device=query.device).triu(diagonal=1)

        scores = query @ key.transpose(-2, -1) * self.scale 
        scores = scores.masked_fill(mask, float("-inf")) 

        # Attn = Softmax (Q * K ^ T / d_k ** 0.5) @ V
        out = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype) @ value 
        return out.reshape(B, H, S, D).transpose(1, 2)


if __name__ == "__main__":
    torch.manual_seed(42)

    B, H, Hkv, S, D = 2, 4, 2, 3, 8

    q = torch.randn(B, H, S, D)
    k = torch.randn(B, Hkv, S, D)
    v = torch.randn(B, Hkv, S, D)

    # 自己实现的 GQA
    attention = Attention(H, D, Hkv)
    out = attention(q, k, v)

    # PyTorch 自带的 GQA
    expected = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=True,
    )

    # PyTorch 输出为 [B, H, S, D]，调整为与自己的输出一致
    expected = expected.transpose(1, 2)

    print("输出形状:", out.shape)
    print("自己的输出:", out[0, 0, 0])
    print("PyTorch 输出:", expected[0, 0, 0])
    print("最大误差:", (out - expected).abs().max().item())

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    print("前向测试通过！")