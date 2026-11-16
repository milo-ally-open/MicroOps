import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class RotaryEmbedding(nn.Module): 
    def __init__(self, head_size: int, rotary_dim: int, max_position_embedding: int, base: float = 10000): 
        super().__init__() 
        # 合法性判断 
        if rotary_dim % 2 != 0: 
            raise ValueError("rotary_dim must be even!")
        if rotary_dim > head_size: 
            raise ValueError("rotary_dim must be smaller than head_size!") 
        # 存储属性 
        self.head_size = head_size 
        self.rotrary_dim = rotary_dim 
        self.half = rotary_dim / 2 
        # 计算 theta_i = base ^ -2i / d (base=10000)
    
        




