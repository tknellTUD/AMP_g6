import torch.nn as nn

class ZeroInitLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__(hidden_dim, eps=eps, elementwise_affine=True)
        # γ ← 0,  β ← 0
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)