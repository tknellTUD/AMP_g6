import torch.nn as nn

class HeightAssociationTransformer(nn.Module):
    def __init__(self, hat_config):
        super().__init__()
        dim = hat_config['dim']
        num_heads = hat_config['num_heads']
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img_seq, lidar_seq):
        """
        img_feats: [B*W, H, C]
        lidar_feats: [B * W, D, C]
        Returns: fused image features [B * W, H, C]
        """

        img_q = self.query_proj(img_seq)
        lidar_k = self.key_proj(lidar_seq)
        lidar_v = self.value_proj(lidar_seq)

        fused, _ = self.attn(img_q, lidar_k, lidar_v)
        fused = self.norm(fused + img_seq)
        # print(f"fused shape: {fused.shape}")

        return fused
