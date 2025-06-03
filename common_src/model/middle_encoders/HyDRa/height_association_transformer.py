import torch.nn as nn

class HeightAssociationTransformer(nn.Module):
    def __init__(self, hat_config):
        super().__init__()
        self.height_bins = hat_config['height_bins']
        dim = hat_config['dim']
        num_heads = hat_config['num_heads']
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img_feats, lidar_feats):
        """
        img_feats: [B, N, H, W, C]
        lidar_feats: [B, N, 1, W, C]
        Returns: fused image features [B, N, H, W, C]
        """
        B, N, H, W, C = img_feats.shape
        D = self.height_bins

        img_seq = img_feats.view(B * N * W, H, C)
        lidar_seq = lidar_feats.view(B * N * W, 1, C).expand(-1, D, -1)

        img_q = self.query_proj(img_seq)
        lidar_k = self.key_proj(lidar_seq)
        lidar_v = self.value_proj(lidar_seq)

        fused, _ = self.attn(img_q, lidar_k, lidar_v)
        fused = self.norm(fused + img_seq)

        return fused.view(B, N, H, W, C)
