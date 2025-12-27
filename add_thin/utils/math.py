import math
import torch

def modulate(x, shift, scale):
    # scale = torch.clamp(scale, -1, 1)
    return x * (1 + scale[:, None]) + shift[:, None]

# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length, device=None):
    pos = torch.arange(length, dtype=torch.float32, device=device)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return emb.unsqueeze(0)

def get_2d_sincos_pos_embed(rng, embed_dim, length, device=None):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    print(f"get_2d_sincos_pos_embed: embeded_dimn : {embed_dim}, length: {length}, grid_size: {grid_size}")
    assert grid_size * grid_size == length
    
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
        return emb

    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0)  # (1, H*W, D)