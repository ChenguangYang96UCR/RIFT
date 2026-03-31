from add_thin.data import Batch
from add_thin.backbones.cnn import CNNSeqEmb
from add_thin.backbones.embeddings import NyquistFrequencyEmbedding
from add_thin.processes.hpp import generate_hpp
from add_thin.diffusion.utils import betas_for_alpha_bar
from add_thin.utils.math import get_2d_sincos_pos_embed, modulate

import math
import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import torch.nn.functional as F
from einops import rearrange
import jax.numpy as jnp
from jax._src import dtypes
from flax import linen as linen
import jax
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import tensorflow.compat.v1 as tf

patch_typeguard()

def xavier_uniform_init(tensor):
    """PyTorch-like Xavier uniform initialization"""
    if len(tensor.shape) == 2:  # Dense layer
        fan_in, fan_out = tensor.shape[0], tensor.shape[1]
    elif len(tensor.shape) == 4:  # Conv layer
        fan_in = tensor.shape[1] * tensor.shape[2] * tensor.shape[3] 
        fan_out = tensor.shape[0]
    else:
        raise ValueError(f"Invalid tensor shape {tensor.shape}")
    
    variance = 2.0 / (fan_in + fan_out)
    scale = math.sqrt(3 * variance)
    with torch.no_grad():
        tensor.uniform_(-scale, scale)
    return tensor

class TrainConfig:
    def __init__(self):
        pass
        # self.dtype = dtype
    
    def apply_init(self, module, name='default', zero=False):
        """Apply initialization to a module"""
        if zero or 'bias' in name:
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if hasattr(module, 'weight'):
                nn.init.constant_(module.weight, 0)
        else:
            if hasattr(module, 'weight'):
                xavier_uniform_init(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Initialize with normal distribution
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.mlp(x)
        return x.unsqueeze(0)    # → (1, hidden_size) 
    
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        t = t.float()
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args).mean(dim=0), torch.sin(args).mean(dim=0)], dim=-1)
        return embedding


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert timesteps.dim() == 1, "timesteps must be 1D"

    device = timesteps.device
    half_dim = embedding_dim // 2

    # log(10000) / (half_dim - 1)
    emb_scale = math.log(10000) / (half_dim - 1)

    # exp(range * -scale)
    emb = torch.exp(
        torch.arange(half_dim, device=device) * -emb_scale
    )

    # timesteps[:, None] * emb[None, :]
    emb = timesteps.to(torch.float32)[:, None] * emb[None, :]

    # concat sin and cos
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    # zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class EventEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, e):
        # event_embedding returns shape (frequency_embedding_size,)
        emb = self.event_embedding(e)   # (D,)
        emb = self.mlp(emb)             # (hidden_size,)
        return emb.unsqueeze(0)         # → (1, hidden_size)

    def event_embedding(self, e):
        """
        e: shape (N,)
        Returns a single embedding vector: shape (frequency_embedding_size,)
        """
        e = e.float()
        dim = self.frequency_embedding_size
        half = dim // 2

        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=e.device) / half
        )

        args = e[:, None] * freqs[None]   # → (N, half)

        # Pool over the event dimension
        cos_part = torch.cos(args).mean(dim=0)  # (half,)
        sin_part = torch.sin(args).mean(dim=0)  # (half,)

        # Final freq embedding: (frequency_embedding_size,)
        return torch.cat([cos_part, sin_part], dim=-1)
    

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, patch_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        
        # Apply Xavier uniform initialization
        xavier_uniform_init(self.proj.weight)
        if bias:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        num_patches = H // self.patch_size
        x = self.proj(x)  # (B, hidden_size, P, P)
        x = rearrange(x, 'b c h w -> b (h w) c', h=num_patches, w=num_patches)
        return x

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, in_dim: int, mlp_dim: int, out_dim: Optional[int] = None, 
                 dropout_rate: float = 0.0):
        super().__init__()
        actual_out_dim = in_dim if out_dim is None else out_dim
        
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, actual_out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Apply Xavier uniform initialization
        xavier_uniform_init(self.fc1.weight)
        xavier_uniform_init(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.mlp = MlpBlock(hidden_size, int(hidden_size * mlp_ratio), dropout_rate=dropout)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Apply Xavier initialization
        xavier_uniform_init(self.qkv.weight)
        xavier_uniform_init(self.attn_proj.weight)
        xavier_uniform_init(self.adaLN_modulation[1].weight)

    def forward(self, x, c):

        # x: (B, C) → (B, 1, C)
        # if x.dim() == 2:
        #     print(f'x need to unsqueeze')
        #     x = x.unsqueeze(1)
        # Calculate adaLN modulation parameters
        adaLN_out = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_out.chunk(6, dim=1)
        
        # Attention block
        x_norm = self.norm1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)

        B, N, C = x_modulated.shape
        qkv = self.qkv(x_modulated).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Attention computation
        q = q * (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_proj(out)
        
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP block
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated2)
        
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        # (B, 1, C) → (B, C)
        x = x.squeeze(1)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, patch_size: int, out_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        # Initialize with zeros
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        
        adaLN_out = self.adaLN_modulation(c)
        shift, scale = adaLN_out.chunk(2, dim=1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

class MLP(nn.Module):
    def __init__(self, L, out_dim, hidden=None, dropout=0.0):
        super().__init__()
        if hidden is None:
            # just one linear projection
            self.mlp = nn.Linear(L, out_dim)
        else:
            # 2-layer MLP with nonlinearity
            self.mlp = nn.Sequential(
                nn.Linear(L, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim)
            )

    def forward(self, x, number_patches):
        B, L = x.shape
        pad_len = (number_patches - (L % number_patches)) % number_patches
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode="constant", value=0)
        x = x.view(x.size(0), number_patches, -1)
        return self.mlp(x)

class DiT_MLP(nn.Module):
    def __init__(self, L, out_dim, hidden=None, dropout=0.0):
        super().__init__()
        if hidden is None:
            # just one linear projection
            self.mlp = nn.Linear(L, out_dim)
        else:
            # 2-layer MLP with nonlinearity
            self.mlp = nn.Sequential(
                nn.Linear(L, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim)
            )

    def forward(self, x):
        return self.mlp(x)

@typechecked
class DiffusionModell(nn.Module):
    """
    Base class for diffusion models.

    Parameters
    ----------
    steps : int, optional
        Number of diffusion steps, by default 100
    """

    def __init__(self, steps: int = 1000) -> None:
        super().__init__()
        self.steps = steps

        # Cosine beta schedule
        beta = betas_for_alpha_bar(
            steps,
            lambda n: math.cos((n + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

        # Compute alpha and alpha_cumprod
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

        # Compute thinning probabilities for posterior and shift it to be indexed with n-1
        add_remove = (1 - self.alpha_cumprod)[:-1] * beta[1:]
        alpha_x0_kept = (self.alpha_cumprod[:-1] - self.alpha_cumprod[1:]) / (
            1 - self.alpha_cumprod[1:]
        )
        alpha_xn_kept = (
            (self.alpha - self.alpha_cumprod) / (1 - self.alpha_cumprod)
        )[1:]

        self.register_buffer("alpha_x0_kept", alpha_x0_kept)
        self.register_buffer("alpha_xn_kept", alpha_xn_kept)
        self.register_buffer("add_remove", add_remove)


@typechecked
class AddThin(DiffusionModell):
    """
    Implementation of AddThin (Add and Thin: Diffusion for Temporal Point Processes).

    Parameters
    ----------
    classifier_model : nn.Module
        Model for predicting the intersection of x_0 and x_n from x_n
    intensity_model : nn.Module
        Model for predicting the intensity of x_0 without x_n
    max_time : float
        T of the temporal point process
    n_max : int, optional
        Maximum number of events, by default 100
    steps : int, optional
        Number of diffusion steps, by default 100
    hidden_dims : int, optional
        Hidden dimensions of the models, by default 128
    emb_dim : int, optional
        Embedding dimensions of the models, by default 32
    encoder_layer : int, optional
        Number of encoder layers, by default 4
    kernel_size : int, optional
        Kernel size of the CNN, by default 16
    forecast : None, optional
        If not None, will turn the model into a conditional one for forecasting
    """

    def __init__(
        self,
        classifier_model,
        intensity_model,
        max_time: float,
        n_max: int = 100,
        steps: int = 100,
        hidden_dims: int = 128,
        emb_dim: int = 32,
        encoder_layer: int = 4,
        kernel_size: int = 16,
        forecast=None,
        patch_size: int = 3,
        hidden_size: int = 128,
        depth: int = 2,
        num_heads: int = 2,
        mlp_ratio: float = 1,
        out_channels: int = 1,
        class_dropout_prob: float = 0.1,
        ignore_dt: bool = False,
        dropout: float = 0.0,
        input_size: int = 256,
        k_steps: int = 10,
        lambda_1: float = 1.0
    ) -> None:
        super().__init__(steps)
        # Set models parametrizing the approximate posterior
        self.classifier_model = classifier_model
        self.intensity_model = intensity_model
        self.temp_x_N = None
        self.n_max = n_max

        self.patch_size = patch_size
        # if patch size is 3, then whole numbers of patches is 9
        self.number_patches = self.patch_size * self.patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.class_dropout_prob = class_dropout_prob
        self.ignore_dt = ignore_dt
        self.input_size = input_size
        self.k_steps = k_steps
        self.lambda_1 = lambda_1
        self.lambda_0_hat = None
        
        # Components
        self.patch_embed = PatchEmbed(patch_size, hidden_size)
        self.time_embed = TimestepEmbedder(hidden_size)
        self.dt_embed = TimestepEmbedder(hidden_size)
        self.event_embed = EventEmbedder(hidden_size)

        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(patch_size, out_channels, hidden_size)
        
        # Logvar embedding for timestep
        self.logvar_embed = nn.Embedding(256, 1)
        nn.init.constant_(self.logvar_embed.weight, 0)
        
        self.initialize_weights()

        # Init forecast settings
        if forecast:
            self.forecast = True
            self.history_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim,
                batch_first=True,
            )
            self.history_mlp = nn.Sequential(
                nn.Linear(2 * emb_dim, hidden_dims), nn.ReLU()
            )
            self.forecast_window = forecast
        else:
            self.forecast = False
            self.history = None

        self.set_encoders(
            hidden_dims=hidden_dims,
            max_time=max_time,
            emb_dim=emb_dim,
            encoder_layer=encoder_layer,
            kernel_size=kernel_size,
            steps=steps,
        )

    def initialize_weights(self):
        # Initialize positional embedding
        num_patches = (self.input_size // self.patch_size) ** 2
        pos_embed = get_2d_sincos_pos_embed(None, self.hidden_size, num_patches)
        self.pos_embed.data.copy_(pos_embed)

    def set_encoders(
        self,
        hidden_dims: int,
        max_time: float,
        emb_dim: int,
        encoder_layer: int,
        kernel_size: int,
        steps: int,
    ) -> None:
        """
        Set the encoders for the model.

        Parameters
        ----------
        hidden_dims : int
            Hidden dimensions of the models
        max_time : float
            T of the temporal point process
        emb_dim : int
            Embedding dimensions of the models
        encoder_layer : int
            Number of encoder layers
        kernel_size : int
            Kernel size of the CNN
        steps : int
            Number of diffusion steps
        """
        # Event time encoder
        position_emb = NyquistFrequencyEmbedding(
            dim=emb_dim // 2, timesteps=max_time
        )
        self.time_encoder = nn.Sequential(position_emb)

        # Diffusion time encoder
        position_emb = NyquistFrequencyEmbedding(dim=emb_dim, timesteps=steps)
        self.diffusion_time_encoder = nn.Sequential(
            position_emb,
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Event sequence encoder
        self.sequence_encoder = CNNSeqEmb(
            emb_layer=encoder_layer,
            input_dim=hidden_dims,
            emb_dims=hidden_dims,
            kernel_size=kernel_size,
        )

    def set_history(self, batch: Batch) -> None:
        B, L = batch.time.shape

        if B == 0 or L == 0:
            device = batch.time.device
            emb_dim = self.history_encoder.hidden_size
            self.history = torch.zeros((0, emb_dim), device=device, dtype=batch.time.dtype)
            return

        time_emb = self.time_encoder(
            torch.cat(
                [batch.time.unsqueeze(-1), batch.tau.unsqueeze(-1)], dim=-1
            )
        ).reshape(B, L, -1)

        embedding = self.history_encoder(time_emb)[0]

        index = (batch.mask.sum(-1).long() - 1).unsqueeze(-1).unsqueeze(-1)
        gather_index = index.repeat(1, 1, embedding.shape[-1])
        self.history = embedding.gather(1, gather_index).squeeze(-2)

    def compute_emb(
        self, n: TensorType[torch.long, "batch"], x_n: Batch
    ) -> Tuple[
        TensorType["batch", "embedding"],
        TensorType["batch", "sequence", "embedding"],
        TensorType["batch", "sequence", "embedding"],
    ]:
        """
        Get the embeddings of x_n.

        Parameters
        ----------
        n : TensorType[torch.long, "batch"]
            Diffusion time step
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType["batch", "embedding"],
            TensorType["batch", "sequence", "embedding"],
            TensorType["batch", "sequence", "embedding"],
        ]
            Diffusion time embedding, event time embedding, event sequence embedding
        """
        device = x_n.time.device
        B, L = x_n.batch_size, x_n.seq_len

        # embed diffusion and process time
        dif_time_emb = self.diffusion_time_encoder(n)

        # Condition ADD-THIN on history by adding it to the diffusion time embedding
        if self.forecast:
            dif_time_emb = self.history_mlp(
                torch.cat([self.history, dif_time_emb], dim=-1)
            )

        # Embed event and interevent time
        time_emb = self.time_encoder(
            torch.cat([x_n.time.to(device).unsqueeze(-1), x_n.tau.to(device).unsqueeze(-1)], dim=-1)
        ).reshape(B, L, -1)

        # Embed event sequence and mask out
        event_emb = self.sequence_encoder(time_emb)
        event_emb = event_emb * x_n.mask[..., None]

        return (
            dif_time_emb,
            time_emb,
            event_emb,
        )

    def get_n(self, shape, device, min=None, max=None) -> TensorType[int]:
        """
        Uniformly sample n, i.e., the diffusion time step per sequence.

        Parameters
        ----------
        shape :
            Shape of the tensor
        device :
            Device of the tensor
        min : None, optional
            Minimum value of n, by default None
        max : None, optional
            Maximum value of n, by default None

        Returns
        -------
        TensorType[int]
            Sampled n
        """
        if min is None or max is None:
            min = 0
            max = self.steps
        return torch.randint(
            min,
            max,
            size=shape,
            device=device,
            dtype=torch.long,
        )


    def DiT(self, x, t, dt, train=False, return_activations=False):
        activations = {}
        data_length = x.shape[0]
        
        # Ensure all tensors are on the same device
        device = x.device
        
        # Convert tensors to appropriate device and dtype if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, device=device)
        
        activations['patch_embed'] = x
        
        # Get timestep embeddings
        # TODO exchange shape from (sequence, embeding) to (sequence, )
        e = x.mean(dim=-1)          # (sequence,)
        x = self.event_embed(e)  # (B, hidden_size)
        t_mean = t.mean(dim=-1)
        te = self.time_embed(t_mean)  # (B, hidden_size)
        dt_mean = dt.mean(dim=-1)
        dte = self.dt_embed(dt_mean)  # (B, hidden_size)
        c = te + dte
        
        # activations['pos_embed'] = pos_embed
        activations['time_embed'] = te
        activations['dt_embed'] = dte
        activations['conditioning'] = c
        
        # Apply DiT blocks
        for i in range(2):
            x = self.blocks[i](x, c)
            activations[f'dit_block_{i}'] = x
        
        # Apply final layer
        x = self.final_layer(x, c)  # (B, num_patches, p*p*c)
        activations['final_layer'] = x
        
        # Flatten for MLP processing
        x_flat = x.reshape(x.shape[0], -1)
        
        # Apply final MLP
        # print(f"DiT_MLP shape , x_flat.shape[1]: {x_flat.shape[1]}, x_flat.shape[0]: {x_flat.shape[0]} data_length : {data_length}")
        mlp = DiT_MLP(x_flat.shape[1], data_length, 1).to(device)
        if train:
            mlp.train()
        else:
            mlp.eval()
        
        e_new = mlp(x_flat)
        scalar = e_new.mean(dim=1).mean()        
        return scalar

    def lambda_0_hat(e_bar, e_0, bandwidth_square):
            kernel_values = np.exp(-((e_bar.reshape((-1, 1)) - e_0.reshape((1, -1))) **2 ) / (bandwidth_square))
            return np.sum(kernel_values, axis = 1) / len(e_0)

    def random_sample_by_interval(self, max_val=1, interval=0.01):
        samples = np.arange(0, max_val + interval, interval)
        return samples

    def build_tau(self, time, mask, tmax):
        # time, mask: torch tensors [B, L]
        # tmax: [B] or scalar

        # 用 tmax 填充 padding
        time_tau = torch.where(mask, time, tmax[:, None] if tmax.ndim == 1 else tmax)

        # diff along sequence
        tau = torch.diff(
            time_tau,
            prepend=torch.zeros_like(time_tau[:, :1]),
            dim=1
        )

        # mask padding
        tau = tau * mask
        return tau

    def scaled_sampling_batchwise(self, k, lambda_1, e_0, e_bar, T, r_k, r_k1, gap=0.1, bandwidth_square=1.5):
        device = e_0.device
        B, L = e_0.shape

        if e_bar is None:
            e_bar = e_0

        # ===== lambda_0_hat on GPU =====
        # [B,L,L]
        def kernel_mean_chunked(e_bar, e_0, bandwidth_square, chunk_size=256):
            B, L = e_bar.shape
            out = torch.zeros(B, L, device=e_bar.device, dtype=e_bar.dtype)

            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                diff = e_bar[:, start:end, None] - e_0[:, None, :]   # [B, chunk, L]
                kernel = torch.exp(-diff.pow(2) / bandwidth_square)
                out[:, start:end] = kernel.mean(dim=2)
            return out
        lambda0_hat = kernel_mean_chunked(e_bar, e_0, bandwidth_square)

        # ===== omega =====
        omega = (lambda_1 ** (r_k - r_k1)) * (lambda0_hat ** (r_k1 - r_k))   # [B,L]

        # ===== thinning =====
        keep = torch.rand_like(omega) < omega
        e_thin = torch.where(keep, e_bar, torch.zeros_like(e_bar))

        # ===== upper bound =====
        upper = 2 * ((lambda_1 ** (k * gap)) * (omega ** (1 - k * gap))).amax(dim=1) + 2

        # ===== add =====
        n_add = torch.poisson(upper * T)      # [B]

        all_new = []
        all_mask = []

        chunk_size = 512
        for b in range(B):
            Nb = int(n_add[b].item())
            accepted_chunks = []

            if Nb > 0:
                for start in range(0, Nb, chunk_size):
                    curr_size = min(chunk_size, Nb - start)
                    e_chunk = torch.rand(curr_size, device=device) * T   # [curr_size]

                    diff = e_chunk[:, None] - e_0[b][None, :]            # [curr_size, L]
                    kernel = torch.exp(-diff**2 / bandwidth_square)
                    lambda0_add = kernel.mean(dim=1)                     # [curr_size]

                    omega_add = (lambda_1 ** (r_k - r_k1)) * (lambda0_add ** (r_k1 - r_k))
                    lambda_bar = (lambda_1 ** r_k1) * (lambda0_add ** (1 - r_k1))

                    p = (omega_add - 1) * lambda_bar / upper[b]
                    p = torch.clamp(p, 0, 1)

                    accept = torch.rand_like(p) < p
                    accepted_chunks.append(e_chunk[accept])

                e_add = (
                    torch.cat(accepted_chunks)
                    if len(accepted_chunks) > 0
                    else torch.empty(0, device=device)
                )
            else:
                e_add = torch.empty(0, device=device)

            merged = torch.sort(torch.cat([e_thin[b][e_thin[b] > 0], e_add]))[0]
            all_new.append(merged)

        # pad
        max_len = max(len(x) for x in all_new)
        time = torch.zeros(B, max_len, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for b, seq in enumerate(all_new):
            time[b, :len(seq)] = seq
            mask[b, :len(seq)] = True

        tau = self.build_tau(time, mask, T)

        return Batch(time=time, mask=mask, tau=tau, tmax=T,
                    unpadded_length=mask.sum(1), kept=None)

    
    def approximate_lambda_0_hat(self, lambda_1, e_rk_array):
        """
        e_rk_array: [steps, batch, value]
        returns:    [value]
        """

        # E_batch[ · ]
        term_k = e_rk_array.mean(dim=1)    # [steps, value]

        # sum over k (path integral)
        path_integral = term_k.sum(dim=0)  # [value]

        lambda_0_hat = lambda_1 * torch.exp(path_integral)
        return lambda_0_hat


    def forward(
        self, x_0: Batch
    ):
        """
        Forward pass to train the model, i.e., predict x_0 from x_n.

        Parameters
        ----------
        x_0 : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType[float, "batch", "sequence_x_n"],
            TensorType[float, "batch"],
            Batch,
        ]
            classification logits, log likelihood of x_0 without x_n, noised data
        """
        # Uniformly sample n
        device = x_0.time.device
        n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_0),),
            device=device,
        )
        
        # Initialize the sample parameter
        r_k = 0.1
        M = 100
        r_k1 = r_k - (1/self.k_steps)
        
        time_rk1 = x_0.time * x_0.mask
        e_new = []
        time = []
        time_0 = x_0.time * x_0.mask
        batch_size = time_0.shape[0]

        # noise the event
        for i in range(1, self.k_steps): 
            bew_batch = self.scaled_sampling_batchwise(i, self.lambda_1, time_0, time_rk1, x_0.tmax, r_k, r_k1)
            time_0 = time_rk1
            time_rk1 = bew_batch.time * bew_batch.mask
            # e_new.append(e_rk)

            # Generate event embeding 
            # TODO update tau
            (dif_time_emb, time_emb, event_emb) = self.compute_emb(n=n, x_n=bew_batch)
            e_new.append(event_emb)
            time.append(time_emb)

        result_rk_array = []

        # For each step
        for index, e in enumerate(e_new):
            batrch_result_array = []
            for batch_index in range(batch_size):
                batch_e = e[batch_index].to(torch.float32).to(device)

                # Generate random timesteps using PyTorch instead of JAX
                # t = torch.rand(self.hidden_size, device=e.device)
                # dt = torch.rand(time[index][batch_index].shape[0], time[index][batch_index].shape[1], device=batch_e.device)
                dt = get_timestep_embedding(
                    torch.tensor([index], device=device),
                    self.hidden_size
                )
                result = self.DiT(batch_e, time[index][batch_index], dt)
                batrch_result_array.append(result)
            result_rk_array.append(batrch_result_array)

        sample_result_rk_array = []

        # For each step
        for index, e in enumerate(e_new):
            batrch_result_array = []
            for batch_index in range(batch_size):
                batch_e = e[batch_index].to(torch.float32).to(device)

                # Generate random timesteps using PyTorch instead of JAX
                # t = torch.rand(self.hidden_size, device=e.device)
                time_smaple = self.random_sample_by_interval(max_val=1)
                time_embeding = get_timestep_embedding(torch.from_numpy(time_smaple), self.hidden_size)

                # dt = torch.rand(time[index][batch_index].shape[0], time_embeding, device=batch_e.device)
                dt = get_timestep_embedding(
                    torch.tensor([index], device=device),
                    self.hidden_size
                )
                result = self.DiT(batch_e, time_embeding.to(device), dt)
                batrch_result_array.append(result)
            sample_result_rk_array.append(batrch_result_array)

        # calculate approximate lambda_0 hat
        # print(f"forward e_rk array length {len(e_rk_array)}, stack array length {torch.stack(e_rk_array).shape}")
        lambda_0_hat = self.approximate_lambda_0_hat(self.lambda_1, torch.tensor(result_rk_array))
        # self.lambda_0_hat = lambda_0_hat
        
        return torch.stack([torch.stack(x) for x in result_rk_array]).to(device), self.lambda_1, self.k_steps, torch.tensor(sample_result_rk_array).to(device), x_0.tmax

    def build_batch_from_events(self, event_lists, tmax, device):
        """
        event_lists: List[np.ndarray], length = B
        each array shape = [L_b]
        """

        B = len(event_lists)
        lengths = torch.tensor([len(x) for x in event_lists], device=device)
        max_len = lengths.max().item()

        # ---- time & mask ----
        time = torch.zeros((B, max_len), device=device)
        mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)

        for b, ev in enumerate(event_lists):
            if len(ev) > 0:
                time[b, :len(ev)] = torch.tensor(ev, device=device)
                mask[b, :len(ev)] = True

        # ---- tau ----
        # padding 用 tmax 填
        if not torch.is_tensor(tmax):
            tmax = torch.tensor(tmax, device=device)
        else:
            tmax = tmax.to(device)

        time_tau = torch.where(mask, time, tmax)
        tau = torch.diff(
            time_tau,
            prepend=torch.zeros((B, 1), device=device),
            dim=1
        )
        tau = tau * mask

        return time, mask, tau, lengths


    def backward_sample(self, x_n, lambda_rk, omega_t):

        def backward_sample_single(time_b, lambda_rk_b, omega_b, tmax):
            kept = []
            for t in time_b:
                if t <= 0:
                    continue
                if np.random.rand() < min(omega_b, 1.0):
                    kept.append(t)

            # build upper bound
            ts = np.linspace(0, tmax, 1000)
            lambda_plus = np.maximum(omega_b - 1, 0) * lambda_rk_b
            M = lambda_plus

            N = np.random.poisson(M * tmax)

            u = np.random.uniform(0, tmax, size=N)
            v = np.random.uniform(0, M, size=N)

            added = []
            for ti, vi in zip(u, v):
                if vi < max(omega_b - 1, 0) * lambda_rk_b:
                    added.append(ti)

            return np.sort(np.concatenate([kept, added]))
        
        B = x_n.time.shape[0]
        event_lists = []
        for b in range(B):
            e_new_b = backward_sample_single(
                x_n.time[b].cpu().numpy(),
                float(lambda_rk[b]),
                float(omega_t[b]),
                float(x_n.tmax[b] if x_n.tmax.ndim > 0 else x_n.tmax),
            )
            event_lists.append(e_new_b)

        time, mask, tau, lengths = self.build_batch_from_events(
            event_lists,
            tmax=x_n.tmax,
            device=x_n.time.device
        )

        return Batch(
                    time=time,
                    mask=mask,
                    tau=tau,
                    tmax=x_n.tmax,
                    unpadded_length=lengths,
                    kept=None,
                )

    def sample(self, n_samples: int, tmax) -> Batch:
        
        # 1. Generate the x_n event list
        x_n = generate_hpp(tmax=tmax, n_sequences=n_samples)
        B = x_n.time.shape[0]
        device = x_n.time.device
        batch_size = x_n.time.shape[0]
        n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_n.time),),
            device=device,
        )

        (dif_time_emb, time_emb, event_emb) = self.compute_emb(n=n, x_n=x_n)

        lambda_1_batch = torch.full(
            (B,),
            float(self.lambda_1),
            device=device,
        )
        for step in range(self.k_steps - 1, 0, -1):
            # 2. Get the omega value from DiT process
            omega_list = []
            for batch_index in range(batch_size):
                batch_e = event_emb[batch_index].to(torch.float32).to(device)
                dt = get_timestep_embedding(
                        torch.tensor([step], device=device),
                        self.hidden_size
                    )
                omega_k_minus_1 = self.DiT(batch_e, time_emb[batch_index].to(device), dt)
                omega_list.append(omega_k_minus_1)

             # 3. Using omega value to scale sample the x_0 value
            x_n_1 = self.backward_sample(x_n, lambda_1_batch, omega_list)
            x_n = x_n_1
            omega_batch = torch.stack(omega_list).to(lambda_1_batch.device)
            lambda_1_batch = lambda_1_batch * omega_batch

        x_0 = x_n
        return x_0


    # def sample(self, n_samples: int, tmax) -> Batch:
    #     """
    #     Sample x_0 from ADD-THIN starting from x_N.

    #     Parameters
    #     ----------
    #     n_samples : int
    #         Number of samples
    #     tmax : float
    #         T of the temporal point process
    #     begin_forecast : None, optional
    #         Beginning of the forecast, by default None
    #     end_forecast : None, optional
    #         End of the forecast, by default None

    #     Returns
    #     -------
    #     Batch
    #         Sampled x_0s
    #     """
    #     # Init x_N by sampling from HPP
    #     x_N = generate_hpp(tmax=tmax, n_sequences=n_samples)
    #     # print('x-n size: ')
    #     # print(x_N.time.size())
    #     x_n_1 = x_N

    #     # Sample x_N-1, ..., x_1 by applying posterior
    #     for n_int in range(self.steps - 1, 0, -1):
    #         n = torch.full(
    #             (n_samples,), n_int, device=tmax.device, dtype=torch.long
    #         )
    #         x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)

    #     # Sample x_0
    #     n = torch.full(
    #         (n_samples,), n_int - 1, device=tmax.device, dtype=torch.long
    #     )
    #     x_0, x_classifed, sampled_x_0, classified_not_x_0 = self.sample_x_0(n=n, x_n=x_n_1)

    #     alpha_n = self.get_n(
    #         min=0,
    #         max=self.steps,
    #         shape=(len(x_0),),
    #         device=x_0.time.device,
    #     )
    #     x_0_kept, x_0_thinned = x_0.thin(alpha=self.alpha_cumprod[alpha_n])
    #     x_n = x_0_kept.add_events(x_N)
    #     self.temp_x_N = x_n

    #     return x_0

    # def sample_x_0(
    #     self, n: TensorType[int], x_n: Batch
    # ) -> Tuple[Batch, Batch, Batch, Batch]:
    #     """
    #     Sample x_0 from x_n by classifying the intersection of x_0 and x_n and sampling from the intensity.

    #     Parameters
    #     ----------
    #     n : TensorType[int]
    #         Diffusion time steps
    #     x_n : Batch
    #         Batch of data

    #     Returns
    #     -------
    #     Tuple[Batch, Batch, Batch, Batch]
    #         x_0, classified_x_0, sampled_x_0, classified_not_x_0
    #     """
    #     (
    #         dif_time_emb,
    #         time_emb,
    #         event_emb,
    #     ) = self.compute_emb(n=n, x_n=x_n)

    #     # Sample x_0\x_n from intensity
    #     sampled_x_0 = self.intensity_model.sample(
    #         event_emb=event_emb,
    #         dif_time_emb=dif_time_emb,
    #         n_samples=1,
    #         x_n=x_n,
    #     )

    #     # Classify (x_0 ∩ x_n) from x_n
    #     x_n_and_x_0_logits = self.classifier_model(
    #         dif_time_emb=dif_time_emb, time_emb=time_emb, event_emb=event_emb
    #     )
    #     classified_x_0, classified_not_x_0 = x_n.thin(
    #         alpha=x_n_and_x_0_logits.sigmoid()
    #     )
    #     return (
    #         classified_x_0.add_events(sampled_x_0),
    #         classified_x_0,
    #         sampled_x_0,
    #         classified_not_x_0,
    #     )

    # def sample_posterior(self, x_n: Batch, n: TensorType[int]) -> Batch:
    #     """
    #     Sample x_n-1 from x_n by predicting x_0 and then sampling from the posterior.

    #     Parameters
    #     ----------
    #     x_n : Batch
    #         Batch of data
    #     n : TensorType
    #         Diffusion time steps

    #     Returns
    #     -------
    #     Batch
    #         x_n-1
    #     """
    #     # Sample x_0 and x_n\x_0
    #     _, classified_x_0, sampled_x_0, classified_not_x_0 = self.sample_x_0(
    #         n=n, x_n=x_n
    #     )

    #     # Sample C
    #     x_0_kept, _ = sampled_x_0.thin(alpha=self.alpha_x0_kept[n - 1])

    #     # Sample D
    #     hpp = generate_hpp(
    #         tmax=x_n.tmax,
    #         n_sequences=x_n.batch_size,
    #         intensity=self.add_remove[n - 1],
    #     )

    #     # Sample E
    #     x_n_kept, _ = classified_not_x_0.thin(alpha=self.alpha_xn_kept[n - 1])

    #     # Superposition of B, C, D, E to attain x_n-1
    #     x_n_1 = (
    #         classified_x_0.add_events(hpp)
    #         .add_events(x_n_kept)
    #         .add_events(x_0_kept)
    #     )
    #     return x_n_1
    

    # ! When use this function, make sure have already use model function create a temporary x_N
    def get_x_N(self):
        """
        Used to get temporary x_N, generated from add-thin model

        Returns:
            Batch : generated x_N batch 
            None : when user did not call sample function before use this function
        """
        if not self.temp_x_N == None:
            return self.temp_x_N
        return None