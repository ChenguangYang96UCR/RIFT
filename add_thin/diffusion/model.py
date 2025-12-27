import math
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from add_thin.data import Batch
from add_thin.backbones.cnn import CNNSeqEmb
from add_thin.backbones.embeddings import NyquistFrequencyEmbedding
from add_thin.processes.hpp import generate_hpp
from add_thin.diffusion.utils import betas_for_alpha_bar
import torch.nn.functional as F
from add_thin.utils.math import get_2d_sincos_pos_embed, modulate
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
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    
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
        return x
    
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        t = t.float()
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

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
        
        print(f"patch size {self.patch_size}, num_patches: {num_patches}, hidden size: {self.hidden_size}")
        
        x = self.proj(x)  # (B, hidden_size, P, P)
        print(f"rearrange x size : {x.shape}")
        
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
        print(f"##1. Final Layer: input x shape {x.shape}, c shape {c.shape}")
        
        adaLN_out = self.adaLN_modulation(c)
        shift, scale = adaLN_out.chunk(2, dim=1)
        print(f"##2. Final Layer: input shift shape {shift.shape}, scale shape {scale.shape}")
        
        x = self.norm_final(x)
        print(f"##3. Final Layer: after layer norm x shape {x.shape}")
        
        x = modulate(x, shift, scale)
        print(f"##4. Final Layer: after modulate x shape {x.shape}")
        
        x = self.linear(x)
        print(f"##5. Final Layer: dense parameter: #1 {self.patch_size * self.patch_size * self.out_channels}, self.patch_size {self.patch_size}, self.out_channels {self.out_channels}")
        print(f"##6. Final Layer: after linear dense x shape {x.shape}")
        return x

class MLP(nn.Module):
    def __init__(self, L, out_dim, hidden=None, dropout=0.0):
        super().__init__()
        print(f"MLP L number is {L}")
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
        print(f"MLP L number is {L}")
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
class Student_DiffusionModell(nn.Module):
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
        
        # Components
        self.patch_embed = PatchEmbed(patch_size, hidden_size)
        self.time_embed = TimestepEmbedder(hidden_size)
        self.dt_embed = TimestepEmbedder(hidden_size)

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
        """
        Set the history to condition the model.

        Parameters
        ----------
        batch : Batch
            Batch of data
        """
        B, L = batch.time.shape

        # Encode event times
        time_emb = self.time_encoder(
            torch.cat(
                [batch.time.unsqueeze(-1), batch.tau.unsqueeze(-1)], dim=-1
            )
        ).reshape(B, L, -1)

        # Compute history embedding
        embedding = self.history_encoder(time_emb)[0]

        # Index relative to time and set history
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
        B, L = x_n.batch_size, x_n.seq_len
        print(f"L value: {L}")

        # embed diffusion and process time
        dif_time_emb = self.diffusion_time_encoder(n)

        # Condition ADD-THIN on history by adding it to the diffusion time embedding
        if self.forecast:
            dif_time_emb = self.history_mlp(
                torch.cat([self.history, dif_time_emb], dim=-1)
            )

        # Embed event and interevent time
        time_emb = self.time_encoder(
            torch.cat([x_n.time.unsqueeze(-1), x_n.tau.unsqueeze(-1)], dim=-1)
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

    def noise(
        self, x_0: Batch, n: TensorType[torch.long, "batch"]
    ) -> Tuple[Batch, Batch]:
        """
        Sample x_n from x_0 by applying the noising process.

        Parameters
        ----------
        x_0 : Batch
            Batch of data
        n : TensorType[torch.long, "batch"]
            Number of noise steps

        Returns
        -------
        Tuple[Batch, Batch]
            x_n and thinned x_0
        """
        # Thin x_0
        x_0_kept, x_0_thinned = x_0.thin(alpha=self.alpha_cumprod[n])

        # Superposition with HPP (add)
        hpp = generate_hpp(
            tmax=x_0.tmax,
            n_sequences=len(x_0),
            intensity=1 - self.alpha_cumprod[n],
        )
        x_n = x_0_kept.add_events(hpp)

        return x_n, x_0_thinned

    def DiT(self, x, t, dt, train=False, return_activations=False):
        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        real_patch = int(self.number_patches ** 0.5)
        print(f"num_pathcs is {self.number_patches}, real_path is {real_patch}")
        assert real_patch * real_patch == self.number_patches
        activations = {}

        batch_size = x.shape[0]
        input_size = x.shape[1]
        data_length = x.shape[1] * x.shape[2]
        
        # Ensure all tensors are on the same device
        device = x.device
        
        # Convert tensors to appropriate device and dtype if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, device=device, dtype=torch.float32)
        
        # Get positional embeddings
        pos_embed = get_2d_sincos_pos_embed(None, self.hidden_size, self.number_patches, device=device)
        pos_embed = pos_embed.to(x.dtype)
        activations['patch_embed'] = x

        # Add positional embeddings
        x = x + pos_embed
        
        # Get timestep embeddings
        te = self.time_embed(t)  # (B, hidden_size)
        dte = self.dt_embed(dt)  # (B, hidden_size)
        c = te + dte
        
        activations['pos_embed'] = pos_embed
        activations['time_embed'] = te
        activations['dt_embed'] = dte
        activations['conditioning'] = c

        print("DiT: Conditioning of shape", c.shape, "dtype", c.dtype)
        
        # Apply DiT blocks
        for i in range(2):
            x = self.blocks[i](x, c)
            activations[f'dit_block_{i}'] = x
        
        # Apply final layer
        x = self.final_layer(x, c)  # (B, num_patches, p*p*c)
        activations['final_layer'] = x
        
        # Reshape to image format
        x = x.reshape(batch_size, real_patch, real_patch, self.patch_size, self.patch_size, 1)
        x = x.permute(0, 1, 3, 2, 4, 5)  # Equivalent to einsum 'bhwpqc->bhpwqc'
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=int(real_patch), W=int(real_patch))
        
        assert x.shape == (batch_size, input_size, input_size, 1), f"x shape is {x.shape}, expected ({batch_size}, {input_size}, {input_size}, 1)"

        # Create logvar embeddings
        t_discrete = torch.floor(t * 256).long()
        logvars = self.logvar_embed(t_discrete) * 100

        # Flatten for MLP processing
        print(f"Before DiT MLP shape {x.shape}")
        x_flat = x.reshape(x.shape[0], -1)
        print(f"After DiT MLP shape {x_flat.shape}")
        
        # Apply final MLP
        mlp = DiT_MLP(x_flat.shape[1], data_length, 1).to(device)
        if train:
            mlp.train()
        else:
            mlp.eval()
            
        e_new = mlp(x_flat)
        
        if return_activations:
            return e_new, logvars, activations
        return e_new

    def scaled_sampling(self, k, lambda_1, e_0, e_bar, T, r_k, r_k1, gap = 0.1):
        bandwidth_square = 1.5
        batch_size = e_0.shape[0]
        print(f"scaled sampling batch size {batch_size}")
        T = T.detach().cpu().numpy()
        e_0 = e_0.detach().cpu().numpy()
        if isinstance(e_bar, torch.Tensor):
            e_bar = e_bar.detach().cpu().numpy()

        def lambda_0_hat(e_bar, e_0, bandwidth_square):
            print(f"before kernel values, e_bar {e_bar.reshape(-1, 1).shape}, e_0 {e_0.reshape(1, -1).shape}")
            kernel_values = np.exp(-((e_bar.reshape((-1, 1)) - e_0.reshape((1, -1))) **2 ) / (bandwidth_square))
            return np.sum(kernel_values, axis = 1) / len(e_0)

        def lambda_0_hat_kmeans(e_bar, e_0, bandwidth_square, n_clusters=1000):
            # cluster e_0
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(e_0.reshape(-1, 1))
            centroids = kmeans.cluster_centers_.reshape(-1)   # shape (n_clusters,)
            counts = np.bincount(kmeans.labels_)              # cluster weights

            # compute kernel vs centroids
            diff = e_bar.reshape(-1, 1) - centroids.reshape(1, -1)
            kernel_values = np.exp(-(diff**2) / bandwidth_square)

            # weighted average
            weighted = kernel_values @ counts
            return weighted / len(e_0)

        # calculating the ratio of omega_bar_t
        kernel_approx_e0_inputs = lambda_0_hat_kmeans(e_bar, e_0, bandwidth_square)
        omega_bar_thinning = (lambda_1 ** (r_k-r_k1))*((kernel_approx_e0_inputs) ** (r_k1-r_k))

        # reshape omega_bar from [total_length] to [batch_size, -1]
        pad_len = (batch_size - (omega_bar_thinning.shape[0] % batch_size)) % batch_size
        if pad_len > 0:
            omega_bar_thinning = np.pad(omega_bar_thinning, (0, pad_len), mode="constant", constant_values=0)
        omega_bar_thinning = omega_bar_thinning.reshape(batch_size, -1)
        keeping_judge = (np.random.rand(batch_size)[:, None] < (omega_bar_thinning))
        print(f"keeping judge shape {keeping_judge.shape}, e_bar shape {e_bar.shape}")

        target_len = keeping_judge.size
        if e_bar.size < target_len:
            # pad with zeros
            e_bar_resized = np.pad(e_bar, (0, target_len - e_bar.size), mode="constant")
        else:
            # truncate
            e_bar_resized = e_bar[:target_len]

        # Step 2: reshape to b's shape
        e_bar = e_bar_resized.reshape(keeping_judge.shape)

        e_thinning_left = e_bar[keeping_judge]

        # event adding through rejection sampling
        # roughly estimate an upper bound of the function difference upper_b = max_t lambda_bar*omega_bar - lambda_bar
        upper_b = 2 * np.max((lambda_1 ** (k * gap)) * (omega_bar_thinning ** (1 - (k * gap)))) + 2

        # uniformly generate new candidate events points
        e_candidate_adding = np.random.rand(np.random.poisson(upper_b * T)) * T

        kernel_approx_e_candidate_adding_inputs = lambda_0_hat_kmeans(e_candidate_adding, e_0, bandwidth_square)
        omega_bar_adding = (lambda_1**(r_k-r_k1))*((kernel_approx_e_candidate_adding_inputs)**(r_k1-r_k))
        lambda_bar_adding = (lambda_1**(r_k1))*((kernel_approx_e_candidate_adding_inputs)**(1-r_k1))

        keeping_judge = (np.random.rand(len(e_candidate_adding)) < ((omega_bar_adding-1)*lambda_bar_adding))
        e_adding = e_candidate_adding[keeping_judge]
        e_new = np.concatenate((e_thinning_left, e_adding))
        return e_new

    def approximate_lambda_0_hat(self, k_steps, lambda_1, e_rk_array):
        """
        based on k_steps' sample value to approximate the lambda_0_hat

        Args:
            steps (int): step number which used to add or thin event
            lambda_1 (float): Given parameter 
            e_rk_array (array): DiT result
        """        

        # factor 1/K^2
        sum_term = e_rk_array.mean() * (k_steps * k_steps) 

        # exponentiate
        result = lambda_1 * torch.exp(sum_term)
        return result


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
        n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_0),),
            device=x_0.time.device,
        )
        
        # Initialize the sample parameter
        r_k = 0.1
        r_k1 = r_k - (1/self.k_steps)
        e_rk1 = x_0.time
        e_new = []
        e_0 = x_0.time
        batch_size = e_0.shape[0]
        # noise the event
        for i in range(1, self.k_steps): 
            e_rk = self.scaled_sampling(i, self.lambda_1, e_0, e_rk1, x_0.tmax, r_k, r_k1)
            e_rk1 = e_rk
            e_new.append(e_rk)
            print(f"e_rk shape is {e_rk.shape}")
        print(f"e_new data length: {len(e_new)}")

        e_rk_array = []
        for e in e_new:
            e = torch.from_numpy(e).to(torch.float32).to(x_0.time.device)
            
            # reshape e from [data_len, ] to [batch, data]
            pad_len = (batch_size - (e.numel() % batch_size)) % batch_size
            print(f"MLP forward L % self.number_patches: {e.numel() % batch_size}, (self.number_patches - (L % self.number_patches)): {(self.number_patches - (e.numel() % self.number_patches))} pad_len: {pad_len}")
            if pad_len > 0:
                e = F.pad(e, (0, pad_len), mode="constant", value=0)
            e = e.view(batch_size, -1)
            B, L = e.shape
            pad_len = (self.number_patches - (L % self.number_patches)) % self.number_patches
            seq_len = L
            if pad_len > 0:
                seq_len = seq_len + pad_len
            
            model = MLP(int(seq_len // self.number_patches), self.hidden_size, hidden=self.hidden_size).to(e.device)
            y = model(e, number_patches=self.number_patches)

            # Generate random timesteps using PyTorch instead of JAX
            t = torch.rand(y.shape[0], device=y.device)
            dt = torch.rand(y.shape[0], device=y.device)
            
            result = self.DiT(y, t, dt)
            print(f" DiT result shape: ", result.shape) 
            e_rk_array.append(result)

        # calculate approximate lambda_0 hat
        print(f"forward e_rk array length {len(e_rk_array)}, stack array length {torch.stack(e_rk_array).shape}")
        lambda_0_hat = self.approximate_lambda_0_hat(self.k_steps, self.lambda_1, torch.stack(e_rk_array))
        print(f"lambda_0_hat value is {lambda_0_hat}")
        return torch.stack(e_rk_array), self.lambda_1, self.k_steps

    def sample(self, n_samples: int, tmax) -> Batch:
        """
        Sample x_0 from ADD-THIN starting from x_N.

        Parameters
        ----------
        n_samples : int
            Number of samples
        tmax : float
            T of the temporal point process
        begin_forecast : None, optional
            Beginning of the forecast, by default None
        end_forecast : None, optional
            End of the forecast, by default None

        Returns
        -------
        Batch
            Sampled x_0s
        """
        # Init x_N by sampling from HPP
        x_N = generate_hpp(tmax=tmax, n_sequences=n_samples)
        # print('x-n size: ')
        # print(x_N.time.size())
        x_n_1 = x_N

        # Sample x_N-1, ..., x_1 by applying posterior
        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full(
                (n_samples,), n_int, device=tmax.device, dtype=torch.long
            )
            x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)

        # Sample x_0
        n = torch.full(
            (n_samples,), n_int - 1, device=tmax.device, dtype=torch.long
        )
        x_0, x_classifed, sampled_x_0, classified_not_x_0 = self.sample_x_0(n=n, x_n=x_n_1)

        alpha_n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_0),),
            device=x_0.time.device,
        )
        x_0_kept, x_0_thinned = x_0.thin(alpha=self.alpha_cumprod[alpha_n])
        x_n = x_0_kept.add_events(x_N)
        self.temp_x_N = x_n

        return x_0

    def sample_x_0(
        self, n: TensorType[int], x_n: Batch
    ) -> Tuple[Batch, Batch, Batch, Batch]:
        """
        Sample x_0 from x_n by classifying the intersection of x_0 and x_n and sampling from the intensity.

        Parameters
        ----------
        n : TensorType[int]
            Diffusion time steps
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[Batch, Batch, Batch, Batch]
            x_0, classified_x_0, sampled_x_0, classified_not_x_0
        """
        (
            dif_time_emb,
            time_emb,
            event_emb,
        ) = self.compute_emb(n=n, x_n=x_n)

        # Sample x_0\x_n from intensity
        sampled_x_0 = self.intensity_model.sample(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            n_samples=1,
            x_n=x_n,
        )

        # Classify (x_0 ∩ x_n) from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb, time_emb=time_emb, event_emb=event_emb
        )
        classified_x_0, classified_not_x_0 = x_n.thin(
            alpha=x_n_and_x_0_logits.sigmoid()
        )
        return (
            classified_x_0.add_events(sampled_x_0),
            classified_x_0,
            sampled_x_0,
            classified_not_x_0,
        )

    def sample_posterior(self, x_n: Batch, n: TensorType[int]) -> Batch:
        """
        Sample x_n-1 from x_n by predicting x_0 and then sampling from the posterior.

        Parameters
        ----------
        x_n : Batch
            Batch of data
        n : TensorType
            Diffusion time steps

        Returns
        -------
        Batch
            x_n-1
        """
        # Sample x_0 and x_n\x_0
        _, classified_x_0, sampled_x_0, classified_not_x_0 = self.sample_x_0(
            n=n, x_n=x_n
        )

        # Sample C
        x_0_kept, _ = sampled_x_0.thin(alpha=self.alpha_x0_kept[n - 1])

        # Sample D
        hpp = generate_hpp(
            tmax=x_n.tmax,
            n_sequences=x_n.batch_size,
            intensity=self.add_remove[n - 1],
        )

        # Sample E
        x_n_kept, _ = classified_not_x_0.thin(alpha=self.alpha_xn_kept[n - 1])

        # Superposition of B, C, D, E to attain x_n-1
        x_n_1 = (
            classified_x_0.add_events(hpp)
            .add_events(x_n_kept)
            .add_events(x_0_kept)
        )
        return x_n_1
    

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


@typechecked
class Student_AddThin(Student_DiffusionModell):
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
        steps: int  = 2,
        hidden_dims: int = 128,
        emb_dim: int = 32,
        encoder_layer: int = 4,
        kernel_size: int = 16,
        forecast=None,
    ) -> None:
        super().__init__(steps)
        # Set models parametrizing the approximate posterior
        self.classifier_model = classifier_model
        self.intensity_model = intensity_model
        self.temp_x_N = None
        self.n_max = n_max

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
        """
        Set the history to condition the model.

        Parameters
        ----------
        batch : Batch
            Batch of data
        """
        B, L = batch.time.shape

        # Encode event times
        time_emb = self.time_encoder(
            torch.cat(
                [batch.time.unsqueeze(-1), batch.tau.unsqueeze(-1)], dim=-1
            )
        ).reshape(B, L, -1)

        # Compute history embedding
        embedding = self.history_encoder(time_emb)[0]

        # Index relative to time and set history
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
            torch.cat([x_n.time.unsqueeze(-1), x_n.tau.unsqueeze(-1)], dim=-1)
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

    def noise(
        self, x_0: Batch, n: TensorType[torch.long, "batch"]
    ) -> Tuple[Batch, Batch]:
        """
        Sample x_n from x_0 by applying the noising process.

        Parameters
        ----------
        x_0 : Batch
            Batch of data
        n : TensorType[torch.long, "batch"]
            Number of noise steps

        Returns
        -------
        Tuple[Batch, Batch]
            x_n and thinned x_0
        """
        # Thin x_0
        x_0_kept, x_0_thinned = x_0.thin(alpha=self.alpha_cumprod[n])

        # Superposition with HPP (add)
        hpp = generate_hpp(
            tmax=x_0.tmax,
            n_sequences=len(x_0),
            intensity=1 - self.alpha_cumprod[n],
        )
        x_n = x_0_kept.add_events(hpp)

        return x_n, x_0_thinned

    def forward(
        self, x: list[Batch]
    ) -> Tuple[
        TensorType[float, "batch", "sequence_x_n"],
        TensorType[float, "batch"],
        Batch,
    ]:
        """
        Forward pass to train the model, i.e., predict x_0 from x_n.

        Parameters
        ----------
        x_0 : Batch
            Batch of data, combine x_0 and x_N [x0, xN]

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

        x_0 = x[0]
        x_n = x[1]
        
        n = self.get_n(
            min=0,
            max=100,
            shape=(len(x_0),),
            device=x_0.time.device,
        )

        # Noise x_0 to get x_n
        # if n == 2: 
        # _, x_0_thin = self.noise(x_0=x_0, n=n)
        # else:
        x_0_thin = x_0

        # Embed x_n
        (dif_time_emb, time_emb, event_emb) = self.compute_emb(n=n, x_n=x_n)

        # Predict x_0 from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb,
            time_emb=time_emb,
            event_emb=event_emb,
        )

        # Evaluate intensity of thinned x_0
        log_like_x_0 = self.intensity_model.log_likelihood(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_0=x_0_thin,
            x_n=x_n,
        )

        return x_n_and_x_0_logits, log_like_x_0, x_n

    def sample(self, n_samples: int, tmax) -> Batch:
        """
        Sample x_0 from ADD-THIN starting from x_N.

        Parameters
        ----------
        n_samples : int
            Number of samples
        tmax : float
            T of the temporal point process
        begin_forecast : None, optional
            Beginning of the forecast, by default None
        end_forecast : None, optional
            End of the forecast, by default None

        Returns
        -------
        Batch
            Sampled x_0s
        """
        # Init x_N by sampling from HPP
        x_N = generate_hpp(tmax=tmax, n_sequences=n_samples)
        self.temp_x_N = x_N
        x_n_1 = x_N

        # Sample x_N-1, ..., x_1 by applying posterior
        for n_int in range(self.steps, 0, -1):
            n = torch.full(
                (n_samples,), n_int, device=tmax.device, dtype=torch.long
            )
            x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)

        # Sample x_0
        n = torch.full(
            (n_samples,), n_int - 1, device=tmax.device, dtype=torch.long
        )
        x_0, x_classifed, sampled_x_0, classified_not_x_0 = self.sample_x_0(n=n, x_n=x_n_1)
        return x_0

    def sample_x_0(
        self, n: TensorType[int], x_n: Batch
    ) -> Tuple[Batch, Batch, Batch, Batch]:
        """
        Sample x_0 from x_n by classifying the intersection of x_0 and x_n and sampling from the intensity.

        Parameters
        ----------
        n : TensorType[int]
            Diffusion time steps
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[Batch, Batch, Batch, Batch]
            x_0, classified_x_0, sampled_x_0, classified_not_x_0
        """
        (
            dif_time_emb,
            time_emb,
            event_emb,
        ) = self.compute_emb(n=n, x_n=x_n)

        # Sample x_0\x_n from intensity
        sampled_x_0 = self.intensity_model.sample(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            n_samples=1,
            x_n=x_n,
        )

        # Classify (x_0 ∩ x_n) from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb, time_emb=time_emb, event_emb=event_emb
        )
        classified_x_0, classified_not_x_0 = x_n.thin(
            alpha=x_n_and_x_0_logits.sigmoid()
        )
        return (
            classified_x_0.add_events(sampled_x_0),
            classified_x_0,
            sampled_x_0,
            classified_not_x_0,
        )

    def sample_posterior(self, x_n: Batch, n: TensorType[int]) -> Batch:
        """
        Sample x_n-1 from x_n by predicting x_0 and then sampling from the posterior.

        Parameters
        ----------
        x_n : Batch
            Batch of data
        n : TensorType
            Diffusion time steps

        Returns
        -------
        Batch
            x_n-1
        """
        # Sample x_0 and x_n\x_0
        _, classified_x_0, sampled_x_0, classified_not_x_0 = self.sample_x_0(
            n=n, x_n=x_n
        )

        # Sample C
        x_0_kept, _ = sampled_x_0.thin(alpha=self.alpha_x0_kept[0])

        # Sample D
        # print(f'n shape: {n.shape}')
        # print(f'add_remove {self.add_remove}, shape: {self.add_remove.shape}')
        #! It has been changed, due to add_remove part is not correct.
        hpp = generate_hpp(
            tmax=x_n.tmax,
            n_sequences=x_n.batch_size
        )

        # Sample E
        x_n_kept, _ = classified_not_x_0.thin(alpha=self.alpha_xn_kept[0])

        # Superposition of B, C, D, E to attain x_n-1
        x_n_1 = (
            classified_x_0.add_events(hpp)
            .add_events(x_n_kept)
            .add_events(x_0_kept)
        )
        return x_n_1
