import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from collections import OrderedDict

from .common import (
    FourierFeatureMapping, ResidualBlock,
    get_activation, get_initializer, count_parameters,
    model_summary, format_model_summary,
    compute_gradient_norm, compute_weight_norm, check_model_health,
)


class FourierOperatorLayer1D(nn.Module):
    def __init__(self, hidden_dim=64, num_modes=12,
                 use_residual=True, use_layer_norm=False,
                 activation_name='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.use_residual = use_residual

        self.weight_local = nn.Linear(hidden_dim, hidden_dim)

        self.weight_spec_real = nn.Parameter(
            torch.randn(num_modes, hidden_dim, hidden_dim) * 0.01
        )
        self.weight_spec_imag = nn.Parameter(
            torch.randn(num_modes, hidden_dim, hidden_dim) * 0.01
        )

        self.bias_spec_real = nn.Parameter(
            torch.zeros(num_modes, hidden_dim)
        )
        self.bias_spec_imag = nn.Parameter(
            torch.zeros(num_modes, hidden_dim)
        )

        self.activation = get_activation(activation_name)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def forward(self, x):
        local_out = self.weight_local(x)

        x_fft = torch.fft.rfft(x, dim=1)
        spec_out = torch.zeros_like(x_fft)

        k = min(self.num_modes, x_fft.shape[1])
        for i in range(k):
            w_real = self.weight_spec_real[i]
            w_imag = self.weight_spec_imag[i]
            b_real = self.bias_spec_real[i]
            b_imag = self.bias_spec_imag[i]

            coeff = x_fft[:, i, :]
            real_part = coeff.real @ w_real - coeff.imag @ w_imag + b_real
            imag_part = coeff.real @ w_imag + coeff.imag @ w_real + b_imag

            spec_out[:, i, :] = torch.complex(real_part, imag_part)

        spectral_out = torch.fft.irfft(spec_out, n=x.shape[1], dim=1)

        out = local_out + spectral_out

        if self.layer_norm is not None:
            out = self.layer_norm(out)

        out = self.activation(out)

        if self.use_residual:
            out = out + x

        return out

    def get_spectral_norm(self):
        with torch.no_grad():
            norms = []
            for i in range(self.num_modes):
                w = torch.complex(self.weight_spec_real[i], self.weight_spec_imag[i])
                norms.append(w.norm().item())
            return norms


class FourierOperatorLayer2D(nn.Module):
    def __init__(self, hidden_dim=64, num_modes_height=8,
                 num_modes_width=8, use_residual=True,
                 use_layer_norm=False, activation_name='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes_height = num_modes_height
        self.num_modes_width = num_modes_width
        self.use_residual = use_residual

        self.weight_local = nn.Linear(hidden_dim, hidden_dim)

        self.weight_spec_real = nn.Parameter(
            torch.randn(num_modes_height, num_modes_width, hidden_dim, hidden_dim) * 0.01
        )
        self.weight_spec_imag = nn.Parameter(
            torch.randn(num_modes_height, num_modes_width, hidden_dim, hidden_dim) * 0.01
        )

        self.activation = get_activation(activation_name)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def forward(self, x):
        local_out = self.weight_local(x)

        if x.dim() == 3:
            x_fft = torch.fft.rfft2(x, dim=(1, 2))
            spec_out = torch.zeros_like(x_fft)

            kh = min(self.num_modes_height, x_fft.shape[1])
            kw = min(self.num_modes_width, x_fft.shape[2])

            for i in range(kh):
                for j in range(kw):
                    w_real = self.weight_spec_real[i, j]
                    w_imag = self.weight_spec_imag[i, j]
                    coeff = x_fft[:, i, j, :]
                    real_part = coeff.real @ w_real - coeff.imag @ w_imag
                    imag_part = coeff.real @ w_imag + coeff.imag @ w_real
                    spec_out[:, i, j, :] = torch.complex(real_part, imag_part)

            spectral_out = torch.fft.irfft2(spec_out, s=x.shape[1:], dim=(1, 2))
        else:
            spectral_out = torch.zeros_like(x)

        out = local_out + spectral_out

        if self.layer_norm is not None:
            out = self.layer_norm(out)

        out = self.activation(out)

        if self.use_residual:
            out = out + x

        return out


class CrossAttentionFNOBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_modes=12, num_heads=4,
                 use_residual=True, activation_name='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.num_heads = num_heads

        self.fno_layer = FourierOperatorLayer1D(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            use_residual=use_residual,
            activation_name=activation_name,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            get_activation(activation_name)(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, context=None):
        fno_out = self.fno_layer(x)

        if context is not None:
            attn_out, _ = self.cross_attn(
                self.norm1(fno_out),
                self.norm1(context),
                self.norm1(context),
            )
            fno_out = fno_out + attn_out

        ffn_out = self.ffn(self.norm2(fno_out))
        out = fno_out + ffn_out

        return out


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.q)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        h = self.W_a(x)
        h = h.view(batch_size, -1, self.num_heads, self.head_dim)
        h = h.transpose(1, 2)

        q = self.q.unsqueeze(0).unsqueeze(0)
        scores = torch.matmul(h, q.transpose(-1, -2)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)

        weights = weights.unsqueeze(-1)
        context = (h * weights).sum(dim=2)
        context = context.view(batch_size, self.hidden_dim)

        context = self.output_proj(context)
        return context


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.q.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        scores = torch.matmul(torch.tanh(self.W_a(x)), self.q)
        weights = torch.softmax(scores, dim=1)
        context = (x * weights.unsqueeze(-1)).sum(dim=1)
        return context


class HierarchicalPooling(nn.Module):
    def __init__(self, hidden_dim=64, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.pool_layers = nn.ModuleList()
        chunk_dim = max(1, hidden_dim // num_levels)
        for i in range(num_levels):
            self.pool_layers.append(nn.Sequential(
                nn.Linear(chunk_dim, chunk_dim),
                nn.Tanh(),
            ))
        self.merge = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        n = x.shape[1]
        chunk_size = max(1, n // self.num_levels)
        pooled_parts = []
        for i in range(self.num_levels):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n) if i < self.num_levels - 1 else n
            chunk = x[:, start:end, :]
            pooled = chunk.mean(dim=1)
            pooled = self.pool_layers[i](pooled)
            pooled_parts.append(pooled)
        return self.merge(torch.cat(pooled_parts, dim=-1))


class FiLMProjection(nn.Module):
    def __init__(self, input_dim=64, film_dim=128,
                 layer_dims=None, activation_name='gelu'):
        super().__init__()
        if layer_dims is None:
            layer_dims = [128, 200, 200, 200, 128]

        self.layer_dims = layer_dims
        act = get_activation(activation_name)

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, film_dim),
            act,
            nn.Linear(film_dim, film_dim),
            act,
        )
        self.gamma_heads = nn.ModuleList([
            nn.Linear(film_dim, dim) for dim in layer_dims
        ])
        self.beta_heads = nn.ModuleList([
            nn.Linear(film_dim, dim) for dim in layer_dims
        ])

    def forward(self, x):
        shared = self.shared_net(x)
        gammas = [head(shared) for head in self.gamma_heads]
        betas = [head(shared) for head in self.beta_heads]
        return gammas, betas


class FiLMProjectionV2(nn.Module):
    def __init__(self, input_dim=64, film_dim=128,
                 layer_dims=None, activation_name='gelu',
                 use_layer_norm=True, dropout=0.0):
        super().__init__()
        if layer_dims is None:
            layer_dims = [128, 200, 200, 200, 128]

        self.layer_dims = layer_dims

        layers = [
            nn.Linear(input_dim, film_dim),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(film_dim))
        layers.append(get_activation(activation_name)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.extend([
            nn.Linear(film_dim, film_dim),
        ])
        if use_layer_norm:
            layers.append(nn.LayerNorm(film_dim))
        layers.append(get_activation(activation_name)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.shared_net = nn.Sequential(*layers)

        self.gamma_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(film_dim, film_dim // 2),
                get_activation(activation_name)(),
                nn.Linear(film_dim // 2, dim),
            )
            for dim in layer_dims
        ])
        self.beta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(film_dim, film_dim // 2),
                get_activation(activation_name)(),
                nn.Linear(film_dim // 2, dim),
            )
            for dim in layer_dims
        ])

        for head in self.gamma_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

        for head in self.beta_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.shared_net(x)
        gammas = [1.0 + head(shared) for head in self.gamma_heads]
        betas = [head(shared) for head in self.beta_heads]
        return gammas, betas


class MediumProfileAugmentor(nn.Module):
    def __init__(self, hidden_dim=64, augmentation_type='noise',
                 noise_scale=0.01, mixup_alpha=0.2):
        super().__init__()
        self.augmentation_type = augmentation_type
        self.noise_scale = noise_scale
        self.mixup_alpha = mixup_alpha
        self.hidden_dim = hidden_dim

        if augmentation_type == 'learned':
            self.augment_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, medium_profile, training=True):
        if not training:
            return medium_profile

        if self.augmentation_type == 'noise':
            noise = torch.randn_like(medium_profile) * self.noise_scale
            return medium_profile + noise

        elif self.augmentation_type == 'mixup':
            if self.mixup_alpha > 0:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            else:
                lam = 1.0
            batch_size = medium_profile.shape[0]
            index = torch.randperm(batch_size, device=medium_profile.device)
            return lam * medium_profile + (1 - lam) * medium_profile[index]

        elif self.augmentation_type == 'learned':
            if medium_profile.dim() == 3:
                b, n, c = medium_profile.shape
                flat = medium_profile.reshape(b * n, c)
                if flat.shape[-1] != self.hidden_dim:
                    return medium_profile
                augmented = flat + self.augment_net(flat)
                return augmented.reshape(b, n, c)
            return medium_profile

        return medium_profile


class MediumSimilarityMetric(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, profile_a, profile_b):
        if profile_a.dim() == 3:
            a = self.proj(profile_a.mean(dim=1))
            b = self.proj(profile_b.mean(dim=1))
        else:
            a = self.proj(profile_a)
            b = self.proj(profile_b)

        cos_sim = F.cosine_similarity(a, b, dim=-1)
        return cos_sim

    def compute_pairwise_similarity(self, profiles):
        n = profiles.shape[0]
        similarities = torch.zeros(n, n, device=profiles.device)
        for i in range(n):
            for j in range(i, n):
                sim = self.forward(profiles[i:i+1], profiles[j:j+1])
                similarities[i, j] = sim
                similarities[j, i] = sim
        return similarities


class FNOEncoder(nn.Module):
    def __init__(self, nz=64, input_channels=4, hidden_dim=64,
                 num_fno_layers=4, num_modes=12, output_dim=128,
                 layer_dims=None, pooling_type='attention',
                 num_attention_heads=4, activation_name='gelu',
                 use_fno_residual=True, use_fno_layer_norm=False,
                 use_cross_attention=False, film_version=1,
                 use_augmentation=False, augmentation_type='noise'):
        super().__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.use_cross_attention = use_cross_attention

        self.lifting = nn.Linear(3, hidden_dim)
        self.freq_embed = nn.Linear(1, hidden_dim)

        if use_cross_attention:
            self.fno_layers = nn.ModuleList()
            for i in range(num_fno_layers):
                self.fno_layers.append(
                    CrossAttentionFNOBlock(
                        hidden_dim=hidden_dim,
                        num_modes=num_modes,
                        num_heads=num_attention_heads,
                        use_residual=use_fno_residual,
                        activation_name=activation_name,
                    )
                )
        else:
            self.fno_layers = nn.ModuleList([
                FourierOperatorLayer1D(
                    hidden_dim=hidden_dim,
                    num_modes=num_modes,
                    use_residual=use_fno_residual,
                    use_layer_norm=use_fno_layer_norm,
                    activation_name=activation_name,
                )
                for _ in range(num_fno_layers)
            ])

        if pooling_type == 'attention':
            self.attention_pool = AttentionPooling(hidden_dim=hidden_dim)
        elif pooling_type == 'multihead':
            self.attention_pool = MultiHeadAttentionPooling(
                hidden_dim=hidden_dim, num_heads=num_attention_heads
            )
        elif pooling_type == 'hierarchical':
            self.attention_pool = HierarchicalPooling(
                hidden_dim=hidden_dim, num_levels=3
            )
        elif pooling_type == 'mean':
            self.attention_pool = None
        elif pooling_type == 'max':
            self.attention_pool = None
        else:
            self.attention_pool = AttentionPooling(hidden_dim=hidden_dim)

        if film_version == 2:
            self.film_projection = FiLMProjectionV2(
                input_dim=hidden_dim, film_dim=output_dim,
                layer_dims=layer_dims, activation_name=activation_name,
            )
        else:
            self.film_projection = FiLMProjection(
                input_dim=hidden_dim, film_dim=output_dim,
                layer_dims=layer_dims, activation_name=activation_name,
            )

        if use_augmentation:
            self.augmentor = MediumProfileAugmentor(
                hidden_dim=hidden_dim,
                augmentation_type=augmentation_type,
            )
        else:
            self.augmentor = None

        self.similarity_metric = MediumSimilarityMetric(hidden_dim)

    def forward(self, medium_profile):
        medium_params = medium_profile[:, :, :3]
        freq_channel = medium_profile[:, :, 3:4]

        x = self.lifting(medium_params) + self.freq_embed(freq_channel)

        if self.augmentor is not None and self.training:
            x = self.augmentor(x, training=True)

        for fno_layer in self.fno_layers:
            if self.use_cross_attention and isinstance(fno_layer, CrossAttentionFNOBlock):
                x = fno_layer(x, context=x)
            else:
                x = fno_layer(x)

        if self.pooling_type == 'attention' or self.pooling_type == 'multihead' or self.pooling_type == 'hierarchical':
            context = self.attention_pool(x)
        elif self.pooling_type == 'mean':
            context = x.mean(dim=1)
        elif self.pooling_type == 'max':
            context = x.max(dim=1)[0]
        else:
            context = self.attention_pool(x)

        gammas, betas = self.film_projection(context)
        return gammas, betas

    def get_attention_weights(self, medium_profile):
        medium_params = medium_profile[:, :, :3]
        freq_channel = medium_profile[:, :, 3:4]
        x = self.lifting(medium_params) + self.freq_embed(freq_channel)
        for fno_layer in self.fno_layers:
            x = fno_layer(x)

        if hasattr(self.attention_pool, 'W_a'):
            scores = torch.matmul(
                torch.tanh(self.attention_pool.W_a(x)),
                self.attention_pool.q
            )
            weights = torch.softmax(scores, dim=1)
            return weights.detach()
        return None

    def get_spectral_features(self, medium_profile):
        medium_params = medium_profile[:, :, :3]
        freq_channel = medium_profile[:, :, 3:4]
        x = self.lifting(medium_params) + self.freq_embed(freq_channel)

        spectral_acts = []
        for fno_layer in self.fno_layers:
            x_fft = torch.fft.rfft(x, dim=1)
            spectral_acts.append(x_fft.abs().mean(dim=-1).detach())
            x = fno_layer(x)

        return spectral_acts

    def get_layer_spectral_norms(self):
        norms = []
        for layer in self.fno_layers:
            if isinstance(layer, FourierOperatorLayer1D):
                norms.append(layer.get_spectral_norm())
            elif isinstance(layer, CrossAttentionFNOBlock):
                norms.append(layer.fno_layer.get_spectral_norm())
        return norms

    def compute_medium_similarity(self, profile_a, profile_b):
        return self.similarity_metric(profile_a, profile_b)


class MultiScaleFNOEncoder(nn.Module):
    def __init__(self, nz=64, input_channels=4, hidden_dim=64,
                 num_fno_layers=4, num_modes_list=None,
                 output_dim=128, layer_dims=None,
                 pooling_type='attention', num_attention_heads=4,
                 activation_name='gelu', use_fno_residual=True):
        super().__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim

        if num_modes_list is None:
            num_modes_list = [4, 8, 12, 16]

        self.num_scales = len(num_modes_list)

        self.lifting = nn.Linear(3, hidden_dim)
        self.freq_embed = nn.Linear(1, hidden_dim)

        self.scale_encoders = nn.ModuleList()
        for num_modes in num_modes_list:
            encoder_layers = nn.ModuleList([
                FourierOperatorLayer1D(
                    hidden_dim=hidden_dim,
                    num_modes=num_modes,
                    use_residual=use_fno_residual,
                    activation_name=activation_name,
                )
                for _ in range(num_fno_layers)
            ])
            self.scale_encoders.append(encoder_layers)

        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim),
            get_activation(activation_name)(),
        )

        if pooling_type == 'attention':
            self.attention_pool = AttentionPooling(hidden_dim=hidden_dim)
        elif pooling_type == 'multihead':
            self.attention_pool = MultiHeadAttentionPooling(
                hidden_dim=hidden_dim, num_heads=num_attention_heads
            )
        else:
            self.attention_pool = AttentionPooling(hidden_dim=hidden_dim)

        self.film_projection = FiLMProjection(
            input_dim=hidden_dim, film_dim=output_dim,
            layer_dims=layer_dims, activation_name=activation_name,
        )

    def forward(self, medium_profile):
        medium_params = medium_profile[:, :, :3]
        freq_channel = medium_profile[:, :, 3:4]

        x_base = self.lifting(medium_params) + self.freq_embed(freq_channel)

        scale_outputs = []
        for encoder_layers in self.scale_encoders:
            x = x_base
            for layer in encoder_layers:
                x = layer(x)
            scale_outputs.append(x)

        fused = self.scale_fusion(
            torch.cat(scale_outputs, dim=-1)
        )

        context = self.attention_pool(fused)
        gammas, betas = self.film_projection(context)
        return gammas, betas


class AdaptiveFNODepthEncoder(nn.Module):
    def __init__(self, nz=64, input_channels=4, hidden_dim=64,
                 max_fno_layers=8, num_modes=12, output_dim=128,
                 layer_dims=None, pooling_type='attention',
                 num_attention_heads=4, activation_name='gelu',
                 use_fno_residual=True, depth_threshold=0.01):
        super().__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim
        self.max_fno_layers = max_fno_layers
        self.depth_threshold = depth_threshold

        self.lifting = nn.Linear(3, hidden_dim)
        self.freq_embed = nn.Linear(1, hidden_dim)

        self.fno_layers = nn.ModuleList([
            FourierOperatorLayer1D(
                hidden_dim=hidden_dim,
                num_modes=num_modes,
                use_residual=use_fno_residual,
                activation_name=activation_name,
            )
            for _ in range(max_fno_layers)
        ])

        self.register_buffer(
            'active_depth', torch.tensor(max_fno_layers, dtype=torch.long)
        )
        self.register_buffer(
            'depth_usage_count', torch.zeros(max_fno_layers, dtype=torch.long)
        )

        if pooling_type == 'attention':
            self.attention_pool = AttentionPooling(hidden_dim=hidden_dim)
        elif pooling_type == 'multihead':
            self.attention_pool = MultiHeadAttentionPooling(
                hidden_dim=hidden_dim, num_heads=num_attention_heads
            )
        else:
            self.attention_pool = AttentionPooling(hidden_dim=hidden_dim)

        self.film_projection = FiLMProjection(
            input_dim=hidden_dim, film_dim=output_dim,
            layer_dims=layer_dims, activation_name=activation_name,
        )

    def set_active_depth(self, depth):
        depth = min(depth, self.max_fno_layers)
        depth = max(depth, 1)
        self.active_depth.fill_(depth)

    def forward(self, medium_profile):
        medium_params = medium_profile[:, :, :3]
        freq_channel = medium_profile[:, :, 3:4]

        x = self.lifting(medium_params) + self.freq_embed(freq_channel)

        active = self.active_depth.item()
        for i in range(active):
            x = self.fno_layers[i](x)

        context = self.attention_pool(x)
        gammas, betas = self.film_projection(context)
        return gammas, betas

    def forward_with_early_exit(self, medium_profile, threshold=None):
        if threshold is None:
            threshold = self.depth_threshold

        medium_params = medium_profile[:, :, :3]
        freq_channel = medium_profile[:, :, 3:4]
        x = self.lifting(medium_params) + self.freq_embed(freq_channel)

        prev_context = None
        for i in range(self.max_fno_layers):
            x = self.fno_layers[i](x)
            context = self.attention_pool(x)

            if prev_context is not None:
                diff = (context - prev_context).abs().mean()
                if diff < threshold:
                    self.depth_usage_count[i] += 1
                    gammas, betas = self.film_projection(context)
                    return gammas, betas, i + 1

            prev_context = context.detach()

        self.depth_usage_count[self.max_fno_layers - 1] += 1
        gammas, betas = self.film_projection(context)
        return gammas, betas, self.max_fno_layers

    def get_depth_statistics(self):
        total = self.depth_usage_count.sum().item()
        if total == 0:
            return {'avg_depth': float(self.max_fno_layers), 'distribution': []}
        distribution = (self.depth_usage_count.float() / total).tolist()
        avg = sum((i + 1) * d for i, d in enumerate(distribution))
        return {'avg_depth': avg, 'distribution': distribution}


class SubnetFRPINNFNO(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, hidden_dims=None,
                 output_dim=14, activation=nn.Tanh, film_dim=128,
                 film_apply_layers=None, init_strategy='xavier_uniform'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 200, 200, 200, 128]

        self.hidden_dims = hidden_dims
        self.fourier_mapping = FourierFeatureMapping(input_dim, fourier_dim)
        self.act = activation()

        if film_apply_layers is None:
            film_apply_layers = list(range(len(hidden_dims)))

        self.film_apply_layers = film_apply_layers

        layers = []
        in_dim = fourier_dim * 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(h_dim, activation) for h_dim in hidden_dims
        ])

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_with_film(self, x, gammas, betas):
        h = self.fourier_mapping(x)

        gamma_idx = 0
        for i, (linear, res) in enumerate(
            zip(self.hidden_layers, self.residual_blocks)
        ):
            h = self.act(linear(h))
            if i in self.film_apply_layers and gamma_idx < len(gammas):
                h = gammas[gamma_idx] * h + betas[gamma_idx]
                gamma_idx += 1
            h = res(h)

        out = self.output_layer(h)
        return out

    def forward(self, x):
        h = self.fourier_mapping(x)
        for linear, res in zip(self.hidden_layers, self.residual_blocks):
            h = self.act(linear(h))
            h = res(h)
        out = self.output_layer(h)
        return out

    def get_film_effect_magnitude(self, x, gammas, betas):
        h_no_film = self.fourier_mapping(x)
        h_with_film = self.fourier_mapping(x)

        gamma_idx = 0
        film_effects = []
        for i, (linear, res) in enumerate(
            zip(self.hidden_layers, self.residual_blocks)
        ):
            h_no_film = self.act(linear(h_no_film))
            h_with_film = self.act(linear(h_with_film))

            if i in self.film_apply_layers and gamma_idx < len(gammas):
                h_modulated = gammas[gamma_idx] * h_with_film + betas[gamma_idx]
                effect = (h_modulated - h_with_film).detach().norm().item()
                film_effects.append(effect)
                h_with_film = h_modulated
                gamma_idx += 1

            h_no_film = res(h_no_film)
            h_with_film = res(h_with_film)

        return film_effects


class FRPINNFNO(nn.Module):
    def __init__(self, num_subnets=10, input_dim=8, fourier_dim=64,
                 hidden_dims=None, output_dim=14,
                 scale_factors=None, activation=nn.Tanh,
                 nz=64, medium_channels=4, fno_hidden_dim=64,
                 num_fno_layers=4, num_modes=12, film_dim=128,
                 pooling_type='attention', num_attention_heads=4,
                 activation_name_fno='gelu',
                 use_fno_residual=True, use_fno_layer_norm=False,
                 aggregation='weighted_mean',
                 init_strategy='xavier_uniform',
                 encoder_type='standard', use_cross_attention=False,
                 film_version=1, use_augmentation=False,
                 augmentation_type='noise', num_modes_list=None,
                 max_fno_depth=8, diversity_weight=0.0):
        super().__init__()
        if scale_factors is None:
            scale_factors = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        assert len(scale_factors) == num_subnets

        self.num_subnets = num_subnets
        self.scale_factors = scale_factors
        self.aggregation = aggregation
        self.nz = nz
        self.encoder_type = encoder_type
        self.diversity_weight = diversity_weight

        self.register_buffer(
            'scale_factors_tensor',
            torch.tensor(scale_factors, dtype=torch.float32)
        )

        if hidden_dims is None:
            hidden_dims = [128, 200, 200, 200, 128]

        layer_dims = list(hidden_dims)

        if encoder_type == 'multiscale':
            self.fno_encoder = MultiScaleFNOEncoder(
                nz=nz,
                input_channels=medium_channels,
                hidden_dim=fno_hidden_dim,
                num_fno_layers=num_fno_layers,
                num_modes_list=num_modes_list or [4, 8, 12, 16],
                output_dim=film_dim,
                layer_dims=layer_dims,
                pooling_type=pooling_type,
                num_attention_heads=num_attention_heads,
                activation_name=activation_name_fno,
                use_fno_residual=use_fno_residual,
            )
        elif encoder_type == 'adaptive_depth':
            self.fno_encoder = AdaptiveFNODepthEncoder(
                nz=nz,
                input_channels=medium_channels,
                hidden_dim=fno_hidden_dim,
                max_fno_layers=max_fno_depth,
                num_modes=num_modes,
                output_dim=film_dim,
                layer_dims=layer_dims,
                pooling_type=pooling_type,
                num_attention_heads=num_attention_heads,
                activation_name=activation_name_fno,
                use_fno_residual=use_fno_residual,
            )
        else:
            self.fno_encoder = FNOEncoder(
                nz=nz,
                input_channels=medium_channels,
                hidden_dim=fno_hidden_dim,
                num_fno_layers=num_fno_layers,
                num_modes=num_modes,
                output_dim=film_dim,
                layer_dims=layer_dims,
                pooling_type=pooling_type,
                num_attention_heads=num_attention_heads,
                activation_name=activation_name_fno,
                use_fno_residual=use_fno_residual,
                use_fno_layer_norm=use_fno_layer_norm,
                use_cross_attention=use_cross_attention,
                film_version=film_version,
                use_augmentation=use_augmentation,
                augmentation_type=augmentation_type,
            )

        self.subnets = nn.ModuleList([
            SubnetFRPINNFNO(
                input_dim=input_dim,
                fourier_dim=fourier_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                film_dim=film_dim,
            )
            for _ in range(num_subnets)
        ])

        if aggregation == 'learnable':
            self.aggregation_weights = nn.Parameter(
                torch.ones(num_subnets, dtype=torch.float32) / num_subnets
            )

    def _apply_scale(self, x, scale_factor):
        scale = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        scale[7] = scale_factor
        return x * scale.unsqueeze(0)

    def _aggregate(self, outputs):
        if self.aggregation == 'weighted_mean':
            stacked = torch.stack(outputs, dim=0)
            return stacked.sum(dim=0) / self.num_subnets
        elif self.aggregation == 'learnable':
            weights = torch.softmax(self.aggregation_weights, dim=0)
            stacked = torch.stack(outputs, dim=0)
            weights = weights.view(-1, 1, 1)
            return (stacked * weights).sum(dim=0)
        elif self.aggregation == 'variance_weighted':
            stacked = torch.stack(outputs, dim=0)
            variances = stacked.var(dim=0)
            inv_var = 1.0 / (variances + 1e-8)
            weights = inv_var / (inv_var.sum(dim=0, keepdim=True) + 1e-8)
            return (stacked * weights).sum(dim=0)
        elif self.aggregation == 'median':
            stacked = torch.stack(outputs, dim=0)
            return stacked.median(dim=0)[0]
        else:
            stacked = torch.stack(outputs, dim=0)
            return stacked.sum(dim=0) / self.num_subnets

    def forward(self, x, medium_profile):
        if medium_profile.dim() == 2:
            medium_profile = medium_profile.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )
        elif medium_profile.shape[0] == 1 and x.shape[0] > 1:
            medium_profile = medium_profile.expand(x.shape[0], -1, -1)
        elif medium_profile.shape[0] != x.shape[0]:
            n = min(medium_profile.shape[0], x.shape[0])
            medium_profile = medium_profile[:n]
            if n < x.shape[0]:
                medium_profile = medium_profile.expand(x.shape[0], -1, -1)

        gammas, betas = self.fno_encoder(medium_profile)

        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet.forward_with_film(x_scaled, gammas, betas)
            outputs.append(out_i / self.scale_factors[i])

        return self._aggregate(outputs)

    def forward_with_diversity(self, x, medium_profile):
        if medium_profile.dim() == 2:
            medium_profile = medium_profile.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )

        gammas, betas = self.fno_encoder(medium_profile)

        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet.forward_with_film(x_scaled, gammas, betas)
            outputs.append(out_i / self.scale_factors[i])

        aggregated = self._aggregate(outputs)

        diversity_loss = torch.tensor(0.0, device=x.device)
        if self.diversity_weight > 0 and len(outputs) > 1:
            stacked = torch.stack(outputs, dim=0)
            mean_out = stacked.mean(dim=0)
            variance = ((stacked - mean_out.unsqueeze(0)) ** 2).mean()
            diversity_loss = -self.diversity_weight * variance

        return aggregated, diversity_loss

    def forward_local_only(self, x):
        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet(x_scaled)
            outputs.append(out_i / self.scale_factors[i])
        return self._aggregate(outputs)

    def get_all_subnet_outputs(self, x, medium_profile):
        if medium_profile.dim() == 2:
            medium_profile = medium_profile.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )
        gammas, betas = self.fno_encoder(medium_profile)
        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet.forward_with_film(x_scaled, gammas, betas)
            outputs.append(out_i / self.scale_factors[i])
        return outputs

    def get_subnet_output(self, x, medium_profile, subnet_idx):
        if medium_profile.dim() == 2:
            medium_profile = medium_profile.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )
        gammas, betas = self.fno_encoder(medium_profile)
        x_scaled = self._apply_scale(x, self.scale_factors[subnet_idx])
        out = self.subnets[subnet_idx].forward_with_film(x_scaled, gammas, betas)
        return out / self.scale_factors[subnet_idx]

    def get_attention_weights(self, medium_profile):
        return self.fno_encoder.get_attention_weights(medium_profile)

    def get_spectral_features(self, medium_profile):
        return self.fno_encoder.get_spectral_features(medium_profile)

    def get_film_parameters(self, medium_profile):
        gammas, betas = self.fno_encoder(medium_profile)
        gamma_norms = [g.norm().item() for g in gammas]
        beta_norms = [b.norm().item() for b in betas]
        return gamma_norms, beta_norms

    def get_film_effect_analysis(self, x, medium_profile):
        if medium_profile.dim() == 2:
            medium_profile = medium_profile.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )
        gammas, betas = self.fno_encoder(medium_profile)

        analysis = {}
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            effects = subnet.get_film_effect_magnitude(x_scaled, gammas, betas)
            analysis[f'subnet_{i}'] = effects

        return analysis

    def get_layer_spectral_norms(self):
        if hasattr(self.fno_encoder, 'get_layer_spectral_norms'):
            return self.fno_encoder.get_layer_spectral_norms()
        return None

    def compute_medium_similarity(self, profile_a, profile_b):
        if hasattr(self.fno_encoder, 'compute_medium_similarity'):
            return self.fno_encoder.compute_medium_similarity(profile_a, profile_b)
        return None

    def get_model_info(self):
        info = {
            'model_type': 'FRPINNs-FNO',
            'num_subnets': self.num_subnets,
            'scale_factors': self.scale_factors,
            'aggregation': self.aggregation,
            'encoder_type': self.encoder_type,
            'total_params': count_parameters(self),
            'trainable_params': count_parameters(self, trainable_only=True),
            'fno_encoder_params': count_parameters(self.fno_encoder),
            'backbone_params': count_parameters(self) - count_parameters(self.fno_encoder),
        }
        subnet_params = []
        for i in range(self.num_subnets):
            subnet_params.append(
                sum(p.numel() for p in self.subnets[i].parameters())
            )
        info['subnet_params'] = subnet_params
        info['subnet_params_mean'] = np.mean(subnet_params)
        info['subnet_params_std'] = np.std(subnet_params)
        return info

    def print_model_info(self):
        info = self.get_model_info()
        print("=" * 60)
        print("FRPINNs-FNO Model Information")
        print("=" * 60)
        print(f"Number of subnets: {info['num_subnets']}")
        print(f"Scale factors: {info['scale_factors']}")
        print(f"Aggregation: {info['aggregation']}")
        print(f"Encoder type: {info['encoder_type']}")
        print(f"Total parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"FNO encoder parameters: {info['fno_encoder_params']:,}")
        print(f"Backbone parameters: {info['backbone_params']:,}")
        print(f"Parameters per subnet: {info['subnet_params']}")
        print(f"Mean params per subnet: {info['subnet_params_mean']:.0f}")
        print(f"Std params per subnet: {info['subnet_params_std']:.0f}")
        print("=" * 60)

    def freeze_fno_encoder(self):
        for param in self.fno_encoder.parameters():
            param.requires_grad = False

    def unfreeze_fno_encoder(self):
        for param in self.fno_encoder.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for subnet in self.subnets:
            for param in subnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for subnet in self.subnets:
            for param in subnet.parameters():
                param.requires_grad = True

    def freeze_subnet(self, subnet_idx):
        for param in self.subnets[subnet_idx].parameters():
            param.requires_grad = False

    def unfreeze_subnet(self, subnet_idx):
        for param in self.subnets[subnet_idx].parameters():
            param.requires_grad = True

    def get_fno_encoder_parameters(self):
        return list(self.fno_encoder.parameters())

    def get_backbone_parameters(self):
        params = []
        for subnet in self.subnets:
            params.extend(list(subnet.parameters()))
        return params

    def get_parameter_groups(self, fno_lr_scale=0.1, base_lr=1e-3):
        return [
            {'params': self.get_fno_encoder_parameters(),
             'lr': base_lr * fno_lr_scale, 'name': 'fno_encoder'},
            {'params': self.get_backbone_parameters(),
             'lr': base_lr, 'name': 'backbone'},
        ]

    def get_layerwise_lr_groups(self, base_lr=1e-3, fno_lr_scale=0.1,
                                 subnet_lr_decay=0.95):
        groups = [
            {'params': self.get_fno_encoder_parameters(),
             'lr': base_lr * fno_lr_scale, 'name': 'fno_encoder'},
        ]
        for i, subnet in enumerate(self.subnets):
            subnet_lr = base_lr * (subnet_lr_decay ** i)
            groups.append({
                'params': list(subnet.parameters()),
                'lr': subnet_lr,
                'name': f'subnet_{i}',
            })
        return groups

    def transfer_from_frpinn(self, frpinn_model):
        with torch.no_grad():
            for i in range(min(self.num_subnets, frpinn_model.num_subnets)):
                src_subnet = frpinn_model.subnets[i]
                tgt_subnet = self.subnets[i]

                src_state = src_subnet.state_dict()
                tgt_state = tgt_subnet.state_dict()

                for key in tgt_state:
                    if key in src_state and src_state[key].shape == tgt_state[key].shape:
                        tgt_state[key] = src_state[key].clone()

                tgt_subnet.load_state_dict(tgt_state, strict=False)

        print(f"Transferred weights from FRPINN to {min(self.num_subnets, frpinn_model.num_subnets)} subnets")


class FRPINNFNOCheckpointManager:
    def __init__(self, model, save_dir='checkpoints', max_checkpoints=5):
        self.model = model
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_list = []

    def save(self, epoch, loss, extra_state=None):
        import os
        os.makedirs(self.save_dir, exist_ok=True)

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }
        if extra_state is not None:
            state.update(extra_state)

        path = os.path.join(self.save_dir, f'frpinn_fno_epoch_{epoch:06d}.pt')
        torch.save(state, path)
        self.checkpoint_list.append(path)

        if len(self.checkpoint_list) > self.max_checkpoints:
            old_path = self.checkpoint_list.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        return path

    def load(self, path, load_optimizer=False, optimizer=None):
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model_state_dict'])
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        return state.get('epoch', 0), state.get('loss', float('inf'))

    def load_best(self):
        if not self.checkpoint_list:
            return None
        best_path = self.checkpoint_list[-1]
        return self.load(best_path)


def create_frpinn_fno_model(config=None):
    if config is None:
        config = {}

    return FRPINNFNO(
        num_subnets=config.get('num_subnets', 10),
        input_dim=config.get('input_dim', 8),
        fourier_dim=config.get('fourier_dim', 64),
        hidden_dims=config.get('hidden_dims', None),
        output_dim=config.get('output_dim', 14),
        scale_factors=config.get('scale_factors', None),
        activation=config.get('activation', nn.Tanh),
        nz=config.get('nz', 64),
        medium_channels=config.get('medium_channels', 4),
        fno_hidden_dim=config.get('fno_hidden_dim', 64),
        num_fno_layers=config.get('num_fno_layers', 4),
        num_modes=config.get('num_modes', 12),
        film_dim=config.get('film_dim', 128),
        pooling_type=config.get('pooling_type', 'attention'),
        num_attention_heads=config.get('num_attention_heads', 4),
        activation_name_fno=config.get('activation_name_fno', 'gelu'),
        use_fno_residual=config.get('use_fno_residual', True),
        use_fno_layer_norm=config.get('use_fno_layer_norm', False),
        aggregation=config.get('aggregation', 'weighted_mean'),
        init_strategy=config.get('init_strategy', 'xavier_uniform'),
        encoder_type=config.get('encoder_type', 'standard'),
        use_cross_attention=config.get('use_cross_attention', False),
        film_version=config.get('film_version', 1),
        use_augmentation=config.get('use_augmentation', False),
        augmentation_type=config.get('augmentation_type', 'noise'),
        diversity_weight=config.get('diversity_weight', 0.0),
    )


def analyze_fno_encoder(encoder, medium_profile):
    analysis = {}

    with torch.no_grad():
        gammas, betas = encoder(medium_profile)

        analysis['num_film_layers'] = len(gammas)
        analysis['gamma_norms'] = [g.norm().item() for g in gammas]
        analysis['beta_norms'] = [b.norm().item() for b in betas]
        analysis['gamma_mean'] = float(np.mean(analysis['gamma_norms']))
        analysis['beta_mean'] = float(np.mean(analysis['beta_norms']))

        if hasattr(encoder, 'get_attention_weights'):
            attn = encoder.get_attention_weights(medium_profile)
            if attn is not None:
                analysis['attention_entropy'] = -(
                    attn * torch.log(attn + 1e-8)
                ).sum(dim=1).mean().item()

        if hasattr(encoder, 'get_spectral_features'):
            spectral = encoder.get_spectral_features(medium_profile)
            analysis['num_spectral_layers'] = len(spectral)
            if spectral:
                analysis['spectral_energy_per_layer'] = [
                    s.mean().item() for s in spectral
                ]

        if hasattr(encoder, 'get_layer_spectral_norms'):
            norms = encoder.get_layer_spectral_norms()
            if norms:
                analysis['spectral_norms_per_layer'] = [
                    float(np.mean(n)) for n in norms
                ]

    return analysis
