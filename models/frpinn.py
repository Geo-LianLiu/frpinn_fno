import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from collections import OrderedDict

from .common import (
    FourierFeatureMapping, MultiBandFourierMapping,
    RandomFourierFeatureMapping, PositionalEncodingFourier,
    ResidualBlock, PreActivationResidualBlock,
    get_activation, get_initializer, apply_weight_init,
    XavierUniformInit, count_parameters, model_summary,
    format_model_summary, GradientCheckpointWrapper,
    compute_gradient_norm, compute_weight_norm, check_model_health,
)


class SubnetworkFRPINN(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, hidden_dims=None,
                 output_dim=14, activation=nn.Tanh,
                 fourier_type='basic', fourier_sigma=None,
                 use_layer_norm=False, dropout=0.0,
                 residual_type='post', init_strategy='xavier_uniform'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 200, 200, 200, 128]

        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.residual_type = residual_type

        if fourier_type == 'basic':
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2
        elif fourier_type == 'multiband':
            self.fourier_mapping = MultiBandFourierMapping(
                input_dim, feature_dim_per_band=fourier_dim // 5
            )
            fourier_out_dim = self.fourier_mapping.output_dim()
        elif fourier_type == 'random':
            self.fourier_mapping = RandomFourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma or 1.0
            )
            fourier_out_dim = fourier_dim * 2
        elif fourier_type == 'positional':
            self.fourier_mapping = PositionalEncodingFourier(
                input_dim, fourier_dim
            )
            fourier_out_dim = fourier_dim
        else:
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2

        self.act = activation()

        self.hidden_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        in_dim = fourier_out_dim
        for i, h_dim in enumerate(hidden_dims):
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            if residual_type == 'post':
                self.residual_blocks.append(
                    ResidualBlock(h_dim, activation, use_layer_norm, dropout)
                        if in_dim == h_dim else None
                )
            elif residual_type == 'pre':
                self.residual_blocks.append(
                    PreActivationResidualBlock(h_dim, activation, use_layer_norm)
                        if in_dim == h_dim else None
                )
            else:
                self.residual_blocks.append(None)

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(h_dim))
            else:
                self.layer_norms.append(None)

            in_dim = h_dim

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        if init_strategy == 'xavier_uniform':
            self._init_xavier()
        elif isinstance(init_strategy, str):
            init_obj = get_initializer(init_strategy)
            self.apply(init_obj)
        else:
            self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fourier_mapping(x)
        for i, linear in enumerate(self.hidden_layers):
            h_prev = h
            h = linear(h)
            if self.layer_norms[i] is not None:
                h = self.layer_norms[i](h)
            h = self.act(h)
            if self.residual_blocks[i] is not None:
                h = self.residual_blocks[i](h)
            elif h_prev.shape == h.shape and i > 0:
                pass
        out = self.output_layer(h)
        return out

    def get_hidden_representations(self, x):
        h = self.fourier_mapping(x)
        representations = [h.detach()]
        for i, linear in enumerate(self.hidden_layers):
            h = linear(h)
            if self.layer_norms[i] is not None:
                h = self.layer_norms[i](h)
            h = self.act(h)
            if self.residual_blocks[i] is not None:
                h = self.residual_blocks[i](h)
            representations.append(h.detach())
        return representations


class SubnetworkFRPINNPaper(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, output_dim=14,
                 activation=nn.Tanh, init_strategy='xavier_uniform'):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.output_dim = output_dim

        self.fourier_mapping = FourierFeatureMapping(input_dim, fourier_dim)
        fourier_out_dim = fourier_dim * 2

        self.act = activation()

        self.layer0 = nn.Linear(fourier_out_dim, 128)
        self.layer1 = nn.Linear(128, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 200)
        self.layer4 = nn.Linear(200, 200)
        self.layer5 = nn.Linear(200, 128)
        self.layer6 = nn.Linear(128, 128)
        self.layer7 = nn.Linear(128, output_dim)

        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h0 = self.act(self.layer0(self.fourier_mapping(x)))
        h1 = self.act(self.layer1(h0))
        h2 = h1 + self.act(self.layer2(h1))
        h3 = h2 + self.act(self.layer3(h2))
        h4 = h3 + self.act(self.layer4(h3))
        h5 = self.act(self.layer5(h4))
        h6 = h5 + self.act(self.layer6(h5))
        out = self.layer7(h6)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        if x.dim() == 2:
            x = x.unsqueeze(1)

        Q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_proj(attn_output)

        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output


class AttentionSubnetFRPINN(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, hidden_dims=None,
                 output_dim=14, activation=nn.Tanh,
                 fourier_type='basic', fourier_sigma=None,
                 num_attention_heads=4, attention_layers=None,
                 use_layer_norm=False, dropout=0.0,
                 init_strategy='xavier_uniform'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 200, 200, 200, 128]

        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        if attention_layers is None:
            attention_layers = [len(hidden_dims) // 2]

        self.attention_layers = attention_layers

        if fourier_type == 'basic':
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2
        elif fourier_type == 'multiband':
            self.fourier_mapping = MultiBandFourierMapping(
                input_dim, feature_dim_per_band=fourier_dim // 5
            )
            fourier_out_dim = self.fourier_mapping.output_dim()
        elif fourier_type == 'random':
            self.fourier_mapping = RandomFourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma or 1.0
            )
            fourier_out_dim = fourier_dim * 2
        else:
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2

        self.act = activation()

        self.hidden_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        in_dim = fourier_out_dim
        for i, h_dim in enumerate(hidden_dims):
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            self.residual_blocks.append(
                ResidualBlock(h_dim, activation, use_layer_norm, dropout)
                if in_dim == h_dim else None
            )

            if i in attention_layers:
                self.attention_modules.append(
                    SelfAttentionLayer(h_dim, num_attention_heads, dropout)
                )
            else:
                self.attention_modules.append(None)

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(h_dim))
            else:
                self.layer_norms.append(None)

            in_dim = h_dim

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fourier_mapping(x)
        for i, linear in enumerate(self.hidden_layers):
            h_prev = h
            h = linear(h)
            if self.layer_norms[i] is not None:
                h = self.layer_norms[i](h)
            h = self.act(h)
            if self.residual_blocks[i] is not None:
                h = self.residual_blocks[i](h)
            if self.attention_modules[i] is not None:
                h = h + self.attention_modules[i](h)
        out = self.output_layer(h)
        return out


class MultiScaleResidualSubnet(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, hidden_dims=None,
                 output_dim=14, activation=nn.Tanh,
                 fourier_type='basic', fourier_sigma=None,
                 skip_connections='dense', use_layer_norm=False,
                 dropout=0.0, init_strategy='xavier_uniform'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 200, 200, 200, 128]

        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.skip_connections = skip_connections

        if fourier_type == 'basic':
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2
        else:
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2

        self.act = activation()

        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        in_dim = fourier_out_dim
        for i, h_dim in enumerate(hidden_dims):
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(h_dim))
            else:
                self.layer_norms.append(None)
            in_dim = h_dim

        if skip_connections == 'dense':
            total_dim = sum(hidden_dims)
            self.dense_merge = nn.Linear(total_dim, hidden_dims[-1])
        elif skip_connections == 'unet':
            n = len(hidden_dims)
            self.unet_skips = nn.ModuleList()
            for i in range(n // 2):
                src_dim = hidden_dims[i]
                tgt_dim = hidden_dims[n - 1 - i]
                if src_dim != tgt_dim:
                    self.unet_skips.append(nn.Linear(src_dim, tgt_dim))
                else:
                    self.unet_skips.append(None)
        elif skip_connections == 'residual':
            self.skip_projections = nn.ModuleList()
            for i in range(1, len(hidden_dims)):
                if hidden_dims[i] != hidden_dims[i - 1]:
                    self.skip_projections.append(
                        nn.Linear(hidden_dims[i - 1], hidden_dims[i])
                    )
                else:
                    self.skip_projections.append(None)

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fourier_mapping(x)

        if self.skip_connections == 'dense':
            all_features = []
            for i, linear in enumerate(self.hidden_layers):
                h = linear(h)
                if self.layer_norms[i] is not None:
                    h = self.layer_norms[i](h)
                h = self.act(h)
                all_features.append(h)
            h = self.dense_merge(torch.cat(all_features, dim=-1))

        elif self.skip_connections == 'unet':
            intermediates = []
            n = len(self.hidden_layers)
            for i in range(n):
                h = self.hidden_layers[i](h)
                if self.layer_norms[i] is not None:
                    h = self.layer_norms[i](h)
                h = self.act(h)
                intermediates.append(h)

            h = intermediates[-1]
            for i in range(n // 2):
                skip_idx = n - 2 - i
                skip_feat = intermediates[skip_idx]
                if i < len(self.unet_skips) and self.unet_skips[i] is not None:
                    skip_feat = self.unet_skips[i](skip_feat)
                h = h + skip_feat

        elif self.skip_connections == 'residual':
            for i, linear in enumerate(self.hidden_layers):
                h_prev = h
                h = linear(h)
                if self.layer_norms[i] is not None:
                    h = self.layer_norms[i](h)
                h = self.act(h)
                if i > 0 and i - 1 < len(self.skip_projections):
                    proj = self.skip_projections[i - 1]
                    if proj is not None:
                        h = h + proj(h_prev)
                    else:
                        h = h + h_prev
        else:
            for i, linear in enumerate(self.hidden_layers):
                h = linear(h)
                if self.layer_norms[i] is not None:
                    h = self.layer_norms[i](h)
                h = self.act(h)

        out = self.output_layer(h)
        return out


class AdaptiveDepthSubnet(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, hidden_dim=200,
                 max_depth=8, output_dim=14, activation=nn.Tanh,
                 fourier_type='basic', fourier_sigma=None,
                 use_layer_norm=False, dropout=0.0,
                 depth_threshold=0.01, init_strategy='xavier_uniform'):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.output_dim = output_dim
        self.depth_threshold = depth_threshold

        if fourier_type == 'basic':
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2
        else:
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2

        self.act = activation()

        self.input_proj = nn.Linear(fourier_out_dim, hidden_dim)

        self.depth_layers = nn.ModuleList()
        self.depth_norms = nn.ModuleList()
        for i in range(max_depth):
            self.depth_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                self.depth_norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.depth_norms.append(None)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.register_buffer(
            'active_depth', torch.tensor(max_depth, dtype=torch.long)
        )
        self.register_buffer(
            'depth_usage_count', torch.zeros(max_depth, dtype=torch.long)
        )

        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_active_depth(self, depth):
        depth = min(depth, self.max_depth)
        depth = max(depth, 1)
        self.active_depth.fill_(depth)

    def forward(self, x):
        h = self.act(self.input_proj(self.fourier_mapping(x)))

        active = self.active_depth.item()
        for i in range(active):
            h_prev = h
            if self.depth_norms[i] is not None:
                h = self.depth_norms[i](h)
            h = h_prev + self.act(self.depth_layers[i](h))

        out = self.output_layer(h)
        return out

    def forward_with_early_exit(self, x, threshold=None):
        if threshold is None:
            threshold = self.depth_threshold

        h = self.act(self.input_proj(self.fourier_mapping(x)))
        outputs = []

        for i in range(self.max_depth):
            h_prev = h
            if self.depth_norms[i] is not None:
                h = self.depth_norms[i](h)
            h = h_prev + self.act(self.depth_layers[i](h))

            out_i = self.output_layer(h)
            outputs.append(out_i)

            if i > 0:
                diff = (out_i - outputs[i - 1]).abs().mean()
                if diff < threshold:
                    self.depth_usage_count[i] += 1
                    return out_i, i + 1

        self.depth_usage_count[self.max_depth - 1] += 1
        return outputs[-1], self.max_depth

    def get_depth_statistics(self):
        total = self.depth_usage_count.sum().item()
        if total == 0:
            return {'avg_depth': float(self.max_depth), 'depth_distribution': [0.0] * self.max_depth}
        distribution = (self.depth_usage_count.float() / total).tolist()
        avg = sum((i + 1) * d for i, d in enumerate(distribution))
        return {'avg_depth': avg, 'depth_distribution': distribution}


class SubnetDiversityRegularizer(nn.Module):
    def __init__(self, num_subnets, diversity_type='correlation',
                 target_diversity=0.5, temperature=1.0):
        super().__init__()
        self.num_subnets = num_subnets
        self.diversity_type = diversity_type
        self.target_diversity = target_diversity
        self.temperature = temperature

    def compute_correlation_penalty(self, outputs):
        n = len(outputs)
        if n < 2:
            return torch.tensor(0.0, device=outputs[0].device)

        total_corr = torch.tensor(0.0, device=outputs[0].device)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                o_i = outputs[i].flatten()
                o_j = outputs[j].flatten()
                o_i_centered = o_i - o_i.mean()
                o_j_centered = o_j - o_j.mean()
                norm_i = o_i_centered.norm() + 1e-8
                norm_j = o_j_centered.norm() + 1e-8
                corr = (o_i_centered * o_j_centered).sum() / (norm_i * norm_j)
                total_corr = total_corr + corr ** 2
                count += 1

        return total_corr / max(count, 1)

    def compute_orthogonality_penalty(self, outputs):
        n = len(outputs)
        if n < 2:
            return torch.tensor(0.0, device=outputs[0].device)

        stacked = torch.stack([o.flatten() for o in outputs], dim=0)
        gram = torch.mm(stacked, stacked.t())
        identity = torch.eye(n, device=gram.device)
        return ((gram - identity) ** 2).sum() / (n * n)

    def compute_variance_penalty(self, outputs):
        stacked = torch.stack(outputs, dim=0)
        mean_out = stacked.mean(dim=0)
        variance = ((stacked - mean_out.unsqueeze(0)) ** 2).mean()
        return -variance

    def forward(self, outputs):
        if self.diversity_type == 'correlation':
            penalty = self.compute_correlation_penalty(outputs)
        elif self.diversity_type == 'orthogonality':
            penalty = self.compute_orthogonality_penalty(outputs)
        elif self.diversity_type == 'variance':
            penalty = self.compute_variance_penalty(outputs)
        elif self.diversity_type == 'combined':
            corr_p = self.compute_correlation_penalty(outputs)
            orth_p = self.compute_orthogonality_penalty(outputs)
            var_p = self.compute_variance_penalty(outputs)
            penalty = corr_p + 0.5 * orth_p - 0.1 * var_p
        else:
            penalty = self.compute_correlation_penalty(outputs)

        return self.temperature * penalty


class SubnetPruningManager:
    def __init__(self, model, prune_threshold=0.01, prune_interval=1000,
                 min_active_subnets=3, prune_strategy='magnitude'):
        self.model = model
        self.prune_threshold = prune_threshold
        self.prune_interval = prune_interval
        self.min_active_subnets = min_active_subnets
        self.prune_strategy = prune_strategy
        self.subnet_losses = {}
        self.subnet_importance = {}
        self.active_subnets = list(range(model.num_subnets))
        self.pruned_subnets = []
        self.prune_history = []

    def record_subnet_loss(self, subnet_idx, loss_value):
        if subnet_idx not in self.subnet_losses:
            self.subnet_losses[subnet_idx] = []
        self.subnet_losses[subnet_idx].append(loss_value)

    def compute_importance(self, subnet_idx):
        if subnet_idx not in self.subnet_losses or len(self.subnet_losses[subnet_idx]) == 0:
            return 1.0

        recent_losses = self.subnet_losses[subnet_idx][-100:]
        avg_loss = np.mean(recent_losses)

        if self.prune_strategy == 'magnitude':
            subnet = self.model.subnets[subnet_idx]
            total_norm = 0.0
            for p in subnet.parameters():
                total_norm += p.data.norm().item() ** 2
            importance = math.sqrt(total_norm)

        elif self.prune_strategy == 'loss':
            importance = 1.0 / (avg_loss + 1e-8)

        elif self.prune_strategy == 'gradient':
            subnet = self.model.subnets[subnet_idx]
            grad_norm = 0.0
            for p in subnet.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            importance = math.sqrt(grad_norm) if grad_norm > 0 else 0.0

        else:
            importance = 1.0 / (avg_loss + 1e-8)

        self.subnet_importance[subnet_idx] = importance
        return importance

    def should_prune(self, epoch):
        if epoch < self.prune_interval:
            return False
        if epoch % self.prune_interval != 0:
            return False
        if len(self.active_subnets) <= self.min_active_subnets:
            return False
        return True

    def prune_step(self, epoch):
        if not self.should_prune(epoch):
            return []

        for idx in self.active_subnets:
            self.compute_importance(idx)

        sorted_subnets = sorted(
            self.active_subnets,
            key=lambda x: self.subnet_importance.get(x, 0.0)
        )

        pruned = []
        for idx in sorted_subnets:
            if len(self.active_subnets) - len(pruned) <= self.min_active_subnets:
                break
            if self.subnet_importance.get(idx, 1.0) < self.prune_threshold:
                self.model.freeze_subnet(idx)
                self.active_subnets.remove(idx)
                self.pruned_subnets.append(idx)
                pruned.append(idx)

        if pruned:
            self.prune_history.append({
                'epoch': epoch,
                'pruned_subnets': pruned,
                'active_subnets': list(self.active_subnets),
                'importance_scores': {k: v for k, v in self.subnet_importance.items()
                                      if k in pruned},
            })

        return pruned

    def restore_subnet(self, subnet_idx):
        if subnet_idx in self.pruned_subnets:
            self.model.unfreeze_subnet(subnet_idx)
            self.pruned_subnets.remove(subnet_idx)
            self.active_subnets.append(subnet_idx)
            return True
        return False

    def get_pruning_summary(self):
        return {
            'active_subnets': list(self.active_subnets),
            'pruned_subnets': list(self.pruned_subnets),
            'num_active': len(self.active_subnets),
            'num_pruned': len(self.pruned_subnets),
            'prune_history': self.prune_history,
        }


class CurriculumScaleScheduler:
    def __init__(self, num_subnets=10, schedule_type='progressive',
                 warmup_epochs=1000, total_epochs=10000,
                 initial_scales=None, final_scales=None):
        self.num_subnets = num_subnets
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        if initial_scales is None:
            self.initial_scales = [1.0] * num_subnets
        else:
            self.initial_scales = initial_scales

        if final_scales is None:
            self.final_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            if len(self.final_scales) != num_subnets:
                self.final_scales = np.linspace(0.5, 10.0, num_subnets).tolist()
        else:
            self.final_scales = final_scales

        self.current_scales = list(self.initial_scales)
        self.epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        if self.schedule_type == 'progressive':
            progress = min(1.0, self.epoch / max(1, self.warmup_epochs))
            for i in range(self.num_subnets):
                self.current_scales[i] = (
                    self.initial_scales[i] + progress * (self.final_scales[i] - self.initial_scales[i])
                )

        elif self.schedule_type == 'cyclical':
            cycle_length = max(1, self.total_epochs // 4)
            phase = (self.epoch % cycle_length) / cycle_length
            for i in range(self.num_subnets):
                mid = (self.initial_scales[i] + self.final_scales[i]) / 2
                amp = (self.final_scales[i] - self.initial_scales[i]) / 2
                self.current_scales[i] = mid + amp * math.cos(math.pi * (1 + phase))

        elif self.schedule_type == 'staircase':
            n_stairs = 5
            stair_length = max(1, self.warmup_epochs // n_stairs)
            stair_idx = min(n_stairs, self.epoch // stair_length)
            progress = stair_idx / n_stairs
            for i in range(self.num_subnets):
                self.current_scales[i] = (
                    self.initial_scales[i] + progress * (self.final_scales[i] - self.initial_scales[i])
                )

        elif self.schedule_type == 'subnet_progressive':
            n_active = min(self.num_subnets, 1 + int(self.epoch / max(1, self.warmup_epochs / self.num_subnets)))
            for i in range(self.num_subnets):
                if i < n_active:
                    self.current_scales[i] = self.final_scales[i]
                else:
                    self.current_scales[i] = 0.0

        return list(self.current_scales)

    def get_scales(self):
        return list(self.current_scales)

    def get_active_subnet_indices(self):
        return [i for i, s in enumerate(self.current_scales) if s > 0]


class SpectralNormSubnetWrapper(nn.Module):
    def __init__(self, subnet, spectral_norm_coeff=1.0):
        super().__init__()
        self.subnet = subnet
        self.spectral_norm_coeff = spectral_norm_coeff

        for name, module in subnet.named_modules():
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module, name='weight')

    def forward(self, x):
        return self.subnet(x)

    def remove_spectral_norm(self):
        for name, module in self.subnet.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    nn.utils.remove_spectral_norm(module, name='weight')
                except ValueError:
                    pass


class MultiTaskHead(nn.Module):
    def __init__(self, hidden_dim, task_dims=None):
        super().__init__()
        if task_dims is None:
            task_dims = {
                'S0': 6,
                'S1': 6,
                'phi': 2,
            }

        self.task_dims = task_dims
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim)
            for name, dim in task_dims.items()
        })

    def forward(self, h):
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(h)
        return outputs

    def get_concatenated_output(self, h):
        outputs = self.forward(h)
        ordered = []
        if 'S0' in outputs:
            ordered.append(outputs['S0'])
        if 'S1' in outputs:
            ordered.append(outputs['S1'])
        if 'phi' in outputs:
            ordered.append(outputs['phi'])
        for name in outputs:
            if name not in ('S0', 'S1', 'phi'):
                ordered.append(outputs[name])
        return torch.cat(ordered, dim=-1)


class MultiTaskSubnetFRPINN(nn.Module):
    def __init__(self, input_dim=8, fourier_dim=64, hidden_dims=None,
                 activation=nn.Tanh, fourier_type='basic',
                 fourier_sigma=None, use_layer_norm=False,
                 dropout=0.0, task_dims=None,
                 init_strategy='xavier_uniform'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 200, 200, 200, 128]

        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.hidden_dims = hidden_dims

        if fourier_type == 'basic':
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2
        else:
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_dim, sigma=fourier_sigma
            )
            fourier_out_dim = fourier_dim * 2

        self.act = activation()

        self.hidden_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        in_dim = fourier_out_dim
        for i, h_dim in enumerate(hidden_dims):
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            self.residual_blocks.append(
                ResidualBlock(h_dim, activation, use_layer_norm, dropout)
                if in_dim == h_dim else None
            )
            in_dim = h_dim

        self.multi_task_head = MultiTaskHead(hidden_dims[-1], task_dims)

        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fourier_mapping(x)
        for i, linear in enumerate(self.hidden_layers):
            h = linear(h)
            h = self.act(h)
            if self.residual_blocks[i] is not None:
                h = self.residual_blocks[i](h)
        return self.multi_task_head.get_concatenated_output(h)

    def forward_task(self, x, task_name):
        h = self.fourier_mapping(x)
        for i, linear in enumerate(self.hidden_layers):
            h = linear(h)
            h = self.act(h)
            if self.residual_blocks[i] is not None:
                h = self.residual_blocks[i](h)
        outputs = self.multi_task_head(h)
        return outputs.get(task_name, None)


class SubnetGradientMonitor:
    def __init__(self, num_subnets, window_size=100):
        self.num_subnets = num_subnets
        self.window_size = window_size
        self.gradient_norms = {i: [] for i in range(num_subnets)}
        self.weight_norms = {i: [] for i in range(num_subnets)}
        self.update_ratios = {i: [] for i in range(num_subnets)}

    def record(self, model, subnet_idx):
        subnet = model.subnets[subnet_idx]
        grad_norm = 0.0
        weight_norm = 0.0
        for p in subnet.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm().item() ** 2
            weight_norm += p.data.norm().item() ** 2

        grad_norm = math.sqrt(grad_norm)
        weight_norm = math.sqrt(weight_norm)

        self.gradient_norms[subnet_idx].append(grad_norm)
        self.weight_norms[subnet_idx].append(weight_norm)

        if weight_norm > 0:
            ratio = grad_norm / (weight_norm + 1e-8)
        else:
            ratio = 0.0
        self.update_ratios[subnet_idx].append(ratio)

        for key in [self.gradient_norms, self.weight_norms, self.update_ratios]:
            if len(key[subnet_idx]) > self.window_size:
                key[subnet_idx] = key[subnet_idx][-self.window_size:]

    def get_subnet_health(self, subnet_idx):
        grad_norms = self.gradient_norms.get(subnet_idx, [])
        weight_norms = self.weight_norms.get(subnet_idx, [])
        ratios = self.update_ratios.get(subnet_idx, [])

        if not grad_norms:
            return {'status': 'unknown', 'grad_norm': 0.0, 'weight_norm': 0.0}

        avg_grad = np.mean(grad_norms[-10:]) if len(grad_norms) >= 10 else np.mean(grad_norms)
        avg_weight = np.mean(weight_norms[-10:]) if len(weight_norms) >= 10 else np.mean(weight_norms)
        avg_ratio = np.mean(ratios[-10:]) if len(ratios) >= 10 else np.mean(ratios)

        status = 'healthy'
        if avg_grad < 1e-10:
            status = 'vanishing_gradient'
        elif avg_grad > 1e4:
            status = 'exploding_gradient'
        elif avg_ratio < 1e-7:
            status = 'stagnant'
        elif avg_ratio > 1e2:
            status = 'unstable'

        return {
            'status': status,
            'avg_grad_norm': avg_grad,
            'avg_weight_norm': avg_weight,
            'avg_update_ratio': avg_ratio,
        }

    def get_all_health(self):
        return {i: self.get_subnet_health(i) for i in range(self.num_subnets)}

    def detect_dead_subnets(self, threshold=1e-8):
        dead = []
        for i in range(self.num_subnets):
            health = self.get_subnet_health(i)
            if health['status'] in ('vanishing_gradient', 'stagnant'):
                if health['avg_grad_norm'] < threshold:
                    dead.append(i)
        return dead


class FRPINNs(nn.Module):
    def __init__(self, num_subnets=10, input_dim=8, fourier_dim=64,
                 hidden_dims=None, output_dim=14,
                 scale_factors=None, activation=nn.Tanh,
                 aggregation='weighted_mean',
                 fourier_type='basic', fourier_sigma=None,
                 use_layer_norm=False, dropout=0.0,
                 residual_type='post', init_strategy='xavier_uniform',
                 subnet_type='flexible', use_spectral_norm=False,
                 diversity_weight=0.0, diversity_type='correlation'):
        super().__init__()
        if scale_factors is None:
            scale_factors = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        assert len(scale_factors) == num_subnets

        self.num_subnets = num_subnets
        self.scale_factors = scale_factors
        self.aggregation = aggregation
        self.subnet_type = subnet_type
        self.diversity_weight = diversity_weight

        self.register_buffer(
            'scale_factors_tensor',
            torch.tensor(scale_factors, dtype=torch.float32)
        )

        if subnet_type == 'paper':
            SubnetClass = SubnetworkFRPINNPaper
        elif subnet_type == 'attention':
            SubnetClass = AttentionSubnetFRPINN
        elif subnet_type == 'multiscale_residual':
            SubnetClass = MultiScaleResidualSubnet
        elif subnet_type == 'adaptive_depth':
            SubnetClass = AdaptiveDepthSubnet
        elif subnet_type == 'multitask':
            SubnetClass = MultiTaskSubnetFRPINN
        else:
            SubnetClass = SubnetworkFRPINN

        subnet_kwargs = dict(
            input_dim=input_dim,
            fourier_dim=fourier_dim,
            output_dim=output_dim,
            activation=activation,
            init_strategy=init_strategy,
        )
        if subnet_type not in ('paper', 'adaptive_depth'):
            subnet_kwargs.update(dict(
                hidden_dims=hidden_dims,
                fourier_type=fourier_type,
                fourier_sigma=fourier_sigma,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ))

        if subnet_type == 'flexible':
            subnet_kwargs['residual_type'] = residual_type

        self.subnets = nn.ModuleList([
            SubnetClass(**subnet_kwargs)
            for _ in range(num_subnets)
        ])

        if use_spectral_norm:
            self.subnets = nn.ModuleList([
                SpectralNormSubnetWrapper(subnet) for subnet in self.subnets
            ])

        if aggregation == 'learnable':
            self.aggregation_weights = nn.Parameter(
                torch.ones(num_subnets, dtype=torch.float32) / num_subnets
            )

        if diversity_weight > 0:
            self.diversity_regularizer = SubnetDiversityRegularizer(
                num_subnets, diversity_type=diversity_type
            )
        else:
            self.diversity_regularizer = None

        self.gradient_monitor = SubnetGradientMonitor(num_subnets)

    def _apply_scale(self, x, scale_factor):
        scale = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        scale[7] = scale_factor
        return x * scale.unsqueeze(0)

    def _aggregate(self, outputs):
        if self.aggregation == 'weighted_mean':
            stacked = torch.stack(outputs, dim=0)
            return stacked.sum(dim=0) / self.num_subnets
        elif self.aggregation == 'sum':
            stacked = torch.stack(outputs, dim=0)
            return stacked.sum(dim=0)
        elif self.aggregation == 'mean':
            stacked = torch.stack(outputs, dim=0)
            return stacked.mean(dim=0)
        elif self.aggregation == 'max':
            stacked = torch.stack(outputs, dim=0)
            return stacked.max(dim=0)[0]
        elif self.aggregation == 'median':
            stacked = torch.stack(outputs, dim=0)
            return stacked.median(dim=0)[0]
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
        else:
            stacked = torch.stack(outputs, dim=0)
            return stacked.sum(dim=0) / self.num_subnets

    def forward(self, x):
        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet(x_scaled)
            outputs.append(out_i / self.scale_factors[i])
        return self._aggregate(outputs)

    def forward_with_diversity(self, x):
        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet(x_scaled)
            outputs.append(out_i / self.scale_factors[i])

        aggregated = self._aggregate(outputs)

        diversity_loss = torch.tensor(0.0, device=x.device)
        if self.diversity_regularizer is not None and self.diversity_weight > 0:
            diversity_loss = self.diversity_weight * self.diversity_regularizer(outputs)

        return aggregated, diversity_loss

    def get_all_subnet_outputs(self, x):
        outputs = []
        for i, subnet in enumerate(self.subnets):
            x_scaled = self._apply_scale(x, self.scale_factors[i])
            out_i = subnet(x_scaled)
            outputs.append(out_i / self.scale_factors[i])
        return outputs

    def get_subnet_output(self, x, subnet_idx):
        x_scaled = self._apply_scale(x, self.scale_factors[subnet_idx])
        out = self.subnets[subnet_idx](x_scaled)
        return out / self.scale_factors[subnet_idx]

    def get_subnet_diversity(self, x):
        outputs = self.get_all_subnet_outputs(x)
        stacked = torch.stack(outputs, dim=0)
        mean_out = stacked.mean(dim=0)
        variance = ((stacked - mean_out.unsqueeze(0)) ** 2).mean()
        return variance.item()

    def get_subnet_correlations(self, x):
        outputs = self.get_all_subnet_outputs(x)
        n = len(outputs)
        correlations = torch.zeros(n, n, device=x.device)
        for i in range(n):
            for j in range(i, n):
                o_i = outputs[i].flatten()
                o_j = outputs[j].flatten()
                corr = torch.corrcoef(torch.stack([o_i, o_j]))[0, 1]
                correlations[i, j] = corr
                correlations[j, i] = corr
        return correlations

    def get_subnet_pairwise_distances(self, x, metric='l2'):
        outputs = self.get_all_subnet_outputs(x)
        n = len(outputs)
        distances = torch.zeros(n, n, device=x.device)
        for i in range(n):
            for j in range(i + 1, n):
                diff = outputs[i] - outputs[j]
                if metric == 'l2':
                    d = (diff ** 2).mean().sqrt()
                elif metric == 'l1':
                    d = diff.abs().mean()
                elif metric == 'cosine':
                    cos_sim = F.cosine_similarity(
                        outputs[i].flatten().unsqueeze(0),
                        outputs[j].flatten().unsqueeze(0)
                    )
                    d = 1.0 - cos_sim
                else:
                    d = (diff ** 2).mean().sqrt()
                distances[i, j] = d
                distances[j, i] = d
        return distances

    def freeze_subnet(self, subnet_idx):
        for param in self.subnets[subnet_idx].parameters():
            param.requires_grad = False

    def unfreeze_subnet(self, subnet_idx):
        for param in self.subnets[subnet_idx].parameters():
            param.requires_grad = True

    def freeze_all_subnets(self):
        for subnet in self.subnets:
            for param in subnet.parameters():
                param.requires_grad = False

    def unfreeze_all_subnets(self):
        for subnet in self.subnets:
            for param in subnet.parameters():
                param.requires_grad = True

    def get_subnet_parameters(self, subnet_idx):
        return list(self.subnets[subnet_idx].parameters())

    def get_subnet_param_count(self, subnet_idx):
        return sum(p.numel() for p in self.subnets[subnet_idx].parameters())

    def get_layerwise_lr_groups(self, base_lr=1e-3, lr_decay=0.9):
        groups = []
        for i, subnet in enumerate(self.subnets):
            subnet_lr = base_lr * (lr_decay ** i)
            groups.append({
                'params': list(subnet.parameters()),
                'lr': subnet_lr,
                'name': f'subnet_{i}',
            })
        return groups

    def record_gradient_stats(self):
        for i in range(self.num_subnets):
            self.gradient_monitor.record(self, i)

    def get_subnet_health_report(self):
        return self.gradient_monitor.get_all_health()

    def detect_dead_subnets(self, threshold=1e-8):
        return self.gradient_monitor.detect_dead_subnets(threshold)

    def get_model_info(self):
        info = {
            'model_type': 'FRPINNs',
            'num_subnets': self.num_subnets,
            'scale_factors': self.scale_factors,
            'aggregation': self.aggregation,
            'subnet_type': self.subnet_type,
            'total_params': count_parameters(self),
            'trainable_params': count_parameters(self, trainable_only=True),
        }
        subnet_params = []
        for i in range(self.num_subnets):
            subnet_params.append(self.get_subnet_param_count(i))
        info['subnet_params'] = subnet_params
        info['subnet_params_mean'] = np.mean(subnet_params)
        info['subnet_params_std'] = np.std(subnet_params)
        return info

    def print_model_info(self):
        info = self.get_model_info()
        print("=" * 60)
        print("FRPINNs Model Information")
        print("=" * 60)
        print(f"Number of subnets: {info['num_subnets']}")
        print(f"Scale factors: {info['scale_factors']}")
        print(f"Aggregation: {info['aggregation']}")
        print(f"Subnet type: {info['subnet_type']}")
        print(f"Total parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"Parameters per subnet: {info['subnet_params']}")
        print(f"Mean params per subnet: {info['subnet_params_mean']:.0f}")
        print(f"Std params per subnet: {info['subnet_params_std']:.0f}")
        print("=" * 60)


class FRPINNsEnsemble(nn.Module):
    def __init__(self, num_models=3, **frpinn_kwargs):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            FRPINNs(**frpinn_kwargs) for _ in range(num_models)
        ])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        stacked = torch.stack(outputs, dim=0)
        return stacked.mean(dim=0)

    def predict_with_uncertainty(self, x):
        outputs = [model(x) for model in self.models]
        stacked = torch.stack(outputs, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std

    def predict_with_confidence(self, x, confidence=0.95):
        mean, std = self.predict_with_uncertainty(x)
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        lower = mean - z * std
        upper = mean + z * std
        return mean, std, lower, upper

    def get_model(self, idx):
        return self.models[idx]

    def get_ensemble_diversity(self, x):
        outputs = [model(x) for model in self.models]
        stacked = torch.stack(outputs, dim=0)
        mean_out = stacked.mean(dim=0)
        variance = ((stacked - mean_out.unsqueeze(0)) ** 2).mean()
        return variance.item()


class FRPINNsCurriculum(nn.Module):
    def __init__(self, num_subnets=10, input_dim=8, fourier_dim=64,
                 hidden_dims=None, output_dim=14,
                 scale_factors=None, activation=nn.Tanh,
                 curriculum_type='progressive',
                 warmup_epochs=1000, total_epochs=10000,
                 **frpinn_kwargs):
        super().__init__()
        self.base_model = FRPINNs(
            num_subnets=num_subnets,
            input_dim=input_dim,
            fourier_dim=fourier_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            scale_factors=scale_factors,
            activation=activation,
            **frpinn_kwargs
        )

        self.curriculum_scheduler = CurriculumScaleScheduler(
            num_subnets=num_subnets,
            schedule_type=curriculum_type,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            final_scales=scale_factors,
        )

        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        new_scales = self.curriculum_scheduler.step(epoch)
        self.base_model.scale_factors = new_scales
        self.base_model.scale_factors_tensor.copy_(
            torch.tensor(new_scales, dtype=torch.float32)
        )

    def forward(self, x):
        return self.base_model(x)

    def get_curriculum_info(self):
        return {
            'current_epoch': self.current_epoch,
            'current_scales': self.curriculum_scheduler.get_scales(),
            'active_subnets': self.curriculum_scheduler.get_active_subnet_indices(),
        }


class FRPINNsCheckpointManager:
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

        path = os.path.join(self.save_dir, f'frpinn_epoch_{epoch:06d}.pt')
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


def create_frpinn_model(config=None):
    if config is None:
        config = {}

    model_type = config.get('model_type', 'base')

    if model_type == 'base':
        return FRPINNs(
            num_subnets=config.get('num_subnets', 10),
            input_dim=config.get('input_dim', 8),
            fourier_dim=config.get('fourier_dim', 64),
            hidden_dims=config.get('hidden_dims', None),
            output_dim=config.get('output_dim', 14),
            scale_factors=config.get('scale_factors', None),
            activation=config.get('activation', nn.Tanh),
            aggregation=config.get('aggregation', 'weighted_mean'),
            fourier_type=config.get('fourier_type', 'basic'),
            fourier_sigma=config.get('fourier_sigma', None),
            use_layer_norm=config.get('use_layer_norm', False),
            dropout=config.get('dropout', 0.0),
            residual_type=config.get('residual_type', 'post'),
            init_strategy=config.get('init_strategy', 'xavier_uniform'),
            subnet_type=config.get('subnet_type', 'flexible'),
            use_spectral_norm=config.get('use_spectral_norm', False),
            diversity_weight=config.get('diversity_weight', 0.0),
            diversity_type=config.get('diversity_type', 'correlation'),
        )

    elif model_type == 'ensemble':
        return FRPINNsEnsemble(
            num_models=config.get('num_models', 3),
            num_subnets=config.get('num_subnets', 10),
            input_dim=config.get('input_dim', 8),
            fourier_dim=config.get('fourier_dim', 64),
            hidden_dims=config.get('hidden_dims', None),
            output_dim=config.get('output_dim', 14),
            scale_factors=config.get('scale_factors', None),
            activation=config.get('activation', nn.Tanh),
        )

    elif model_type == 'curriculum':
        return FRPINNsCurriculum(
            num_subnets=config.get('num_subnets', 10),
            input_dim=config.get('input_dim', 8),
            fourier_dim=config.get('fourier_dim', 64),
            hidden_dims=config.get('hidden_dims', None),
            output_dim=config.get('output_dim', 14),
            scale_factors=config.get('scale_factors', None),
            activation=config.get('activation', nn.Tanh),
            curriculum_type=config.get('curriculum_type', 'progressive'),
            warmup_epochs=config.get('warmup_epochs', 1000),
            total_epochs=config.get('total_epochs', 10000),
        )

    else:
        return FRPINNs(
            num_subnets=config.get('num_subnets', 10),
            input_dim=config.get('input_dim', 8),
            fourier_dim=config.get('fourier_dim', 64),
            output_dim=config.get('output_dim', 14),
        )


def analyze_subnet_contributions(model, x, n_samples=100):
    results = {
        'individual_losses': [],
        'agreement_scores': [],
        'scale_effect': [],
    }

    with torch.no_grad():
        all_outputs = model.get_all_subnet_outputs(x)
        aggregated = model(x)

        for i, out_i in enumerate(all_outputs):
            diff = (out_i - aggregated).abs().mean().item()
            results['individual_losses'].append(diff)

        for i in range(len(all_outputs)):
            for j in range(i + 1, len(all_outputs)):
                agreement = F.cosine_similarity(
                    all_outputs[i].flatten().unsqueeze(0),
                    all_outputs[j].flatten().unsqueeze(0)
                ).item()
                results['agreement_scores'].append(
                    (i, j, agreement)
                )

        for i, sf in enumerate(model.scale_factors):
            x_scaled = model._apply_scale(x, sf)
            out_scaled = model.subnets[i](x_scaled)
            out_unscaled = model.subnets[i](x)
            scale_effect = (out_scaled / sf - out_unscaled).abs().mean().item()
            results['scale_effect'].append((sf, scale_effect))

    return results
