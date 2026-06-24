import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Optional, Tuple, Dict


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64, sigma=None,
                 learnable_scale=False, mapping_type='basic'):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.mapping_type = mapping_type

        self.B = nn.Parameter(torch.empty(feature_dim, input_dim))
        self.b_f = nn.Parameter(torch.empty(feature_dim))

        if learnable_scale:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('log_sigma', torch.tensor(0.0))

        if sigma is not None:
            with torch.no_grad():
                nn.init.normal_(self.B, mean=0.0, std=sigma)
        else:
            self._init_weights()

        nn.init.zeros_(self.b_f)

    def _init_weights(self):
        std = math.sqrt(2.0 / (self.input_dim + self.feature_dim))
        nn.init.normal_(self.B, mean=0.0, std=std)

    def forward(self, x):
        scale = torch.exp(self.log_sigma)
        linear_part = torch.matmul(x, (scale * self.B).t()) + self.b_f
        cos_feat = torch.cos(linear_part)
        sin_feat = torch.sin(linear_part)
        return torch.cat([cos_feat, sin_feat], dim=-1)

    def output_dim(self):
        return self.feature_dim * 2


class MultiBandFourierMapping(nn.Module):
    def __init__(self, input_dim=8, bands=None, feature_dim_per_band=32):
        super().__init__()
        if bands is None:
            bands = [1.0, 2.0, 4.0, 8.0, 16.0]

        self.input_dim = input_dim
        self.bands = bands
        self.feature_dim_per_band = feature_dim_per_band
        self.num_bands = len(bands)

        self.band_mappings = nn.ModuleList([
            FourierFeatureMapping(
                input_dim=input_dim,
                feature_dim=feature_dim_per_band,
                sigma=sigma
            )
            for sigma in bands
        ])

    def forward(self, x):
        features = []
        for mapping in self.band_mappings:
            features.append(mapping(x))
        return torch.cat(features, dim=-1)

    def output_dim(self):
        return self.feature_dim_per_band * 2 * self.num_bands


class RandomFourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64, sigma=1.0,
                 freeze_B=True):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        B = torch.randn(feature_dim, input_dim) * sigma
        if freeze_B:
            self.register_buffer('B', B)
        else:
            self.B = nn.Parameter(B)

    def forward(self, x):
        linear_part = torch.matmul(x, self.B.t())
        cos_feat = torch.cos(2.0 * math.pi * linear_part)
        sin_feat = torch.sin(2.0 * math.pi * linear_part)
        return torch.cat([cos_feat, sin_feat], dim=-1)

    def output_dim(self):
        return self.feature_dim * 2


class PositionalEncodingFourier(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64, max_freq=10.0,
                 num_freqs=4):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_freqs = num_freqs

        freq_bands = torch.linspace(1.0, max_freq, num_freqs)
        self.register_buffer('freq_bands', freq_bands)

        self.projection = nn.Linear(input_dim * num_freqs * 2, feature_dim)

    def forward(self, x):
        features = []
        for freq in self.freq_bands:
            features.append(torch.cos(2.0 * math.pi * freq * x))
            features.append(torch.sin(2.0 * math.pi * freq * x))
        multi_freq = torch.cat(features, dim=-1)
        return self.projection(multi_freq)

    def output_dim(self):
        return self.feature_dim


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation=nn.Tanh,
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation()

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return residual + out


class PreActivationResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation=nn.Tanh,
                 use_layer_norm=False):
        super().__init__()
        self.activation = activation()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def forward(self, x):
        out = self.activation(x)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return x + self.linear(out)


class Swish(nn.Module):
    def __init__(self, beta=1.0, learnable=False):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Sine(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


def get_activation(name, **kwargs):
    activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'swish': lambda: Swish(kwargs.get('beta', 1.0), kwargs.get('learnable', False)),
        'sine': lambda: Sine(kwargs.get('omega_0', 30.0)),
        'softplus': nn.Softplus,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU,
        'mish': nn.Mish,
    }
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    act_cls = activations[name_lower]
    if callable(act_cls) and not isinstance(act_cls, type):
        return act_cls()
    return act_cls()


class XavierUniformInit:
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=self.gain)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)


class XavierNormalInit:
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=self.gain)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                std = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                nn.init.normal_(module.bias, mean=0.0, std=std)


class KaimingUniformInit:
    def __init__(self, mode='fan_in', nonlinearity='leaky_relu', a=0.0):
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.a = a

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(
                module.weight, mode=self.mode,
                nonlinearity=self.nonlinearity, a=self.a
            )
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)


class KaimingNormalInit:
    def __init__(self, mode='fan_in', nonlinearity='leaky_relu', a=0.0):
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.a = a

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, mode=self.mode,
                nonlinearity=self.nonlinearity, a=self.a
            )
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                std = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                nn.init.normal_(module.bias, mean=0.0, std=std)


class OrthogonalInit:
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=self.gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class TruncatedNormalInit:
    def __init__(self, std=0.02, mean=0.0):
        self.std = std
        self.mean = mean

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight, mean=self.mean, std=self.std,
                a=-2.0 * self.std, b=2.0 * self.std
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def get_initializer(name, **kwargs):
    initializers = {
        'xavier_uniform': XavierUniformInit,
        'xavier_normal': XavierNormalInit,
        'kaiming_uniform': KaimingUniformInit,
        'kaiming_normal': KaimingNormalInit,
        'orthogonal': OrthogonalInit,
        'truncated_normal': TruncatedNormalInit,
    }
    name_lower = name.lower()
    if name_lower not in initializers:
        raise ValueError(
            f"Unknown initializer: {name}. Available: {list(initializers.keys())}"
        )
    return initializers[name_lower](**kwargs)


def apply_weight_init(module, init_strategy):
    if isinstance(init_strategy, str):
        init_strategy = get_initializer(init_strategy)
    module.apply(init_strategy)


class GradientMonitor:
    def __init__(self):
        self.gradient_stats = {}
        self._hooks = []

    def register_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._collect_stats(n, grad)
                )
                self._hooks.append(hook)

    def _collect_stats(self, name, grad):
        if name not in self.gradient_stats:
            self.gradient_stats[name] = {
                'norms': [], 'max_vals': [], 'min_vals': [],
                'mean_vals': [], 'has_nan': [], 'has_inf': []
            }
        stats = self.gradient_stats[name]
        stats['norms'].append(grad.norm().item())
        stats['max_vals'].append(grad.max().item())
        stats['min_vals'].append(grad.min().item())
        stats['mean_vals'].append(grad.mean().item())
        stats['has_nan'].append(torch.isnan(grad).any().item())
        stats['has_inf'].append(torch.isinf(grad).any().item())

    def get_summary(self, last_n=100):
        summary = {}
        for name, stats in self.gradient_stats.items():
            n = min(last_n, len(stats['norms']))
            if n == 0:
                continue
            summary[name] = {
                'avg_norm': np.mean(stats['norms'][-n:]),
                'max_norm': np.max(stats['norms'][-n:]),
                'min_norm': np.min(stats['norms'][-n:]),
                'avg_mean': np.mean(stats['mean_vals'][-n:]),
                'nan_count': sum(stats['has_nan'][-n:]),
                'inf_count': sum(stats['has_inf'][-n:]),
            }
        return summary

    def get_layer_wise_norms(self, last_n=100):
        norms = {}
        for name, stats in self.gradient_stats.items():
            n = min(last_n, len(stats['norms']))
            if n > 0:
                norms[name] = np.mean(stats['norms'][-n:])
        return norms

    def reset(self):
        self.gradient_stats = {}

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class GradientCheckpointWrapper(nn.Module):
    def __init__(self, module, num_segments=2):
        super().__init__()
        self.module = module
        self.num_segments = num_segments

    def forward(self, x):
        if not self.training or self.num_segments <= 1:
            return self.module(x)

        from torch.utils.checkpoint import checkpoint

        layers = list(self.module.children())
        if len(layers) == 0:
            return self.module(x)

        segment_size = max(1, len(layers) // self.num_segments)
        segments = []
        for i in range(0, len(layers), segment_size):
            segment = nn.Sequential(*layers[i:i + segment_size])
            segments.append(segment)

        out = x
        for segment in segments:
            out = checkpoint(segment, out, use_reentrant=False)
        return out


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model, input_shape=None, device='cpu'):
    info = {}
    info['total_params'] = count_parameters(model, trainable_only=False)
    info['trainable_params'] = count_parameters(model, trainable_only=True)
    info['non_trainable_params'] = info['total_params'] - info['trainable_params']

    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_info.append({
                'name': name,
                'type': 'Linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'params': module.weight.numel() + (
                    module.bias.numel() if module.bias is not None else 0
                ),
            })
        elif isinstance(module, (nn.Tanh, nn.ReLU, nn.GELU, nn.SiLU, nn.Softplus,
                                 nn.ELU, nn.LeakyReLU, nn.Mish)):
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'in_features': None,
                'out_features': None,
                'params': 0,
            })

    info['layers'] = layer_info

    if input_shape is not None:
        try:
            dummy = torch.randn(*input_shape, device=device)
            with torch.no_grad():
                out = model(dummy)
            if isinstance(out, torch.Tensor):
                info['output_shape'] = tuple(out.shape)
            else:
                info['output_shape'] = 'complex'
        except Exception as e:
            info['output_shape'] = f'error: {str(e)}'

    return info


def format_model_summary(info):
    lines = []
    lines.append("=" * 70)
    lines.append("Model Summary")
    lines.append("=" * 70)
    lines.append(f"Total params:     {info['total_params']:>12,}")
    lines.append(f"Trainable params: {info['trainable_params']:>12,}")
    lines.append(f"Non-trainable:    {info['non_trainable_params']:>12,}")
    lines.append("-" * 70)
    lines.append(f"{'Layer':<35} {'Type':<12} {'In':>6} {'Out':>6} {'Params':>10}")
    lines.append("-" * 70)
    for layer in info.get('layers', []):
        name = layer['name'][:33] if layer['name'] else ''
        in_f = str(layer['in_features']) if layer['in_features'] else '-'
        out_f = str(layer['out_features']) if layer['out_features'] else '-'
        lines.append(
            f"{name:<35} {layer['type']:<12} {in_f:>6} {out_f:>6} "
            f"{layer['params']:>10,}"
        )
    lines.append("=" * 70)
    if 'output_shape' in info:
        lines.append(f"Output shape: {info['output_shape']}")
    return "\n".join(lines)


def compute_ntk_approximation(model, x1, x2=None, subsample=500):
    if x2 is None:
        x2 = x1

    n1 = min(x1.shape[0], subsample)
    n2 = min(x2.shape[0], subsample)
    x1_sub = x1[:n1].detach().requires_grad_(True)
    x2_sub = x2[:n2].detach().requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]

    out1 = model(x1_sub)
    out2 = model(x2_sub)

    output_dim = out1.shape[-1]
    ntk_diag = torch.zeros(output_dim, device=x1.device)

    for c in range(output_dim):
        grad1_list = []
        for i in range(n1):
            model.zero_grad()
            out1_i = model(x1_sub[i:i+1])
            grad1 = torch.autograd.grad(
                out1_i[0, c], params, retain_graph=True, allow_unused=True
            )
            grad1_vec = torch.cat([
                g.reshape(-1) for g in grad1 if g is not None
            ])
            grad1_list.append(grad1_vec)

        for j in range(n2):
            model.zero_grad()
            out2_j = model(x2_sub[j:j+1])
            grad2 = torch.autograd.grad(
                out2_j[0, c], params, retain_graph=True, allow_unused=True
            )
            grad2_vec = torch.cat([
                g.reshape(-1) for g in grad2 if g is not None
            ])

            for grad1_vec in grad1_list:
                ntk_diag[c] += (grad1_vec * grad2_vec).sum()

    ntk_diag /= (n1 * n2)
    return ntk_diag


def spectral_analysis(model, x, n_harmonics=50):
    with torch.no_grad():
        pred = model(x)

    if pred.dim() == 2:
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred.cpu().numpy()

    spectra = {}
    for c in range(pred_np.shape[-1]):
        signal = pred_np[:, c]
        spectrum = np.abs(np.fft.rfft(signal - signal.mean(), n=n_harmonics * 2))
        spectra[c] = spectrum[:n_harmonics]

    return spectra


def compute_gradient_norm(model, norm_type=2.0):
    total_norm = 0.0
    per_layer_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type).item()
            total_norm += param_norm ** norm_type
            per_layer_norms[name] = param_norm
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm, per_layer_norms


def compute_weight_norm(model, norm_type=2.0):
    total_norm = 0.0
    per_layer_norms = {}
    for name, param in model.named_parameters():
        param_norm = param.data.norm(norm_type).item()
        total_norm += param_norm ** norm_type
        per_layer_norms[name] = param_norm
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm, per_layer_norms


def check_model_health(model):
    issues = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN in parameter: {name}")
        if torch.isinf(param).any():
            issues.append(f"Inf in parameter: {name}")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN in gradient: {name}")
            if torch.isinf(param.grad).any():
                issues.append(f"Inf in gradient: {name}")

    for name, buf in model.named_buffers():
        if torch.isnan(buf).any():
            issues.append(f"NaN in buffer: {name}")
        if torch.isinf(buf).any():
            issues.append(f"Inf in buffer: {name}")

    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'num_issues': len(issues)
    }


class GaussianFourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64, sigma=1.0,
                 scale_schedule='constant', learnable_sigma=False):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.scale_schedule = scale_schedule

        B = torch.randn(feature_dim, input_dim) * sigma
        if learnable_sigma:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        else:
            self.register_buffer('log_sigma', torch.tensor(math.log(sigma)))

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        linear_part = torch.matmul(x, (sigma * self.B).t())
        cos_feat = torch.cos(2.0 * math.pi * linear_part)
        sin_feat = torch.sin(2.0 * math.pi * linear_part)
        return torch.cat([cos_feat, sin_feat], dim=-1)

    def output_dim(self):
        return self.feature_dim * 2


class TrainableFourierMapping(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64, num_frequencies=4,
                 max_freq=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_frequencies = num_frequencies

        self.freq_weights = nn.Parameter(
            torch.linspace(1.0, max_freq, num_frequencies).unsqueeze(1).expand(
                num_frequencies, input_dim
            ).clone()
        )
        self.phase_shifts = nn.Parameter(
            torch.zeros(num_frequencies, input_dim)
        )
        self.projection = nn.Linear(
            input_dim * num_frequencies * 2, feature_dim
        )

    def forward(self, x):
        features = []
        for i in range(self.num_frequencies):
            scaled = self.freq_weights[i] * x + self.phase_shifts[i]
            features.append(torch.cos(2.0 * math.pi * scaled))
            features.append(torch.sin(2.0 * math.pi * scaled))
        multi_freq = torch.cat(features, dim=-1)
        return self.projection(multi_freq)

    def output_dim(self):
        return self.feature_dim


class ProgressiveFourierMapping(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64, max_sigma=10.0,
                 warmup_epochs=1000):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.max_sigma = max_sigma
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.B = nn.Parameter(torch.randn(feature_dim, input_dim))
        self.b_f = nn.Parameter(torch.zeros(feature_dim))

        self.register_buffer('sigma_progress', torch.tensor(0.0))

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        progress = min(1.0, epoch / max(1, self.warmup_epochs))
        self.sigma_progress.fill_(progress * self.max_sigma)

    def forward(self, x):
        sigma = self.sigma_progress
        linear_part = torch.matmul(x, (sigma * self.B).t()) + self.b_f
        cos_feat = torch.cos(linear_part)
        sin_feat = torch.sin(linear_part)
        return torch.cat([cos_feat, sin_feat], dim=-1)

    def output_dim(self):
        return self.feature_dim * 2


class HighwayBlock(nn.Module):
    def __init__(self, hidden_dim, activation=nn.Tanh,
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        self.transform_gate = nn.Linear(hidden_dim, hidden_dim)
        self.carry_gate = nn.Linear(hidden_dim, hidden_dim)
        self.transform = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation()

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        T = torch.sigmoid(self.transform_gate(x))
        C = 1.0 - T
        H = self.activation(self.transform(x))
        if self.layer_norm is not None:
            H = self.layer_norm(H)
        if self.dropout is not None:
            H = self.dropout(H)
        return T * H + C * x


class DenseBlock(nn.Module):
    def __init__(self, hidden_dim, growth_rate=32, num_layers=4,
                 activation=nn.Tanh, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        for i in range(num_layers):
            in_dim = hidden_dim + growth_rate * i
            self.layers.append(nn.Linear(in_dim, growth_rate))

        self.transition = nn.Linear(
            hidden_dim + growth_rate * num_layers, hidden_dim
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = self.activation(layer(torch.cat(features, dim=-1)))
            if self.dropout is not None:
                out = self.dropout(out)
            features.append(out)
        return self.transition(torch.cat(features, dim=-1))


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, hidden_dim, reduction=4, activation=nn.Tanh):
        super().__init__()
        self.squeeze = nn.Linear(hidden_dim, hidden_dim // reduction)
        self.excitation = nn.Linear(hidden_dim // reduction, hidden_dim)
        self.activation = activation()

    def forward(self, x):
        scale = self.activation(self.squeeze(x))
        scale = torch.sigmoid(self.excitation(scale))
        return x * scale


class Snake(nn.Module):
    def __init__(self, hidden_dim=1, learnable_alpha=True):
        super().__init__()
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.ones(hidden_dim))
        else:
            self.register_buffer('alpha', torch.ones(hidden_dim))

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class GatedActivation(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        gate = torch.sigmoid(self.fc_gate(x))
        value = torch.tanh(self.fc_value(x))
        return gate * value


class MILossActivation(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        return self.alpha * torch.tanh(self.beta * x)


class UniformInit:
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, self.a, self.b)
            if module.bias is not None:
                nn.init.uniform_(module.bias, self.a, self.b)


class ConstantInit:
    def __init__(self, weight_val=0.0, bias_val=0.0):
        self.weight_val = weight_val
        self.bias_val = bias_val

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, self.weight_val)
            if module.bias is not None:
                nn.init.constant_(module.bias, self.bias_val)


class NormalInit:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=self.mean, std=self.std)
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=self.mean, std=self.std)


class ScaledUniformInit:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            limit = self.scale / math.sqrt(fan_in) if fan_in > 0 else self.scale
            nn.init.uniform_(module.weight, -limit, limit)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -limit, limit)


class LSUVInit:
    def __init__(self, variance_scale=1.0, max_iter=10, tol=1e-3):
        self.variance_scale = variance_scale
        self.max_iter = max_iter
        self.tol = tol

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class SpectralNormWrapper(nn.Module):
    def __init__(self, module, n_power_iterations=1, eps=1e-12):
        super().__init__()
        self.module = torch.nn.utils.spectral_norm(
            module, n_power_iterations=n_power_iterations, eps=eps
        )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class StochasticWeightAveraging:
    def __init__(self, model, avg_start=0, avg_freq=1, device=None):
        self.model = model
        self.avg_start = avg_start
        self.avg_freq = avg_freq
        self.n_averaged = 0
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.avg_state_dict = None

    def update(self, epoch):
        if epoch < self.avg_start:
            return
        if (epoch - self.avg_start) % self.avg_freq != 0:
            return

        if self.avg_state_dict is None:
            self.avg_state_dict = {
                k: v.clone().to(self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            for k, v in self.model.state_dict().items():
                self.avg_state_dict[k].mul_(self.n_averaged).add_(v.to(self.device))
                self.avg_state_dict[k].div_(self.n_averaged + 1)

        self.n_averaged += 1

    def apply_avg_weights(self):
        if self.avg_state_dict is not None and self.n_averaged > 0:
            self.model.load_state_dict(self.avg_state_dict)
            return True
        return False

    def get_avg_state_dict(self):
        return self.avg_state_dict


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().to(self.device)

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.to(self.device), alpha=1.0 - self.decay)

    def apply_ema_weights(self):
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore_weights(self):
        if hasattr(self, 'backup'):
            self.model.load_state_dict(self.backup)
            del self.backup


def compute_jacobian(model, x, output_idx=None):
    x = x.detach().requires_grad_(True)
    output = model(x)

    if output_idx is not None:
        output = output[:, output_idx]

    batch_size = x.shape[0]
    output_dim = output.shape[-1]
    input_dim = x.shape[-1]

    jacobian = torch.zeros(batch_size, output_dim, input_dim, device=x.device)
    for i in range(output_dim):
        model.zero_grad()
        grad = torch.autograd.grad(
            output[:, i].sum(), x, retain_graph=True, create_graph=False
        )[0]
        jacobian[:, i, :] = grad

    return jacobian


def compute_eigenvalues(model, x, n_samples=100):
    x_sub = x[:n_samples].detach().requires_grad_(True)
    output = model(x_sub)

    output_dim = output.shape[-1]
    params = [p for p in model.parameters() if p.requires_grad]

    eigenvalues = {}
    for c in range(min(output_dim, 5)):
        model.zero_grad()
        grad = torch.autograd.grad(
            output[:, c].sum(), params, create_graph=True
        )
        grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])

        hessian_vec_prod = torch.autograd.grad(
            grad_vec.sum(), params, retain_graph=False
        )
        hvp = torch.cat([g.reshape(-1) for g in hessian_vec_prod if g is not None])

        eigenvalues[c] = {
            'grad_norm': grad_vec.norm().item(),
            'hvp_norm': hvp.norm().item(),
            'curvature_ratio': (hvp.norm() / (grad_vec.norm() + 1e-30)).item(),
        }

    return eigenvalues


def layer_wise_lr(model, base_lr=1e-3, decay_factor=0.9,
                  min_lr=1e-7, layer_type=nn.Linear):
    param_groups = []
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, layer_type)]
    n_layers = len(layers)

    for i, (name, module) in enumerate(layers):
        lr = max(base_lr * (decay_factor ** (n_layers - 1 - i)), min_lr)
        param_groups.append({
            'params': list(module.parameters()),
            'lr': lr,
            'name': name,
        })

    return param_groups


def compute_condition_number(model, x):
    x = x.detach().requires_grad_(True)
    output = model(x)

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    grad_dict = {}
    for c in range(output.shape[-1]):
        model.zero_grad()
        grad = torch.autograd.grad(
            output[:, c].sum(), params, retain_graph=True, allow_unused=True
        )
        grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])
        grad_dict[c] = grad_vec

    gram_matrix = torch.zeros(
        output.shape[-1], output.shape[-1], device=x.device
    )
    for i in range(output.shape[-1]):
        for j in range(i, output.shape[-1]):
            gram_matrix[i, j] = torch.dot(grad_dict[i], grad_dict[j])
            gram_matrix[j, i] = gram_matrix[i, j]

    try:
        eigenvalues = torch.linalg.eigvalsh(gram_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) > 0:
            condition_number = (eigenvalues.max() / eigenvalues.min()).item()
        else:
            condition_number = float('inf')
    except Exception:
        condition_number = float('inf')

    return {
        'condition_number': condition_number,
        'gram_trace': gram_matrix.trace().item(),
        'gram_rank': torch.linalg.matrix_rank(gram_matrix).item(),
        'n_params': n_params,
    }


def apply_spectral_norm_to_linear(model, n_power_iterations=1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            torch.nn.utils.spectral_norm(
                module, n_power_iterations=n_power_iterations
            )
    return model


def remove_spectral_norm(model):
    for name, module in model.named_modules():
        try:
            torch.nn.utils.remove_spectral_norm(module)
        except ValueError:
            pass
    return model


def get_fourier_mapping(name, **kwargs):
    mappings = {
        'basic': FourierFeatureMapping,
        'multiband': MultiBandFourierMapping,
        'random': RandomFourierFeatureMapping,
        'positional': PositionalEncodingFourier,
        'gaussian': GaussianFourierFeatureMapping,
        'trainable': TrainableFourierMapping,
        'progressive': ProgressiveFourierMapping,
    }
    name_lower = name.lower()
    if name_lower not in mappings:
        raise ValueError(
            f"Unknown Fourier mapping: {name}. Available: {list(mappings.keys())}"
        )
    return mappings[name_lower](**kwargs)


def get_residual_block(name, **kwargs):
    blocks = {
        'post': ResidualBlock,
        'pre': PreActivationResidualBlock,
        'highway': HighwayBlock,
        'dense': DenseBlock,
        'se': SqueezeExcitationBlock,
    }
    name_lower = name.lower()
    if name_lower not in blocks:
        raise ValueError(
            f"Unknown residual block: {name}. Available: {list(blocks.keys())}"
        )
    return blocks[name_lower](**kwargs)
