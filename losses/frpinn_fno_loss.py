import torch
import torch.nn as nn
import math
import numpy as np
from collections import OrderedDict


class AdaptiveLossWeightsFNO(nn.Module):
    def __init__(self, num_losses=5, init_weights=None, eps=1e-8, alpha=0.1):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.num_losses = num_losses
        if init_weights is None:
            init_weights = [1.0] * num_losses
        self.log_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(math.log(w), dtype=torch.float32))
            for w in init_weights
        ])
        self.register_buffer('running_means',
                             torch.ones(num_losses, dtype=torch.float32))
        self.register_buffer('running_vars',
                             torch.ones(num_losses, dtype=torch.float32))
        self._initialized = False

    def forward(self):
        weights = [torch.exp(lw) for lw in self.log_weights]
        return weights

    def normalize_losses(self, losses):
        losses_tensor = torch.stack([l.detach() for l in losses])
        if not self._initialized:
            self.running_means.copy_(losses_tensor + self.eps)
            self.running_vars.copy_(torch.ones_like(losses_tensor))
            self._initialized = True
        else:
            delta = losses_tensor - self.running_means
            self.running_means.mul_(1.0 - self.alpha).add_(
                self.alpha * (losses_tensor + self.eps)
            )
            self.running_vars.mul_(1.0 - self.alpha).add_(
                self.alpha * (delta ** 2)
            )
        normalized = []
        for i, l in enumerate(losses):
            std = torch.sqrt(self.running_vars[i] + self.eps)
            normalized.append(l / std)
        return normalized

    def get_weights(self):
        with torch.no_grad():
            return [torch.exp(lw).item() for lw in self.log_weights]


class NTKAdaptiveWeightsFNO(nn.Module):
    def __init__(self, num_losses=5, update_interval=200, eps=1e-8,
                 decay=0.9, subsample_size=100):
        super().__init__()
        self.num_losses = num_losses
        self.update_interval = update_interval
        self.eps = eps
        self.decay = decay
        self.subsample_size = subsample_size
        self.register_buffer('ntk_weights',
                             torch.ones(num_losses, dtype=torch.float32))
        self.register_buffer('step_count',
                             torch.tensor(0, dtype=torch.long))

    def compute_ntk_trace(self, model, x, medium_profile, output_indices=None):
        if output_indices is None:
            output_indices = list(range(14))
        n = x.shape[0]
        n_out = len(output_indices)
        if n > self.subsample_size:
            indices = torch.randperm(n)[:self.subsample_size]
            x_sub = x[indices]
        else:
            x_sub = x
        n_sub = x_sub.shape[0]
        x_grad = x_sub.clone().detach().requires_grad_(True)
        pred = model(x_grad, medium_profile)
        kernel_trace = torch.tensor(0.0, device=x.device)
        for idx in output_indices:
            output_i = pred[:, idx].sum()
            grads = torch.autograd.grad(
                output_i, model.parameters(),
                create_graph=False, retain_graph=True
            )
            grad_flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
            kernel_trace = kernel_trace + (grad_flat ** 2).sum()
        kernel_trace = kernel_trace / (n_sub * n_out + self.eps)
        return kernel_trace

    def update_weights(self, model, loss_inputs, medium_profiles):
        self.step_count += 1
        if self.step_count % self.update_interval != 0:
            return
        kernel_traces = []
        for i, (x_in, mp) in enumerate(zip(loss_inputs, medium_profiles)):
            if x_in is None or x_in.shape[0] == 0:
                kernel_traces.append(torch.tensor(1.0, device=x_in.device if x_in is not None else 'cpu'))
                continue
            trace = self.compute_ntk_trace(model, x_in, mp)
            kernel_traces.append(trace)
        traces_tensor = torch.stack(kernel_traces)
        traces_max = traces_tensor.max() + self.eps
        normalized_traces = traces_tensor / traces_max
        new_weights = 1.0 / (normalized_traces + self.eps)
        new_weights = new_weights / (new_weights.sum() + self.eps) * self.num_losses
        self.ntk_weights.mul_(self.decay).add_(
            (1.0 - self.decay) * new_weights
        )

    def get_weights(self):
        with torch.no_grad():
            return self.ntk_weights.tolist()

    def forward(self):
        return [self.ntk_weights[i] for i in range(self.num_losses)]


class CrossMediumConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-8, temperature=0.1):
        super().__init__()
        self.eps = eps
        self.temperature = temperature

    def forward(self, predictions_list, medium_profiles_list):
        if len(predictions_list) < 2:
            return torch.tensor(0.0)
        total_loss = torch.tensor(0.0, device=predictions_list[0].device)
        n_pairs = 0
        for i in range(len(predictions_list)):
            for j in range(i + 1, len(predictions_list)):
                pred_i = predictions_list[i]
                pred_j = predictions_list[j]
                profile_i = medium_profiles_list[i]
                profile_j = medium_profiles_list[j]
                profile_diff = (profile_i - profile_j).abs().mean()
                if profile_diff < self.eps:
                    pred_diff = (pred_i - pred_j).abs().mean()
                    total_loss = total_loss + pred_diff
                    n_pairs += 1
        if n_pairs > 0:
            total_loss = total_loss / n_pairs
        return total_loss


class MediumAwareWeighting(nn.Module):
    def __init__(self, num_medium_models=30, eps=1e-8, momentum=0.9):
        super().__init__()
        self.num_medium_models = num_medium_models
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('medium_loss_history',
                             torch.zeros(num_medium_models, dtype=torch.float32))
        self.register_buffer('medium_weights',
                             torch.ones(num_medium_models, dtype=torch.float32) / num_medium_models)
        self._initialized = False

    def update(self, medium_idx, loss_val):
        if medium_idx >= self.num_medium_models:
            return
        if not self._initialized:
            self.medium_loss_history[medium_idx] = loss_val
            self._initialized = True
        else:
            self.medium_loss_history[medium_idx] = (
                self.momentum * self.medium_loss_history[medium_idx] +
                (1.0 - self.momentum) * loss_val
            )
        inv_losses = 1.0 / (self.medium_loss_history + self.eps)
        self.medium_weights.copy_(inv_losses / (inv_losses.sum() + self.eps))

    def get_weight(self, medium_idx):
        if medium_idx >= self.num_medium_models:
            return 1.0
        return self.medium_weights[medium_idx].item()

    def get_weights(self):
        with torch.no_grad():
            return self.medium_weights.tolist()


class CausalTrainingWeightingFNO(nn.Module):
    def __init__(self, num_chunks=10, eps=1e-8, temperature=1.0):
        super().__init__()
        self.num_chunks = num_chunks
        self.eps = eps
        self.temperature = temperature

    def compute_weights(self, R_values, rho_values):
        n = R_values.shape[0]
        chunk_size = max(n // self.num_chunks, 1)
        sorted_idx = torch.argsort(rho_values.squeeze())
        weights = torch.ones(n, 1, device=R_values.device)
        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n)
            if start >= n:
                break
            chunk_indices = sorted_idx[start:end]
            chunk_residuals = R_values[chunk_indices].abs().mean()
            causal_w = torch.exp(-self.temperature * chunk_residuals.detach())
            weights[chunk_indices] = causal_w
        return weights


class PerComponentLossTrackerFNO:
    def __init__(self, component_names=None):
        if component_names is None:
            self.component_names = [
                'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
                'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
                'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
                'Kphi_Re', 'Kphi_Im'
            ]
        else:
            self.component_names = component_names
        self.history = {name: [] for name in self.component_names}
        self.pde_per_component = {name: [] for name in self.component_names}
        self.gauge_per_component = {name: [] for name in self.component_names}
        self.per_medium_history = {}

    def update(self, loss_dict, medium_idx=None):
        for name in self.component_names:
            key = f'pde_{name}'
            if key in loss_dict:
                self.pde_per_component[name].append(loss_dict[key])
            key = f'gauge_{name}'
            if key in loss_dict:
                self.gauge_per_component[name].append(loss_dict[key])
            if name in loss_dict:
                self.history[name].append(loss_dict[name])
        if medium_idx is not None:
            if medium_idx not in self.per_medium_history:
                self.per_medium_history[medium_idx] = []
            self.per_medium_history[medium_idx].append(loss_dict.get('total', 0.0))

    def get_summary(self):
        summary = {}
        for name in self.component_names:
            pde_vals = self.pde_per_component[name]
            gauge_vals = self.gauge_per_component[name]
            if len(pde_vals) > 0:
                summary[f'pde_{name}_last'] = pde_vals[-1]
                summary[f'pde_{name}_mean'] = np.mean(pde_vals[-100:])
            if len(gauge_vals) > 0:
                summary[f'gauge_{name}_last'] = gauge_vals[-1]
                summary[f'gauge_{name}_mean'] = np.mean(gauge_vals[-100:])
        return summary

    def get_worst_components(self, n=3):
        if not any(len(v) > 0 for v in self.pde_per_component.values()):
            return []
        last_losses = {}
        for name in self.component_names:
            vals = self.pde_per_component[name]
            if len(vals) > 0:
                last_losses[name] = vals[-1]
        sorted_components = sorted(last_losses.items(), key=lambda x: x[1], reverse=True)
        return sorted_components[:n]

    def get_per_medium_summary(self):
        summary = {}
        for mid_idx, history in self.per_medium_history.items():
            if len(history) > 0:
                summary[f'medium_{mid_idx}_last'] = history[-1]
                summary[f'medium_{mid_idx}_mean'] = np.mean(history[-100:])
        return summary


class LossDiagnosticsFNO:
    def __init__(self, loss_names=None):
        if loss_names is None:
            self.loss_names = ['pde', 'gauge', 'bc', 'data', 'sym']
        else:
            self.loss_names = loss_names
        self.history = {name: [] for name in self.loss_names}
        self.gradient_norms = {name: [] for name in self.loss_names}
        self.weight_history = {name: [] for name in self.loss_names}
        self.per_medium_history = {}

    def record(self, loss_dict, medium_idx=None):
        for name in self.loss_names:
            if name in loss_dict:
                self.history[name].append(loss_dict[name])
            w_key = f'w_{name}'
            if w_key in loss_dict:
                self.weight_history[name].append(loss_dict[w_key])
        if medium_idx is not None:
            if medium_idx not in self.per_medium_history:
                self.per_medium_history[medium_idx] = {name: [] for name in self.loss_names}
            for name in self.loss_names:
                if name in loss_dict:
                    self.per_medium_history[medium_idx][name].append(loss_dict[name])

    def check_loss_imbalance(self, window=100):
        if len(self.history['pde']) < window:
            return None
        recent = {}
        for name in self.loss_names:
            vals = self.history[name][-window:]
            if len(vals) > 0:
                recent[name] = np.mean(vals)
        if not recent:
            return None
        max_loss = max(recent.values())
        min_loss = min(recent.values())
        if max_loss < 1e-30:
            return None
        ratio = max_loss / (min_loss + 1e-30)
        return {
            'ratio': ratio,
            'is_imbalanced': ratio > 1e4,
            'max_loss_name': max(recent, key=recent.get),
            'min_loss_name': min(recent, key=recent.get),
        }

    def get_convergence_rate(self, name, window=500):
        vals = self.history.get(name, [])
        if len(vals) < window:
            return None
        recent = vals[-window:]
        log_vals = np.log10(np.array(recent) + 1e-30)
        x = np.arange(len(log_vals))
        slope, _ = np.polyfit(x, log_vals, 1)
        return slope

    def get_summary(self):
        summary = {}
        for name in self.loss_names:
            vals = self.history[name]
            if len(vals) > 0:
                summary[f'{name}_last'] = vals[-1]
                summary[f'{name}_min'] = min(vals)
                summary[f'{name}_mean_100'] = np.mean(vals[-100:]) if len(vals) >= 100 else np.mean(vals)
                rate = self.get_convergence_rate(name)
                if rate is not None:
                    summary[f'{name}_convergence_rate'] = rate
        imbalance = self.check_loss_imbalance()
        if imbalance is not None:
            summary['loss_imbalance'] = imbalance
        return summary

    def get_per_medium_summary(self):
        summary = {}
        for mid_idx, history in self.per_medium_history.items():
            for name in self.loss_names:
                vals = history[name]
                if len(vals) > 0:
                    summary[f'medium_{mid_idx}_{name}_last'] = vals[-1]
                    summary[f'medium_{mid_idx}_{name}_mean'] = np.mean(vals[-50:])
        return summary


class FRPINNsFNOLoss(nn.Module):
    def __init__(self, layer_info_dict, frequencies, eps=1e-8,
                 beta1=1.0, beta2=1.0, adaptive=True, rho_range=None,
                 use_complex_km=False, weighting_strategy='ema',
                 ntk_update_interval=200, causal_chunks=10,
                 causal_temperature=1.0, num_medium_models=30,
                 cross_medium_weight=0.01, track_per_component=True):
        super().__init__()
        self.layer_info_dict = layer_info_dict
        self.frequencies = frequencies
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.adaptive = adaptive
        self.rho_range = rho_range or (0.1, 800e3)
        self.use_complex_km = use_complex_km
        self.weighting_strategy = weighting_strategy
        self.num_losses = 5
        self.num_medium_models = num_medium_models
        self.cross_medium_weight = cross_medium_weight
        self.track_per_component = track_per_component

        if self.adaptive:
            if weighting_strategy == 'ntk':
                self.weight_module = NTKAdaptiveWeightsFNO(
                    num_losses=5, update_interval=ntk_update_interval
                )
            else:
                self.weight_module = AdaptiveLossWeightsFNO(num_losses=5)

        self.cross_medium_loss = CrossMediumConsistencyLoss(
            eps=eps, temperature=0.1
        )
        self.medium_aware_weighting = MediumAwareWeighting(
            num_medium_models=num_medium_models
        )
        self.causal_weighting = CausalTrainingWeightingFNO(
            num_chunks=causal_chunks, temperature=causal_temperature
        )
        self.component_tracker = PerComponentLossTrackerFNO()
        self.diagnostics = LossDiagnosticsFNO()

    def _get_layer_index(self, z, layer_info):
        layer_idx = torch.zeros_like(z, dtype=torch.long)
        for i, layer in enumerate(layer_info):
            mask = (z >= layer['z_min']) & (z < layer['z_max'])
            layer_idx[mask] = i
        return layer_idx

    def _get_wavenumber(self, z, layer_info, freq):
        layer_idx = self._get_layer_index(z, layer_info)
        k = torch.zeros_like(z)
        for i, layer in enumerate(layer_info):
            mask = (layer_idx == i)
            mu = layer.get('mu_r', 1.0) * 4.0 * math.pi * 1e-7
            eps_val = layer.get('eps_r', 1.0) * 8.854e-12
            k_val = 2.0 * math.pi * freq * math.sqrt(mu * eps_val)
            k[mask] = k_val
        return k

    def _get_complex_km2(self, z, layer_info, freq):
        omega = 2.0 * math.pi * freq
        layer_idx = self._get_layer_index(z, layer_info)
        km2_re = torch.zeros_like(z)
        km2_im = torch.zeros_like(z)
        for i, layer in enumerate(layer_info):
            mask = (layer_idx == i)
            mu = layer.get('mu_r', 1.0) * 4.0 * math.pi * 1e-7
            eps_val = layer.get('eps_r', 1.0) * 8.854e-12
            sigma = layer.get('sigma', 0.0)
            km2_re[mask] = omega ** 2 * mu * eps_val
            km2_im[mask] = -omega * mu * sigma
        return km2_re, km2_im

    def _compute_laplacian_cylindrical(self, Kc, x_in, rho):
        grad = torch.autograd.grad(Kc.sum(), x_in, create_graph=True)[0]
        dK_dx = grad[:, 0:1]
        dK_dy = grad[:, 1:2]
        dK_dz = grad[:, 2:3]

        d2K_dx2 = torch.autograd.grad(
            dK_dx.sum(), x_in, create_graph=True
        )[0][:, 0:1]
        d2K_dy2 = torch.autograd.grad(
            dK_dy.sum(), x_in, create_graph=True
        )[0][:, 1:2]
        d2K_dz2 = torch.autograd.grad(
            dK_dz.sum(), x_in, create_graph=True
        )[0][:, 2:3]

        return d2K_dx2 + d2K_dy2 + d2K_dz2

    def _compute_laplacian_cartesian(self, Kc, x_in):
        grad = torch.autograd.grad(Kc.sum(), x_in, create_graph=True)[0]
        dK_dx = grad[:, 0:1]
        dK_dy = grad[:, 1:2]
        dK_dz = grad[:, 2:3]

        d2K_dx2 = torch.autograd.grad(
            dK_dx.sum(), x_in, create_graph=True
        )[0][:, 0:1]
        d2K_dy2 = torch.autograd.grad(
            dK_dy.sum(), x_in, create_graph=True
        )[0][:, 1:2]
        d2K_dz2 = torch.autograd.grad(
            dK_dz.sum(), x_in, create_graph=True
        )[0][:, 2:3]

        return d2K_dx2 + d2K_dy2 + d2K_dz2

    def _compute_source_weight(self, x, xp, y, yp, z, zp, freq):
        R = torch.sqrt((x - xp) ** 2 + (y - yp) ** 2 + (z - zp) ** 2)
        wavelength = 3e8 / freq if freq > 0 else 1e10
        R_ref = wavelength / 10.0
        w = torch.min(R ** 2 / (R_ref ** 2 + self.eps), torch.ones_like(R))
        return w, R

    def _compute_sommerfeld_weight(self, rho, freq):
        wavelength = 3e8 / freq if freq > 0 else 1e10
        rho_ref = max(self.rho_range[1], wavelength * 2)
        w = (rho_ref / (rho + self.eps)) ** 2
        w = torch.min(w, torch.ones_like(w) * 10.0)
        return w

    def compute_pde_loss(self, model, x_interior, medium_profile,
                         layer_info, freq_idx, use_causal=False):
        x = x_interior[:, 0:1]
        y = x_interior[:, 1:2]
        z = x_interior[:, 2:3]
        xp = x_interior[:, 3:4]
        yp = x_interior[:, 4:5]
        zp = x_interior[:, 5:6]
        rho = x_interior[:, 6:7]

        x_in = x_interior.clone().detach().requires_grad_(True)
        pred = model(x_in, medium_profile)

        Kxx_Re, Kxx_Im = pred[:, 0:1], pred[:, 1:2]
        Kxz_Re, Kxz_Im = pred[:, 2:3], pred[:, 3:4]
        Kyz_Re, Kyz_Im = pred[:, 4:5], pred[:, 5:6]
        Kzx_Re, Kzx_Im = pred[:, 6:7], pred[:, 7:8]
        Kzy_Re, Kzy_Im = pred[:, 8:9], pred[:, 9:10]
        Kzz_Re, Kzz_Im = pred[:, 10:11], pred[:, 11:12]
        Kphi_Re, Kphi_Im = pred[:, 12:13], pred[:, 13:14]

        freq = self.frequencies[freq_idx]
        km = self._get_wavenumber(z.detach(), layer_info, freq)
        km2 = km ** 2

        w, R = self._compute_source_weight(x, xp, y, yp, z, zp, freq)

        if use_causal:
            causal_w = self.causal_weighting.compute_weights(R, rho)
            w = w * causal_w

        loss_pde = torch.tensor(0.0, device=x_in.device)
        component_losses = OrderedDict()

        s0_names_Re = ['Kxx_Re', 'Kzz_Re', 'Kphi_Re']
        s0_names_Im = ['Kxx_Im', 'Kzz_Im', 'Kphi_Im']
        s1_names_Re = ['Kxz_Re', 'Kyz_Re', 'Kzx_Re', 'Kzy_Re']
        s1_names_Im = ['Kxz_Im', 'Kyz_Im', 'Kzx_Im', 'Kzy_Im']

        s0_components_Re = [Kxx_Re, Kzz_Re, Kphi_Re]
        s0_components_Im = [Kxx_Im, Kzz_Im, Kphi_Im]
        s1_components_Re = [Kxz_Re, Kyz_Re, Kzx_Re, Kzy_Re]
        s1_components_Im = [Kxz_Im, Kyz_Im, Kzx_Im, Kzy_Im]

        if self.use_complex_km:
            km2_re, km2_im = self._get_complex_km2(z.detach(), layer_info, freq)
            for name_Re, name_Im, Kc_Re, Kc_Im in zip(
                s0_names_Re, s0_names_Im,
                s0_components_Re, s0_components_Im
            ):
                laplacian_Re = self._compute_laplacian_cylindrical(Kc_Re, x_in, rho)
                laplacian_Im = self._compute_laplacian_cylindrical(Kc_Im, x_in, rho)
                residual_Re = laplacian_Re + km2_re * Kc_Re - km2_im * Kc_Im
                residual_Im = laplacian_Im + km2_re * Kc_Im + km2_im * Kc_Re
                comp_loss = (
                    w * (residual_Re ** 2 + residual_Im ** 2) / (km2_re.abs() + 1.0)
                ).mean()
                loss_pde = loss_pde + comp_loss
                component_losses[f'pde_{name_Re}'] = comp_loss.item()
                component_losses[f'pde_{name_Im}'] = comp_loss.item()

            for name_Re, name_Im, Kc_Re, Kc_Im in zip(
                s1_names_Re, s1_names_Im,
                s1_components_Re, s1_components_Im
            ):
                laplacian_Re = self._compute_laplacian_cartesian(Kc_Re, x_in)
                laplacian_Im = self._compute_laplacian_cartesian(Kc_Im, x_in)
                residual_Re = laplacian_Re + km2_re * Kc_Re - km2_im * Kc_Im
                residual_Im = laplacian_Im + km2_re * Kc_Im + km2_im * Kc_Re
                comp_loss = (
                    w * (residual_Re ** 2 + residual_Im ** 2) / (km2_re.abs() + 1.0)
                ).mean()
                loss_pde = loss_pde + comp_loss
                component_losses[f'pde_{name_Re}'] = comp_loss.item()
                component_losses[f'pde_{name_Im}'] = comp_loss.item()
        else:
            for name_Re, name_Im, Kc_Re, Kc_Im in zip(
                s0_names_Re, s0_names_Im,
                s0_components_Re, s0_components_Im
            ):
                laplacian_Re = self._compute_laplacian_cylindrical(Kc_Re, x_in, rho)
                residual_Re = laplacian_Re + km2 * Kc_Re

                laplacian_Im = self._compute_laplacian_cylindrical(Kc_Im, x_in, rho)
                residual_Im = laplacian_Im + km2 * Kc_Im

                comp_loss = (
                    w * (residual_Re ** 2 + residual_Im ** 2) / (km2 + 1.0)
                ).mean()
                loss_pde = loss_pde + comp_loss
                component_losses[f'pde_{name_Re}'] = comp_loss.item()
                component_losses[f'pde_{name_Im}'] = comp_loss.item()

            for name_Re, name_Im, Kc_Re, Kc_Im in zip(
                s1_names_Re, s1_names_Im,
                s1_components_Re, s1_components_Im
            ):
                laplacian_Re = self._compute_laplacian_cartesian(Kc_Re, x_in)
                residual_Re = laplacian_Re + km2 * Kc_Re

                laplacian_Im = self._compute_laplacian_cartesian(Kc_Im, x_in)
                residual_Im = laplacian_Im + km2 * Kc_Im

                comp_loss = (
                    w * (residual_Re ** 2 + residual_Im ** 2) / (km2 + 1.0)
                ).mean()
                loss_pde = loss_pde + comp_loss
                component_losses[f'pde_{name_Re}'] = comp_loss.item()
                component_losses[f'pde_{name_Im}'] = comp_loss.item()

        return loss_pde, component_losses

    def compute_gauge_loss(self, model, x_gauge, medium_profile,
                           layer_info, freq_idx):
        x_in = x_gauge.clone().detach().requires_grad_(True)
        pred = model(x_in, medium_profile)

        Kxx_Re, Kxx_Im = pred[:, 0:1], pred[:, 1:2]
        Kxz_Re, Kxz_Im = pred[:, 2:3], pred[:, 3:4]
        Kyz_Re, Kyz_Im = pred[:, 4:5], pred[:, 5:6]
        Kzx_Re, Kzx_Im = pred[:, 6:7], pred[:, 7:8]
        Kzy_Re, Kzy_Im = pred[:, 8:9], pred[:, 9:10]
        Kzz_Re, Kzz_Im = pred[:, 10:11], pred[:, 11:12]
        Kphi_Re, Kphi_Im = pred[:, 12:13], pred[:, 13:14]

        freq = self.frequencies[freq_idx]
        omega = 2.0 * math.pi * freq
        z = x_in[:, 2:3]
        km = self._get_wavenumber(z.detach(), layer_info, freq)
        km2 = km ** 2

        loss_gauge = torch.tensor(0.0, device=x_in.device)
        component_losses = OrderedDict()

        for label, Kxx, Kxz, Kyz, Kzx, Kzy, Kzz, Kphi in [
            ('Re', Kxx_Re, Kxz_Re, Kyz_Re, Kzx_Re, Kzy_Re, Kzz_Re, Kphi_Re),
            ('Im', Kxx_Im, Kxz_Im, Kyz_Im, Kzx_Im, Kzy_Im, Kzz_Im, Kphi_Im),
        ]:
            grad_Kxx = torch.autograd.grad(Kxx.sum(), x_in, create_graph=True)[0]
            dKxx_dx = grad_Kxx[:, 0:1]
            dKxx_dy = grad_Kxx[:, 1:2]

            grad_Kzx = torch.autograd.grad(Kzx.sum(), x_in, create_graph=True)[0]
            dKzx_dz = grad_Kzx[:, 2:3]

            grad_Kphi = torch.autograd.grad(Kphi.sum(), x_in, create_graph=True)[0]
            dKphi_dxp = grad_Kphi[:, 3:4]
            dKphi_dyp = grad_Kphi[:, 4:5]
            dKphi_dzp = grad_Kphi[:, 5:6]

            grad_Kxz = torch.autograd.grad(Kxz.sum(), x_in, create_graph=True)[0]
            dKxz_dx = grad_Kxz[:, 0:1]

            grad_Kyz = torch.autograd.grad(Kyz.sum(), x_in, create_graph=True)[0]
            dKyz_dy = grad_Kyz[:, 1:2]

            grad_Kzz = torch.autograd.grad(Kzz.sum(), x_in, create_graph=True)[0]
            dKzz_dz = grad_Kzz[:, 2:3]

            grad_Kzy = torch.autograd.grad(Kzy.sum(), x_in, create_graph=True)[0]
            dKzy_dz = grad_Kzy[:, 2:3]

            R_gauge_x = (omega / (km2 + self.eps)) * (dKxx_dx + dKzx_dz) \
                + (1.0 / (omega + self.eps)) * dKphi_dxp
            R_gauge_y = (omega / (km2 + self.eps)) * (dKxx_dy + dKzy_dz) \
                + (1.0 / (omega + self.eps)) * dKphi_dyp
            R_gauge_z = (omega / (km2 + self.eps)) * (dKxz_dx + dKyz_dy + dKzz_dz) \
                + (1.0 / (omega + self.eps)) * dKphi_dzp

            loss_x = (R_gauge_x ** 2 / (km2 + 1.0)).mean()
            loss_y = (R_gauge_y ** 2 / (km2 + 1.0)).mean()
            loss_z = (R_gauge_z ** 2 / (km2 + 1.0)).mean()

            loss_gauge = loss_gauge + loss_x + loss_y + loss_z

            component_losses[f'gauge_x_{label}'] = loss_x.item()
            component_losses[f'gauge_y_{label}'] = loss_y.item()
            component_losses[f'gauge_z_{label}'] = loss_z.item()

        return loss_gauge, component_losses

    def compute_bc_loss(self, model, x_interface, x_radiation,
                        medium_profile, layer_info, freq_idx):
        component_losses = OrderedDict()

        if x_interface is not None and x_interface.shape[0] > 0:
            z_vals = x_interface[:, 2:3]
            delta = 1e-4
            x_above = x_interface.clone()
            x_above[:, 2:3] = z_vals + delta
            x_below = x_interface.clone()
            x_below[:, 2:3] = z_vals - delta

            pred_above = model(x_above, medium_profile)
            pred_below = model(x_below, medium_profile)

            loss_cont = ((pred_above - pred_below) ** 2).mean()

            component_names = [
                'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
                'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
                'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
                'Kphi_Re', 'Kphi_Im'
            ]
            for i, name in enumerate(component_names):
                diff = (pred_above[:, i:i+1] - pred_below[:, i:i+1]) ** 2
                component_losses[f'bc_cont_{name}'] = diff.mean().item()

            x_above_data = x_interface.clone().detach()
            x_above_data[:, 2:3] = z_vals + delta
            x_above_grad = x_above_data.requires_grad_(True)
            pred_above_grad = model(x_above_grad, medium_profile)

            x_below_data = x_interface.clone().detach()
            x_below_data[:, 2:3] = z_vals - delta
            x_below_grad = x_below_data.requires_grad_(True)
            pred_below_grad = model(x_below_grad, medium_profile)

            loss_normal_deriv = torch.tensor(0.0, device=x_interface.device)
            for comp_idx in range(pred_above_grad.shape[1]):
                grad_above_i = torch.autograd.grad(
                    pred_above_grad[:, comp_idx].sum(), x_above_grad,
                    create_graph=True, allow_unused=True
                )
                grad_below_i = torch.autograd.grad(
                    pred_below_grad[:, comp_idx].sum(), x_below_grad,
                    create_graph=True, allow_unused=True
                )
                if grad_above_i[0] is not None and grad_below_i[0] is not None:
                    dz_above_i = grad_above_i[0][:, 2:3]
                    dz_below_i = grad_below_i[0][:, 2:3]
                    loss_normal_deriv = loss_normal_deriv + (
                        (dz_above_i - dz_below_i) ** 2
                    ).mean()

            if loss_normal_deriv.item() > 0:
                loss_cont = loss_cont + 0.1 * loss_normal_deriv
                component_losses['bc_normal_deriv'] = loss_normal_deriv.item()
        else:
            loss_cont = torch.tensor(0.0, device=next(model.parameters()).device)

        if x_radiation is not None and x_radiation.shape[0] > 0:
            rho_rad = x_radiation[:, 6:7]
            rho_ref = self.rho_range[1]
            decay_weight = (rho_ref / (rho_rad + self.eps)) ** 2

            pred_rad = model(x_radiation, medium_profile)
            loss_rad = (decay_weight * pred_rad ** 2).mean()

            freq = self.frequencies[freq_idx]
            sommerfeld_w = self._compute_sommerfeld_weight(rho_rad, freq)
            x_rad_grad = x_radiation.clone().detach().requires_grad_(True)
            pred_rad_grad = model(x_rad_grad, medium_profile)
            grad_pred = torch.autograd.grad(
                pred_rad_grad.sum(), x_rad_grad,
                create_graph=True
            )[0]
            d_pred_dx = grad_pred[:, 0:1]
            d_pred_dy = grad_pred[:, 1:2]
            d_pred_drho = (d_pred_dx * x_radiation[:, 0:1] +
                           d_pred_dy * x_radiation[:, 1:2]) / (rho_rad + self.eps)
            km = self._get_wavenumber(x_radiation[:, 2:3].detach(), layer_info, freq)
            sommerfeld_res = d_pred_drho + 1j * km.unsqueeze(-1) * pred_rad_grad
            loss_sommerfeld = (sommerfeld_w * sommerfeld_res.abs() ** 2).mean()
            loss_rad = loss_rad + 0.1 * loss_sommerfeld

            component_losses['bc_radiation_decay'] = loss_rad.item()
            component_losses['bc_sommerfeld'] = loss_sommerfeld.item()
        else:
            loss_rad = torch.tensor(0.0, device=next(model.parameters()).device)

        total_bc = loss_rad + loss_cont
        component_losses['bc_total'] = total_bc.item()
        return total_bc, component_losses

    def compute_data_loss(self, model, x_data, y_data, medium_profile):
        pred = model(x_data, medium_profile)
        Kc_ref = y_data.abs().max(dim=0, keepdim=True)[0]
        Kc_ref_sq = Kc_ref ** 2 + self.eps
        loss = ((pred - y_data) ** 2 / Kc_ref_sq).mean()

        component_losses = OrderedDict()
        component_names = [
            'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
            'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
            'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
            'Kphi_Re', 'Kphi_Im'
        ]
        for i, name in enumerate(component_names):
            diff_sq = (pred[:, i:i+1] - y_data[:, i:i+1]) ** 2
            ref_sq = y_data[:, i:i+1].abs().max() ** 2 + self.eps
            component_losses[f'data_{name}'] = (diff_sq / ref_sq).mean().item()

        rel_error = (pred - y_data).abs() / (y_data.abs() + self.eps)
        component_losses['data_rel_error_mean'] = rel_error.mean().item()
        component_losses['data_rel_error_max'] = rel_error.max().item()

        return loss, component_losses

    def compute_symmetry_loss(self, model, x_sym, medium_profile):
        x_in = x_sym.clone().detach().requires_grad_(True)
        pred = model(x_in, medium_profile)

        Kxx_Re, Kxx_Im = pred[:, 0:1], pred[:, 1:2]
        Kxz_Re, Kxz_Im = pred[:, 2:3], pred[:, 3:4]
        Kyz_Re, Kyz_Im = pred[:, 4:5], pred[:, 5:6]
        Kzx_Re, Kzx_Im = pred[:, 6:7], pred[:, 7:8]
        Kzy_Re, Kzy_Im = pred[:, 8:9], pred[:, 9:10]

        dx = x_in[:, 0:1] - x_in[:, 3:4]
        dy = x_in[:, 1:2] - x_in[:, 4:5]

        loss_azi = (
            (Kyz_Re * dx - Kxz_Re * dy) ** 2 +
            (Kyz_Im * dx - Kxz_Im * dy) ** 2 +
            (Kzy_Re * dx - Kzx_Re * dy) ** 2 +
            (Kzy_Im * dx - Kzx_Im * dy) ** 2
        ).mean()

        loss_rot = torch.tensor(0.0, device=x_in.device)
        theta = torch.rand(x_in.shape[0], 1, device=x_in.device) * 2.0 * math.pi
        rho = x_in[:, 6:7]
        x_rot = x_in.detach().clone()
        x_rot[:, 0:1] = x_in[:, 3:4].detach() + rho.detach() * torch.cos(theta)
        x_rot[:, 1:2] = x_in[:, 4:5].detach() + rho.detach() * torch.sin(theta)

        pred_rot = model(x_rot, medium_profile)
        rot_indices_Re = [0, 10, 12]
        rot_indices_Im = [1, 11, 13]
        for idx in rot_indices_Re:
            loss_rot = loss_rot + ((pred[:, idx:idx+1] - pred_rot[:, idx:idx+1]) ** 2).mean()
        for idx in rot_indices_Im:
            loss_rot = loss_rot + ((pred[:, idx:idx+1] - pred_rot[:, idx:idx+1]) ** 2).mean()

        loss_reci = torch.tensor(0.0, device=x_in.device)
        x_swap = x_in.detach().clone()
        x_swap[:, 0:1] = x_in[:, 3:4].detach()
        x_swap[:, 1:2] = x_in[:, 4:5].detach()
        x_swap[:, 2:3] = x_in[:, 5:6].detach()
        x_swap[:, 3:4] = x_in[:, 0:1].detach()
        x_swap[:, 4:5] = x_in[:, 1:2].detach()
        x_swap[:, 5:6] = x_in[:, 2:3].detach()
        pred_swap = model(x_swap, medium_profile)

        loss_reci = (
            (pred[:, 2:3] - pred_swap[:, 6:7]) ** 2 +
            (pred[:, 3:4] - pred_swap[:, 7:8]) ** 2 +
            (pred[:, 4:5] - pred_swap[:, 8:9]) ** 2 +
            (pred[:, 5:6] - pred_swap[:, 9:10]) ** 2
        ).mean()

        loss_sym = loss_azi + self.beta1 * loss_rot + self.beta2 * loss_reci

        component_losses = OrderedDict()
        component_losses['sym_azi'] = loss_azi.item()
        component_losses['sym_rot'] = loss_rot.item()
        component_losses['sym_reci'] = loss_reci.item()

        return loss_sym, component_losses

    def compute_cross_medium_loss(self, model, batch_list,
                                  medium_profiles_list, layer_info_list,
                                  freq_idx):
        if len(batch_list) < 2:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        predictions_list = []
        profiles_for_comparison = []
        for batch, mp, li in zip(batch_list, medium_profiles_list, layer_info_list):
            x_interior = batch['interior']
            n_sample = min(100, x_interior.shape[0])
            x_sample = x_interior[:n_sample]
            with torch.no_grad():
                pred = model(x_sample, mp)
            predictions_list.append(pred)
            profiles_for_comparison.append(mp)

        cross_loss = self.cross_medium_loss(predictions_list, profiles_for_comparison)
        return cross_loss

    def compute_multi_medium_aggregated_loss(self, model, batches_dict,
                                              medium_profiles_dict,
                                              layer_info_dict,
                                              freq_idx, medium_indices):
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        all_component_losses = OrderedDict()
        medium_losses = {}

        for mid_idx in medium_indices:
            batch = batches_dict.get(mid_idx, None)
            mp = medium_profiles_dict.get(mid_idx, None)
            li = layer_info_dict.get(mid_idx, None)

            if batch is None or mp is None or li is None:
                continue

            loss_pde, pde_comp = self.compute_pde_loss(
                model, batch['interior'], mp, li, freq_idx
            )
            loss_gauge, gauge_comp = self.compute_gauge_loss(
                model, batch['gauge'], mp, li, freq_idx
            )
            loss_bc, bc_comp = self.compute_bc_loss(
                model, batch.get('interface', None),
                batch.get('radiation', None),
                mp, li, freq_idx
            )

            loss_data = torch.tensor(0.0, device=total_loss.device)
            data_comp = OrderedDict()
            if batch.get('data_x') is not None and batch.get('data_y') is not None:
                loss_data, data_comp = self.compute_data_loss(
                    model, batch['data_x'], batch['data_y'], mp
                )

            loss_sym = torch.tensor(0.0, device=total_loss.device)
            sym_comp = OrderedDict()
            if batch.get('symmetry') is not None:
                loss_sym, sym_comp = self.compute_symmetry_loss(
                    model, batch['symmetry'], mp
                )

            medium_total = loss_pde + loss_gauge + loss_bc + loss_data + loss_sym
            medium_weight = self.medium_aware_weighting.get_weight(mid_idx)
            weighted_loss = medium_weight * medium_total
            total_loss = total_loss + weighted_loss

            medium_losses[mid_idx] = medium_total.item()
            self.medium_aware_weighting.update(mid_idx, medium_total.item())

            for k, v in pde_comp.items():
                all_component_losses[f'm{mid_idx}_{k}'] = v
            for k, v in gauge_comp.items():
                all_component_losses[f'm{mid_idx}_{k}'] = v
            for k, v in bc_comp.items():
                all_component_losses[f'm{mid_idx}_{k}'] = v
            for k, v in data_comp.items():
                all_component_losses[f'm{mid_idx}_{k}'] = v
            for k, v in sym_comp.items():
                all_component_losses[f'm{mid_idx}_{k}'] = v

        if len(medium_indices) > 1:
            batch_list = [batches_dict[mid] for mid in medium_indices
                          if mid in batches_dict]
            mp_list = [medium_profiles_dict[mid] for mid in medium_indices
                       if mid in medium_profiles_dict]
            li_list = [layer_info_dict[mid] for mid in medium_indices
                       if mid in layer_info_dict]
            cross_loss = self.compute_cross_medium_loss(
                model, batch_list, mp_list, li_list, freq_idx
            )
            total_loss = total_loss + self.cross_medium_weight * cross_loss
            all_component_losses['cross_medium'] = cross_loss.item()

        total_loss = total_loss / len(medium_indices)
        return total_loss, all_component_losses, medium_losses

    def forward(self, model, batch, medium_profile, layer_info, freq_idx,
                medium_idx=None, epoch=0):
        x_interior = batch['interior']
        x_gauge = batch['gauge']
        x_interface = batch.get('interface', None)
        x_radiation = batch.get('radiation', None)
        x_data = batch.get('data_x', None)
        y_data = batch.get('data_y', None)
        x_sym = batch.get('symmetry', None)

        use_causal = epoch > 2000

        loss_pde, pde_comp = self.compute_pde_loss(
            model, x_interior, medium_profile, layer_info, freq_idx,
            use_causal=use_causal
        )
        loss_gauge, gauge_comp = self.compute_gauge_loss(
            model, x_gauge, medium_profile, layer_info, freq_idx
        )
        loss_bc, bc_comp = self.compute_bc_loss(
            model, x_interface, x_radiation,
            medium_profile, layer_info, freq_idx
        )

        loss_data = torch.tensor(0.0, device=loss_pde.device)
        data_comp = OrderedDict()
        if x_data is not None and y_data is not None:
            loss_data, data_comp = self.compute_data_loss(
                model, x_data, y_data, medium_profile
            )

        loss_sym = torch.tensor(0.0, device=loss_pde.device)
        sym_comp = OrderedDict()
        if x_sym is not None:
            loss_sym, sym_comp = self.compute_symmetry_loss(
                model, x_sym, medium_profile
            )

        losses = [loss_pde, loss_gauge, loss_bc, loss_data, loss_sym]

        if self.adaptive:
            weights = self.weight_module()
            normalized_losses = self.weight_module.normalize_losses(losses)
            total_loss = sum(w * nl for w, nl in zip(weights, normalized_losses))
        else:
            total_loss = sum(losses)

        if medium_idx is not None:
            medium_w = self.medium_aware_weighting.get_weight(medium_idx)
            total_loss = medium_w * total_loss
            self.medium_aware_weighting.update(medium_idx, total_loss.item())

        loss_dict = {
            'pde': loss_pde.item(),
            'gauge': loss_gauge.item(),
            'bc': loss_bc.item(),
            'data': loss_data.item(),
            'sym': loss_sym.item(),
            'total': total_loss.item()
        }

        if self.adaptive:
            w_vals = self.weight_module.get_weights()
            loss_dict['w_pde'] = w_vals[0]
            loss_dict['w_gauge'] = w_vals[1]
            loss_dict['w_bc'] = w_vals[2]
            loss_dict['w_data'] = w_vals[3]
            loss_dict['w_sym'] = w_vals[4]

        all_comp = OrderedDict()
        all_comp.update(pde_comp)
        all_comp.update(gauge_comp)
        all_comp.update(bc_comp)
        all_comp.update(data_comp)
        all_comp.update(sym_comp)
        loss_dict.update(all_comp)

        if self.track_per_component:
            self.component_tracker.update(loss_dict, medium_idx=medium_idx)

        self.diagnostics.record(loss_dict, medium_idx=medium_idx)

        if self.weighting_strategy == 'ntk' and self.adaptive:
            loss_inputs = [x_interior, x_gauge, x_interface, x_data, x_sym]
            mp_list = [medium_profile] * 5
            self.weight_module.update_weights(model, loss_inputs, mp_list)

        return total_loss, loss_dict

    def get_diagnostics_summary(self):
        summary = self.diagnostics.get_summary()
        if self.track_per_component:
            summary['component_summary'] = self.component_tracker.get_summary()
            summary['worst_components'] = self.component_tracker.get_worst_components()
            summary['per_medium_summary'] = self.component_tracker.get_per_medium_summary()
        summary['per_medium_diagnostics'] = self.diagnostics.get_per_medium_summary()
        summary['medium_weights'] = self.medium_aware_weighting.get_weights()
        return summary
