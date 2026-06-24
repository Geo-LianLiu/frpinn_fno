import os
import math
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Union


class RelativeErrorMetrics:
    def __init__(self, eps=1e-30):
        self.eps = eps

    def relative_l2_error(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        numerator = torch.norm(pred - target, p=2)
        denominator = torch.norm(target, p=2) + self.eps
        return (numerator / denominator).item()

    def relative_l1_error(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        numerator = torch.norm(pred - target, p=1)
        denominator = torch.norm(target, p=1) + self.eps
        return (numerator / denominator).item()

    def relative_linf_error(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        numerator = torch.max(torch.abs(pred - target))
        denominator = torch.max(torch.abs(target)) + self.eps
        return (numerator / denominator).item()

    def mean_absolute_error(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        return torch.mean(torch.abs(pred - target)).item()

    def root_mean_squared_error(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()

    def mean_squared_error(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        return torch.mean((pred - target) ** 2).item()

    def normalized_rmse(self, pred, target):
        rmse = self.root_mean_squared_error(pred, target)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        target_range = target.max() - target.min() + self.eps
        return rmse / target_range.item()

    def r_squared(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2) + self.eps
        return (1.0 - ss_res / ss_tot).item()

    def all_metrics(self, pred, target):
        return OrderedDict([
            ('rel_l2', self.relative_l2_error(pred, target)),
            ('rel_l1', self.relative_l1_error(pred, target)),
            ('rel_linf', self.relative_linf_error(pred, target)),
            ('mae', self.mean_absolute_error(pred, target)),
            ('rmse', self.root_mean_squared_error(pred, target)),
            ('mse', self.mean_squared_error(pred, target)),
            ('nrmse', self.normalized_rmse(pred, target)),
            ('r2', self.r_squared(pred, target)),
        ])


class ComponentWiseEvaluator:
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
        self.metrics = RelativeErrorMetrics()

    def evaluate_components(self, pred, target):
        results = OrderedDict()
        for i, name in enumerate(self.component_names):
            pred_i = pred[:, i]
            target_i = target[:, i]
            results[name] = self.metrics.all_metrics(pred_i, target_i)
        return results

    def get_worst_components(self, results, metric='rel_l2', n=3):
        component_scores = []
        for name, metrics in results.items():
            component_scores.append((name, metrics.get(metric, float('inf'))))
        component_scores.sort(key=lambda x: x[1], reverse=True)
        return component_scores[:n]

    def get_best_components(self, results, metric='rel_l2', n=3):
        component_scores = []
        for name, metrics in results.items():
            component_scores.append((name, metrics.get(metric, 0.0)))
        component_scores.sort(key=lambda x: x[1])
        return component_scores[:n]

    def get_average_metrics(self, results, metric='rel_l2'):
        values = [metrics[metric] for metrics in results.values()]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }

    def get_symmetry_group_metrics(self, results):
        s0_names = ['Kxx_Re', 'Kxx_Im', 'Kzz_Re', 'Kzz_Im', 'Kphi_Re', 'Kphi_Im']
        s1_names = ['Kxz_Re', 'Kxz_Im', 'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
                     'Kzy_Re', 'Kzy_Im']
        real_names = [n for n in self.component_names if 'Re' in n]
        imag_names = [n for n in self.component_names if 'Im' in n]

        group_metrics = OrderedDict()

        s0_l2 = [results[n]['rel_l2'] for n in s0_names if n in results]
        if s0_l2:
            group_metrics['S0_group'] = {
                'mean_rel_l2': np.mean(s0_l2),
                'max_rel_l2': np.max(s0_l2),
                'min_rel_l2': np.min(s0_l2),
            }

        s1_l2 = [results[n]['rel_l2'] for n in s1_names if n in results]
        if s1_l2:
            group_metrics['S1_group'] = {
                'mean_rel_l2': np.mean(s1_l2),
                'max_rel_l2': np.max(s1_l2),
                'min_rel_l2': np.min(s1_l2),
            }

        real_l2 = [results[n]['rel_l2'] for n in real_names if n in results]
        if real_l2:
            group_metrics['Real_parts'] = {
                'mean_rel_l2': np.mean(real_l2),
                'max_rel_l2': np.max(real_l2),
                'min_rel_l2': np.min(real_l2),
            }

        imag_l2 = [results[n]['rel_l2'] for n in imag_names if n in results]
        if imag_l2:
            group_metrics['Imaginary_parts'] = {
                'mean_rel_l2': np.mean(imag_l2),
                'max_rel_l2': np.max(imag_l2),
                'min_rel_l2': np.min(imag_l2),
            }

        return group_metrics


class FrequencyDependentEvaluator:
    def __init__(self, frequencies, component_names=None):
        self.frequencies = frequencies
        self.component_evaluator = ComponentWiseEvaluator(component_names)

    def evaluate_across_frequencies(self, model, data_dict, device='cpu'):
        results = OrderedDict()
        for freq_idx, freq in enumerate(self.frequencies):
            if freq_idx not in data_dict:
                continue
            data = data_dict[freq_idx]
            x = data['x'].to(device)
            target = data['y'].to(device)

            model.eval()
            with torch.no_grad():
                pred = model(x)

            if hasattr(model, 'fno_encoder'):
                pass

            comp_results = self.component_evaluator.evaluate_components(pred, target)
            results[freq] = {
                'freq_idx': freq_idx,
                'freq_hz': freq,
                'component_metrics': comp_results,
                'overall_rel_l2': RelativeErrorMetrics().relative_l2_error(pred, target),
            }
        return results

    def get_frequency_trend(self, results, metric='rel_l2'):
        freqs = []
        values = []
        for freq, res in results.items():
            freqs.append(res['freq_hz'])
            values.append(res['overall_rel_l2'])
        return np.array(freqs), np.array(values)

    def get_component_frequency_trend(self, results, component_name, metric='rel_l2'):
        freqs = []
        values = []
        for freq, res in results.items():
            if component_name in res['component_metrics']:
                freqs.append(res['freq_hz'])
                values.append(res['component_metrics'][component_name][metric])
        return np.array(freqs), np.array(values)

    def find_worst_frequency(self, results):
        worst_freq = None
        worst_error = -1.0
        for freq, res in results.items():
            if res['overall_rel_l2'] > worst_error:
                worst_error = res['overall_rel_l2']
                worst_freq = res['freq_hz']
        return worst_freq, worst_error

    def find_best_frequency(self, results):
        best_freq = None
        best_error = float('inf')
        for freq, res in results.items():
            if res['overall_rel_l2'] < best_error:
                best_error = res['overall_rel_l2']
                best_freq = res['freq_hz']
        return best_freq, best_error


class SpatialErrorAnalyzer:
    def __init__(self, eps=1e-30):
        self.eps = eps

    def compute_spatial_error_field(self, pred, target, coordinates):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        abs_error = torch.abs(pred - target)
        rel_error = abs_error / (torch.abs(target) + self.eps)
        return {
            'coordinates': coordinates,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'mean_abs_error': abs_error.mean(dim=0),
            'mean_rel_error': rel_error.mean(dim=0),
            'max_abs_error': abs_error.max(dim=0)[0],
            'max_rel_error': rel_error.max(dim=0)[0],
        }

    def compute_radial_error_profile(self, pred, target, rho_values, n_bins=20):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        if isinstance(rho_values, np.ndarray):
            rho_values = torch.tensor(rho_values, dtype=torch.float32)

        abs_error = torch.abs(pred - target)
        rel_error = abs_error / (torch.abs(target) + self.eps)

        rho_flat = rho_values.flatten()
        rho_min = rho_flat.min().item()
        rho_max = rho_flat.max().item()
        bin_edges = np.linspace(rho_min, rho_max, n_bins + 1)

        radial_profile = OrderedDict()
        for i in range(n_bins):
            mask = (rho_flat >= bin_edges[i]) & (rho_flat < bin_edges[i + 1])
            if mask.sum() > 0:
                radial_profile[f'bin_{i}'] = {
                    'rho_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
                    'rho_range': (bin_edges[i], bin_edges[i + 1]),
                    'n_points': mask.sum().item(),
                    'mean_abs_error': abs_error[mask].mean().item(),
                    'mean_rel_error': rel_error[mask].mean().item(),
                    'max_abs_error': abs_error[mask].max().item(),
                    'max_rel_error': rel_error[mask].max().item(),
                }
        return radial_profile

    def compute_depth_error_profile(self, pred, target, z_values, n_bins=20):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        if isinstance(z_values, np.ndarray):
            z_values = torch.tensor(z_values, dtype=torch.float32)

        abs_error = torch.abs(pred - target)
        rel_error = abs_error / (torch.abs(target) + self.eps)

        z_flat = z_values.flatten()
        z_min = z_flat.min().item()
        z_max = z_flat.max().item()
        bin_edges = np.linspace(z_min, z_max, n_bins + 1)

        depth_profile = OrderedDict()
        for i in range(n_bins):
            mask = (z_flat >= bin_edges[i]) & (z_flat < bin_edges[i + 1])
            if mask.sum() > 0:
                depth_profile[f'bin_{i}'] = {
                    'z_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
                    'z_range': (bin_edges[i], bin_edges[i + 1]),
                    'n_points': mask.sum().item(),
                    'mean_abs_error': abs_error[mask].mean().item(),
                    'mean_rel_error': rel_error[mask].mean().item(),
                    'max_abs_error': abs_error[mask].max().item(),
                    'max_rel_error': rel_error[mask].max().item(),
                }
        return depth_profile

    def compute_interface_error(self, pred, target, z_values, interface_depths,
                                tolerance=0.5):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)
        if isinstance(z_values, np.ndarray):
            z_values = torch.tensor(z_values, dtype=torch.float32)

        abs_error = torch.abs(pred - target)
        rel_error = abs_error / (torch.abs(target) + self.eps)

        interface_errors = OrderedDict()
        z_flat = z_values.flatten()
        for depth in interface_depths:
            mask = torch.abs(z_flat - depth) < tolerance
            if mask.sum() > 0:
                interface_errors[f'interface_z={depth:.1f}'] = {
                    'depth': depth,
                    'n_points': mask.sum().item(),
                    'mean_abs_error': abs_error[mask].mean().item(),
                    'mean_rel_error': rel_error[mask].mean().item(),
                    'max_abs_error': abs_error[mask].max().item(),
                    'max_rel_error': rel_error[mask].max().item(),
                }
        return interface_errors

    def compute_near_source_error(self, pred, target, x, xp, y, yp, z, zp,
                                  radius=1.0):
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)

        R = torch.sqrt((x - xp) ** 2 + (y - yp) ** 2 + (z - zp) ** 2)
        abs_error = torch.abs(pred - target)
        rel_error = abs_error / (torch.abs(target) + self.eps)

        near_mask = R.flatten() < radius
        far_mask = R.flatten() >= radius

        result = OrderedDict()
        if near_mask.sum() > 0:
            result['near_source'] = {
                'radius': radius,
                'n_points': near_mask.sum().item(),
                'mean_abs_error': abs_error[near_mask].mean().item(),
                'mean_rel_error': rel_error[near_mask].mean().item(),
            }
        if far_mask.sum() > 0:
            result['far_source'] = {
                'n_points': far_mask.sum().item(),
                'mean_abs_error': abs_error[far_mask].mean().item(),
                'mean_rel_error': rel_error[far_mask].mean().item(),
            }
        return result


class AnalyticalSolutionComparator:
    def __init__(self, eps=1e-30):
        self.eps = eps

    def homogeneous_green_function(self, R, freq, eps_r=1.0, mu_r=1.0):
        mu = mu_r * 4.0 * math.pi * 1e-7
        eps_val = eps_r * 8.854e-12
        k = 2.0 * math.pi * freq * math.sqrt(mu * eps_val)
        R_safe = np.maximum(R, self.eps)
        G = np.exp(-1j * k * R_safe) / (4.0 * math.pi * R_safe)
        return G

    def layered_medium_approximation(self, R, z, z_src, freq, layer_info):
        omega = 2.0 * math.pi * freq
        mu0 = 4.0 * math.pi * 1e-7
        eps0 = 8.854e-12

        layer_idx = 0
        for i, layer in enumerate(layer_info):
            if layer['z_min'] <= z < layer['z_max']:
                layer_idx = i
                break

        layer = layer_info[layer_idx]
        mu = layer.get('mu_r', 1.0) * mu0
        eps_val = layer.get('eps_r', 1.0) * eps0
        sigma = layer.get('sigma', 0.0)

        k_real = omega * math.sqrt(mu * eps_val)
        k_imag = 0.5 * sigma * math.sqrt(mu / (eps_val + self.eps))

        R_safe = np.maximum(R, self.eps)
        attenuation = np.exp(-k_imag * R_safe)
        G = attenuation * np.exp(-1j * k_real * R_safe) / (4.0 * math.pi * R_safe)
        return G

    def sommerfeld_integral_approximation(self, rho, z, z_src, freq, layer_info,
                                          n_quad=100, lambda_max=10.0):
        omega = 2.0 * math.pi * freq
        mu0 = 4.0 * math.pi * 1e-7
        eps0 = 8.854e-12

        lambda_vals = np.linspace(1e-6, lambda_max, n_quad)
        d_lambda = lambda_vals[1] - lambda_vals[0]

        integrand = np.zeros(n_quad, dtype=complex)
        for i, lam in enumerate(lambda_vals):
            k_rho = lam
            for layer in layer_info:
                mu = layer.get('mu_r', 1.0) * mu0
                eps_val = layer.get('eps_r', 1.0) * eps0
                sigma = layer.get('sigma', 0.0)
                kz = np.sqrt(omega ** 2 * mu * eps_val - k_rho ** 2 + 0j)
                if kz.imag < 0:
                    kz = -kz

            bessel_j0 = j0(k_rho * rho) if rho > 0 else 1.0
            integrand[i] = bessel_j0 * np.exp(-1j * kz * abs(z - z_src))

        integral = np.trapz(integrand, lambda_vals)
        return integral / (2.0 * math.pi)

    def compare_with_analytical(self, pred, coordinates, freq, layer_info,
                                method='homogeneous'):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        xp = coordinates[:, 3]
        yp = coordinates[:, 4]
        zp = coordinates[:, 5]

        R = np.sqrt((x - xp) ** 2 + (y - yp) ** 2 + (z - zp) ** 2)

        if method == 'homogeneous':
            eps_r = layer_info[0].get('eps_r', 1.0) if len(layer_info) > 0 else 1.0
            mu_r = layer_info[0].get('mu_r', 1.0) if len(layer_info) > 0 else 1.0
            analytical = self.homogeneous_green_function(R, freq, eps_r, mu_r)
        elif method == 'layered':
            analytical = self.layered_medium_approximation(R, z, zp, freq, layer_info)
        else:
            analytical = self.homogeneous_green_function(R, freq)

        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = pred

        pred_complex = pred_np[:, 0] + 1j * pred_np[:, 1]
        analytical_magnitude = np.abs(analytical)
        pred_magnitude = np.abs(pred_complex)

        rel_error = np.abs(pred_magnitude - analytical_magnitude) / (
            analytical_magnitude + self.eps
        )

        return OrderedDict([
            ('analytical', analytical),
            ('analytical_magnitude', analytical_magnitude),
            ('pred_magnitude', pred_magnitude),
            ('mean_rel_error', np.mean(rel_error)),
            ('max_rel_error', np.max(rel_error)),
            ('median_rel_error', np.median(rel_error)),
        ])


class ConvergenceAnalyzer:
    def __init__(self, window_sizes=None):
        if window_sizes is None:
            self.window_sizes = [100, 500, 1000, 2000]
        else:
            self.window_sizes = window_sizes

    def compute_convergence_rate(self, loss_history, loss_name='total'):
        values = [h.get(loss_name, 0.0) for h in loss_history]
        if len(values) < 10:
            return None

        log_values = np.log10(np.array(values) + 1e-30)
        epochs = np.arange(len(log_values))

        rates = OrderedDict()
        for window in self.window_sizes:
            if len(values) >= window:
                recent = log_values[-window:]
                x = np.arange(len(recent))
                slope, intercept = np.polyfit(x, recent, 1)
                rates[f'window_{window}'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'epochs_to_converge': max(0, -intercept / (slope + 1e-30)) if slope < 0 else None,
                }
        return rates

    def detect_plateau(self, loss_history, loss_name='total', patience=500,
                       threshold=1e-6):
        values = [h.get(loss_name, 0.0) for h in loss_history]
        if len(values) < patience:
            return None

        plateau_start = None
        for i in range(patience, len(values)):
            window = values[i - patience:i]
            if max(window) - min(window) < threshold:
                plateau_start = i - patience
                break

        if plateau_start is not None:
            return {
                'plateau_detected': True,
                'plateau_start_epoch': plateau_start,
                'plateau_value': np.mean(values[plateau_start:plateau_start + patience]),
                'plateau_range': max(values[plateau_start:]) - min(values[plateau_start:]),
            }
        return {'plateau_detected': False}

    def compute_loss_component_ratios(self, loss_history):
        if len(loss_history) < 10:
            return None

        loss_names = ['pde', 'gauge', 'bc', 'data', 'sym']
        ratios = OrderedDict()

        for name in loss_names:
            values = [h.get(name, 0.0) for h in loss_history]
            if len(values) > 0:
                ratios[name] = {
                    'initial': values[0] if values[0] > 0 else None,
                    'final': values[-1],
                    'reduction_ratio': values[0] / (values[-1] + 1e-30) if values[0] > 0 else None,
                    'min_value': min(values),
                    'min_epoch': values.index(min(values)),
                }
        return ratios

    def compute_ema_convergence(self, loss_history, loss_name='total', alpha=0.01):
        values = [h.get(loss_name, 0.0) for h in loss_history]
        if len(values) < 2:
            return None

        ema = [values[0]]
        for v in values[1:]:
            ema.append(alpha * v + (1.0 - alpha) * ema[-1])

        ema_change = [abs(ema[i] - ema[i - 1]) for i in range(1, len(ema))]
        convergence_epoch = None
        threshold = 1e-8
        for i, change in enumerate(ema_change):
            if change < threshold and i > 100:
                convergence_epoch = i
                break

        return {
            'ema_values': ema,
            'ema_changes': ema_change,
            'convergence_epoch': convergence_epoch,
            'final_ema': ema[-1],
            'final_change': ema_change[-1] if ema_change else None,
        }


class PDEResidualEvaluator:
    def __init__(self, layer_info, frequencies, eps=1e-30):
        self.layer_info = layer_info
        self.frequencies = frequencies
        self.eps = eps

    def compute_pde_residuals(self, model, x_interior, freq_idx, device='cpu'):
        model.eval()
        x_in = x_interior.clone().detach().requires_grad_(True).to(device)
        pred = model(x_in)

        residuals = OrderedDict()
        component_names = [
            'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
            'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
            'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
            'Kphi_Re', 'Kphi_Im'
        ]

        freq = self.frequencies[freq_idx]
        z = x_in[:, 2:3].detach()

        for i, name in enumerate(component_names):
            Kc = pred[:, i:i + 1]
            grad = torch.autograd.grad(Kc.sum(), x_in, create_graph=True)[0]
            d2K_dx2 = torch.autograd.grad(
                grad[:, 0:1].sum(), x_in, create_graph=True
            )[0][:, 0:1]
            d2K_dy2 = torch.autograd.grad(
                grad[:, 1:2].sum(), x_in, create_graph=True
            )[0][:, 1:2]
            d2K_dz2 = torch.autograd.grad(
                grad[:, 2:3].sum(), x_in, create_graph=True
            )[0][:, 2:3]

            laplacian = d2K_dx2 + d2K_dy2 + d2K_dz2

            omega = 2.0 * math.pi * freq
            mu0 = 4.0 * math.pi * 1e-7
            eps0 = 8.854e-12

            layer_idx_tensor = torch.zeros_like(z, dtype=torch.long)
            for li, layer in enumerate(self.layer_info):
                mask = (z >= layer['z_min']) & (z < layer['z_max'])
                layer_idx_tensor[mask] = li

            km2 = torch.zeros_like(z)
            for li, layer in enumerate(self.layer_info):
                mask = (layer_idx_tensor == li)
                mu = layer.get('mu_r', 1.0) * mu0
                eps_val = layer.get('eps_r', 1.0) * eps0
                km2[mask] = omega ** 2 * mu * eps_val

            residual = laplacian + km2 * Kc
            residuals[name] = {
                'mean': residual.abs().mean().item(),
                'max': residual.abs().max().item(),
                'std': residual.abs().std().item(),
                'l2_norm': residual.norm().item(),
            }

        return residuals

    def compute_total_pde_residual(self, residuals):
        total = 0.0
        for name, res in residuals.items():
            total += res['l2_norm'] ** 2
        return math.sqrt(total)

    def get_residual_summary(self, residuals):
        means = [res['mean'] for res in residuals.values()]
        maxs = [res['max'] for res in residuals.values()]
        return {
            'mean_of_means': np.mean(means),
            'max_of_maxs': np.max(maxs),
            'total_l2': self.compute_total_pde_residual(residuals),
            'worst_component': max(residuals.keys(), key=lambda k: residuals[k]['mean']),
            'best_component': min(residuals.keys(), key=lambda k: residuals[k]['mean']),
        }


class GaugeConditionEvaluator:
    def __init__(self, eps=1e-30):
        self.eps = eps

    def compute_gauge_residuals(self, model, x_points, freq_idx, frequencies,
                                layer_info, device='cpu'):
        model.eval()
        x_in = x_points.clone().detach().requires_grad_(True).to(device)
        pred = model(x_in)

        Kxx_Re, Kxx_Im = pred[:, 0:1], pred[:, 1:2]
        Kxz_Re, Kxz_Im = pred[:, 2:3], pred[:, 3:4]
        Kyz_Re, Kyz_Im = pred[:, 4:5], pred[:, 5:6]
        Kzx_Re, Kzx_Im = pred[:, 6:7], pred[:, 7:8]
        Kzy_Re, Kzy_Im = pred[:, 8:9], pred[:, 9:10]
        Kzz_Re, Kzz_Im = pred[:, 10:11], pred[:, 11:12]
        Kphi_Re, Kphi_Im = pred[:, 12:13], pred[:, 13:14]

        freq = frequencies[freq_idx]
        omega = 2.0 * math.pi * freq

        grad_Kxx_Re = torch.autograd.grad(Kxx_Re.sum(), x_in, create_graph=True)[0]
        grad_Kzx_Re = torch.autograd.grad(Kzx_Re.sum(), x_in, create_graph=True)[0]
        grad_Kzy_Re = torch.autograd.grad(Kzy_Re.sum(), x_in, create_graph=True)[0]
        grad_Kzz_Re = torch.autograd.grad(Kzz_Re.sum(), x_in, create_graph=True)[0]

        grad_Kxx_Im = torch.autograd.grad(Kxx_Im.sum(), x_in, create_graph=True)[0]
        grad_Kzx_Im = torch.autograd.grad(Kzx_Im.sum(), x_in, create_graph=True)[0]
        grad_Kzy_Im = torch.autograd.grad(Kzy_Im.sum(), x_in, create_graph=True)[0]
        grad_Kzz_Im = torch.autograd.grad(Kzz_Im.sum(), x_in, create_graph=True)[0]

        div_S0_Re = (grad_Kxx_Re[:, 0:1] + grad_Kzx_Re[:, 2:3] +
                     grad_Kzy_Re[:, 1:2] + grad_Kzz_Re[:, 2:3])
        div_S0_Im = (grad_Kxx_Im[:, 0:1] + grad_Kzx_Im[:, 2:3] +
                     grad_Kzy_Im[:, 1:2] + grad_Kzz_Im[:, 2:3])

        z = x_in[:, 2:3].detach()
        mu0 = 4.0 * math.pi * 1e-7
        eps0 = 8.854e-12
        layer_idx_tensor = torch.zeros_like(z, dtype=torch.long)
        for li, layer in enumerate(layer_info):
            mask = (z >= layer['z_min']) & (z < layer['z_max'])
            layer_idx_tensor[mask] = li

        km2 = torch.zeros_like(z)
        for li, layer in enumerate(layer_info):
            mask = (layer_idx_tensor == li)
            mu = layer.get('mu_r', 1.0) * mu0
            eps_val = layer.get('eps_r', 1.0) * eps0
            km2[mask] = omega ** 2 * mu * eps_val

        gauge_Re = div_S0_Re - 1j * (km2 / (omega + self.eps)) * Kphi_Re
        gauge_Im = div_S0_Im - 1j * (km2 / (omega + self.eps)) * Kphi_Im

        return OrderedDict([
            ('gauge_Re_mean', gauge_Re.real.abs().mean().item()),
            ('gauge_Re_max', gauge_Re.real.abs().max().item()),
            ('gauge_Im_mean', gauge_Im.real.abs().mean().item()),
            ('gauge_Im_max', gauge_Im.real.abs().max().item()),
        ])


class BoundaryConditionEvaluator:
    def __init__(self, eps=1e-30):
        self.eps = eps

    def compute_interface_residuals(self, model, interface_points, freq_idx,
                                   frequencies, layer_info, device='cpu'):
        model.eval()
        x_in = interface_points.clone().detach().requires_grad_(True).to(device)
        pred = model(x_in)

        residuals = OrderedDict()
        component_names = [
            'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
            'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
            'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
            'Kphi_Re', 'Kphi_Im'
        ]

        for i, name in enumerate(component_names):
            Kc = pred[:, i:i + 1]
            grad = torch.autograd.grad(Kc.sum(), x_in, create_graph=True)[0]
            dK_dz = grad[:, 2:3]
            residuals[name] = {
                'continuity_error': dK_dz.abs().mean().item(),
                'max_gradient': dK_dz.abs().max().item(),
            }

        return residuals

    def compute_radiation_condition_residuals(self, model, far_field_points,
                                              freq_idx, frequencies, device='cpu'):
        model.eval()
        x_in = far_field_points.clone().detach().requires_grad_(True).to(device)
        pred = model(x_in)

        freq = frequencies[freq_idx]
        omega = 2.0 * math.pi * freq
        c = 3e8
        k = omega / c

        rho = x_in[:, 6:7]

        residuals = OrderedDict()
        component_names = [
            'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
            'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
            'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
            'Kphi_Re', 'Kphi_Im'
        ]

        for i, name in enumerate(component_names):
            Kc = pred[:, i:i + 1]
            grad = torch.autograd.grad(Kc.sum(), x_in, create_graph=True)[0]
            dK_dx = grad[:, 0:1]
            dK_dy = grad[:, 1:2]

            rho_safe = rho + self.eps
            dK_drho = (x_in[:, 0:1] * dK_dx + x_in[:, 1:2] * dK_dy) / rho_safe

            sommerfeld_res = dK_drho + 1j * k * Kc
            residuals[name] = {
                'mean_abs': sommerfeld_res.abs().mean().item(),
                'max_abs': sommerfeld_res.abs().max().item(),
            }

        return residuals


class ModelEvaluator:
    def __init__(self, model, layer_info, frequencies, device='cpu',
                 component_names=None):
        self.model = model
        self.layer_info = layer_info
        self.frequencies = frequencies
        self.device = device

        self.error_metrics = RelativeErrorMetrics()
        self.component_evaluator = ComponentWiseEvaluator(component_names)
        self.spatial_analyzer = SpatialErrorAnalyzer()
        self.pde_evaluator = PDEResidualEvaluator(layer_info, frequencies)
        self.gauge_evaluator = GaugeConditionEvaluator()
        self.bc_evaluator = BoundaryConditionEvaluator()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.freq_evaluator = FrequencyDependentEvaluator(frequencies, component_names)

    def evaluate(self, data_dict, loss_history=None, compute_pde=True,
                 compute_gauge=True, compute_bc=True):
        results = OrderedDict()

        all_pred = []
        all_target = []

        for freq_idx in range(len(self.frequencies)):
            if freq_idx not in data_dict:
                continue
            data = data_dict[freq_idx]
            x = data['x'].to(self.device)
            target = data['y'].to(self.device)

            self.model.eval()
            with torch.no_grad():
                pred = self.model(x)

            all_pred.append(pred.cpu())
            all_target.append(target.cpu())

            comp_results = self.component_evaluator.evaluate_components(pred, target)
            results[f'freq_{freq_idx}'] = {
                'frequency': self.frequencies[freq_idx],
                'component_metrics': comp_results,
                'overall_rel_l2': self.error_metrics.relative_l2_error(pred, target),
                'overall_rel_l1': self.error_metrics.relative_l1_error(pred, target),
            }

        if all_pred:
            all_pred = torch.cat(all_pred, dim=0)
            all_target = torch.cat(all_target, dim=0)
            results['overall'] = self.error_metrics.all_metrics(all_pred, all_target)

        if loss_history is not None:
            convergence = self.convergence_analyzer.compute_convergence_rate(loss_history)
            plateau = self.convergence_analyzer.detect_plateau(loss_history)
            ratios = self.convergence_analyzer.compute_loss_component_ratios(loss_history)
            results['convergence'] = convergence
            results['plateau'] = plateau
            results['loss_ratios'] = ratios

        return results

    def evaluate_pde_residuals(self, x_interior, freq_idx):
        return self.pde_evaluator.compute_pde_residuals(
            self.model, x_interior, freq_idx, self.device
        )

    def evaluate_gauge_condition(self, x_points, freq_idx):
        return self.gauge_evaluator.compute_gauge_residuals(
            self.model, x_points, freq_idx, self.frequencies,
            self.layer_info, self.device
        )

    def evaluate_boundary_conditions(self, interface_points, far_field_points,
                                     freq_idx):
        interface_res = self.bc_evaluator.compute_interface_residuals(
            self.model, interface_points, freq_idx,
            self.frequencies, self.layer_info, self.device
        )
        radiation_res = self.bc_evaluator.compute_radiation_condition_residuals(
            self.model, far_field_points, freq_idx,
            self.frequencies, self.device
        )
        return {'interface': interface_res, 'radiation': radiation_res}

    def full_evaluation(self, data_dict, interior_points_dict,
                        interface_points_dict, far_field_points_dict,
                        loss_history=None):
        results = OrderedDict()

        data_results = self.evaluate(data_dict, loss_history=loss_history)
        results['data_evaluation'] = data_results

        pde_results = OrderedDict()
        for freq_idx in range(len(self.frequencies)):
            if freq_idx in interior_points_dict:
                pde_res = self.evaluate_pde_residuals(
                    interior_points_dict[freq_idx], freq_idx
                )
                pde_results[f'freq_{freq_idx}'] = pde_res
        results['pde_residuals'] = pde_results

        gauge_results = OrderedDict()
        for freq_idx in range(len(self.frequencies)):
            if freq_idx in interior_points_dict:
                gauge_res = self.evaluate_gauge_condition(
                    interior_points_dict[freq_idx], freq_idx
                )
                gauge_results[f'freq_{freq_idx}'] = gauge_res
        results['gauge_condition'] = gauge_results

        bc_results = OrderedDict()
        for freq_idx in range(len(self.frequencies)):
            if freq_idx in interface_points_dict and freq_idx in far_field_points_dict:
                bc_res = self.evaluate_boundary_conditions(
                    interface_points_dict[freq_idx],
                    far_field_points_dict[freq_idx],
                    freq_idx
                )
                bc_results[f'freq_{freq_idx}'] = bc_res
        results['boundary_conditions'] = bc_results

        return results


class MultiMediumEvaluator:
    def __init__(self, model, layer_info_dict, frequencies, medium_profiles_tensor,
                 device='cpu', component_names=None):
        self.model = model
        self.layer_info_dict = layer_info_dict
        self.frequencies = frequencies
        self.medium_profiles_tensor = medium_profiles_tensor
        self.device = device
        self.error_metrics = RelativeErrorMetrics()
        self.component_evaluator = ComponentWiseEvaluator(component_names)

    def evaluate_medium(self, medium_idx, data_dict, freq_idx):
        if medium_idx not in data_dict or freq_idx not in data_dict[medium_idx]:
            return None

        data = data_dict[medium_idx][freq_idx]
        x = data['x'].to(self.device)
        target = data['y'].to(self.device)
        medium_profile = self.medium_profiles_tensor[(medium_idx, freq_idx)]

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, medium_profile)

        comp_results = self.component_evaluator.evaluate_components(pred, target)
        overall = self.error_metrics.all_metrics(pred, target)

        return OrderedDict([
            ('medium_idx', medium_idx),
            ('freq_idx', freq_idx),
            ('component_metrics', comp_results),
            ('overall_metrics', overall),
        ])

    def evaluate_all_media(self, data_dict):
        results = OrderedDict()
        for medium_idx in self.layer_info_dict.keys():
            medium_results = OrderedDict()
            for freq_idx in range(len(self.frequencies)):
                res = self.evaluate_medium(medium_idx, data_dict, freq_idx)
                if res is not None:
                    medium_results[f'freq_{freq_idx}'] = res
            results[f'medium_{medium_idx}'] = medium_results
        return results

    def compute_generalization_metrics(self, results):
        medium_avg_errors = OrderedDict()
        for medium_key, medium_res in results.items():
            errors = []
            for freq_key, freq_res in medium_res.items():
                if 'overall_metrics' in freq_res:
                    errors.append(freq_res['overall_metrics']['rel_l2'])
            if errors:
                medium_avg_errors[medium_key] = {
                    'mean_rel_l2': np.mean(errors),
                    'std_rel_l2': np.std(errors),
                    'max_rel_l2': np.max(errors),
                    'min_rel_l2': np.min(errors),
                }

        if medium_avg_errors:
            all_means = [v['mean_rel_l2'] for v in medium_avg_errors.values()]
            return OrderedDict([
                ('overall_mean', np.mean(all_means)),
                ('overall_std', np.std(all_means)),
                ('worst_medium', max(medium_avg_errors.keys(),
                                     key=lambda k: medium_avg_errors[k]['mean_rel_l2'])),
                ('best_medium', min(medium_avg_errors.keys(),
                                    key=lambda k: medium_avg_errors[k]['mean_rel_l2'])),
                ('per_medium', medium_avg_errors),
            ])
        return None

    def compute_cross_medium_consistency(self, test_points, freq_idx):
        self.model.eval()
        predictions = OrderedDict()

        for medium_idx in self.layer_info_dict.keys():
            medium_profile = self.medium_profiles_tensor[(medium_idx, freq_idx)]
            x = test_points.to(self.device)
            with torch.no_grad():
                pred = self.model(x, medium_profile)
            predictions[medium_idx] = pred.cpu()

        consistency = OrderedDict()
        medium_indices = list(predictions.keys())
        for i in range(len(medium_indices)):
            for j in range(i + 1, len(medium_indices)):
                mi = medium_indices[i]
                mj = medium_indices[j]
                pred_i = predictions[mi]
                pred_j = predictions[mj]
                diff = (pred_i - pred_j).abs().mean().item()
                consistency[f'medium_{mi}_vs_{mj}'] = {
                    'mean_abs_diff': diff,
                    'max_abs_diff': (pred_i - pred_j).abs().max().item(),
                    'rel_diff': diff / (pred_i.abs().mean().item() + 1e-30),
                }

        return consistency


class EvaluationReporter:
    def __init__(self, output_dir='evaluation_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, results, filename='evaluation_report.txt'):
        filepath = os.path.join(self.output_dir, filename)
        lines = []
        lines.append("=" * 70)
        lines.append("FRPINNs/FRPINNs-FNO Evaluation Report")
        lines.append("=" * 70)
        lines.append("")

        self._format_section(lines, results, indent=0)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return filepath

    def _format_section(self, lines, data, indent=0):
        prefix = "  " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    self._format_section(lines, value, indent + 1)
                elif isinstance(value, (list, np.ndarray)):
                    if len(str(value)) < 80:
                        lines.append(f"{prefix}{key}: {value}")
                    else:
                        lines.append(f"{prefix}{key}: [{len(value)} items]")
                elif isinstance(value, float):
                    if abs(value) < 1e-3 or abs(value) > 1e5:
                        lines.append(f"{prefix}{key}: {value:.6e}")
                    else:
                        lines.append(f"{prefix}{key}: {value:.6f}")
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    lines.append(f"{prefix}[{i}]:")
                    self._format_section(lines, item, indent + 1)
                else:
                    lines.append(f"{prefix}[{i}]: {item}")

    def save_results_json(self, results, filename='evaluation_results.json'):
        import json
        filepath = os.path.join(self.output_dir, filename)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, OrderedDict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable = convert_to_serializable(results)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        return filepath

    def generate_comparison_table(self, results_dict, metric='rel_l2'):
        lines = []
        header = f"{'Model':<25} {'Mean Rel L2':<15} {'Max Rel L2':<15} {'Min Rel L2':<15}"
        lines.append(header)
        lines.append("-" * 70)

        for model_name, results in results_dict.items():
            if isinstance(results, dict) and 'overall' in results:
                overall = results['overall']
                mean_val = overall.get('rel_l2', float('nan'))
                lines.append(
                    f"{model_name:<25} {mean_val:<15.6e} {'-':<15} {'-':<15}"
                )
            elif isinstance(results, (int, float)):
                lines.append(f"{model_name:<25} {results:<15.6e} {'-':<15} {'-':<15}")

        return '\n'.join(lines)


class BenchmarkComparator:
    def __init__(self):
        self.methods = OrderedDict()

    def add_method(self, name, results):
        self.methods[name] = results

    def compare(self, metric='rel_l2'):
        comparison = OrderedDict()
        for name, results in self.methods.items():
            if isinstance(results, dict):
                if 'overall' in results:
                    comparison[name] = results['overall'].get(metric, None)
                elif metric in results:
                    comparison[name] = results[metric]
                else:
                    for key, val in results.items():
                        if isinstance(val, dict) and metric in val:
                            comparison[f"{name}_{key}"] = val[metric]
                            break
            elif isinstance(results, (int, float)):
                comparison[name] = results

        if comparison:
            sorted_methods = sorted(comparison.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
            return OrderedDict(sorted_methods)
        return comparison

    def compute_speedup(self, baseline_name, method_name):
        if baseline_name not in self.methods or method_name not in self.methods:
            return None
        baseline = self.methods[baseline_name]
        method = self.methods[method_name]
        if isinstance(baseline, dict) and isinstance(method, dict):
            baseline_error = baseline.get('overall', {}).get('rel_l2', None)
            method_error = method.get('overall', {}).get('rel_l2', None)
            if baseline_error and method_error and method_error > 0:
                return baseline_error / method_error
        return None

    def generate_latex_table(self, metric='rel_l2', caption="Comparison of Methods"):
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + caption + "}")
        lines.append(r"\begin{tabular}{lc}")
        lines.append(r"\hline")
        lines.append(r"Method & Relative L2 Error \\")
        lines.append(r"\hline")

        comparison = self.compare(metric)
        for name, value in comparison.items():
            if value is not None:
                lines.append(f"{name} & {value:.4e} \\\\")
            else:
                lines.append(f"{name} & N/A \\\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return '\n'.join(lines)


class UncertaintyEstimator:
    def __init__(self, n_forward_passes=10, eps=1e-30):
        self.n_forward_passes = n_forward_passes
        self.eps = eps

    def mc_dropout_uncertainty(self, model, x, n_passes=None):
        if n_passes is None:
            n_passes = self.n_forward_passes

        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model.modules()
        )
        if not has_dropout:
            return None

        model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                pred = model(x)
                predictions.append(pred.cpu())

        stacked = torch.stack(predictions, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        return OrderedDict([
            ('mean', mean),
            ('std', std),
            ('mean_uncertainty', std.mean().item()),
            ('max_uncertainty', std.max().item()),
            ('relative_uncertainty', (std / (mean.abs() + self.eps)).mean().item()),
        ])

    def ensemble_uncertainty(self, models, x):
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred.cpu())

        stacked = torch.stack(predictions, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        return OrderedDict([
            ('mean', mean),
            ('std', std),
            ('mean_uncertainty', std.mean().item()),
            ('max_uncertainty', std.max().item()),
            ('relative_uncertainty', (std / (mean.abs() + self.eps)).mean().item()),
            ('n_models', len(models)),
        ])

    def subnet_disagreement(self, model, x):
        if not hasattr(model, 'get_all_subnet_outputs'):
            return None

        outputs = model.get_all_subnet_outputs(x)
        stacked = torch.stack(outputs, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        return OrderedDict([
            ('mean', mean),
            ('std', std),
            ('mean_disagreement', std.mean().item()),
            ('max_disagreement', std.max().item()),
            ('relative_disagreement', (std / (mean.abs() + self.eps)).mean().item()),
            ('n_subnets', len(outputs)),
        ])


class ModelComplexityEvaluator:
    def __init__(self):
        pass

    def count_parameters(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return OrderedDict([
            ('total', total),
            ('trainable', trainable),
            ('non_trainable', total - trainable),
        ])

    def estimate_flops(self, model, input_shape):
        total_flops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
                if module.bias is not None:
                    total_flops += module.out_features
        return OrderedDict([
            ('estimated_flops', total_flops),
            ('input_shape', input_shape),
        ])

    def measure_inference_time(self, model, x, n_runs=100, device='cpu'):
        import time
        model.eval()
        model.to(device)
        x = x.to(device)

        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        elapsed = time.time() - start

        return OrderedDict([
            ('total_time_s', elapsed),
            ('avg_time_ms', elapsed / n_runs * 1000),
            ('throughput_samples_per_s', n_runs * x.shape[0] / elapsed),
        ])

    def get_model_size_mb(self, model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_bytes = param_size + buffer_size
        return OrderedDict([
            ('param_size_mb', param_size / (1024 ** 2)),
            ('buffer_size_mb', buffer_size / (1024 ** 2)),
            ('total_size_mb', total_bytes / (1024 ** 2)),
        ])


def create_evaluator(model_type='frpinn', model=None, layer_info=None,
                     frequencies=None, device='cpu', **kwargs):
    if model_type == 'frpinn':
        return ModelEvaluator(
            model=model,
            layer_info=layer_info,
            frequencies=frequencies,
            device=device,
            component_names=kwargs.get('component_names', None),
        )
    elif model_type == 'frpinn_fno':
        return MultiMediumEvaluator(
            model=model,
            layer_info_dict=kwargs.get('layer_info_dict', {}),
            frequencies=frequencies,
            medium_profiles_tensor=kwargs.get('medium_profiles_tensor', {}),
            device=device,
            component_names=kwargs.get('component_names', None),
        )
    else:
        return ModelEvaluator(
            model=model,
            layer_info=layer_info,
            frequencies=frequencies,
            device=device,
        )
