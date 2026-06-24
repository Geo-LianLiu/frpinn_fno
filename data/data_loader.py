import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
import math
from typing import Optional, List, Dict, Tuple, Union, Callable


def latin_hypercube_sampling(n_samples, bounds):
    dim = len(bounds)
    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        low, high = bounds[d]
        perm = np.random.permutation(n_samples)
        grid = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
        samples[:, d] = low + grid * (high - low)
    return samples


def log_uniform_sampling(n_samples, low, high):
    log_low = np.log10(low + 1e-30)
    log_high = np.log10(high)
    log_samples = np.linspace(log_low, log_high, n_samples)
    return 10.0 ** log_samples


def sobol_sampling(n_samples, bounds, scramble=True, seed=None):
    try:
        from scipy.stats.qmc import Sobol
        dim = len(bounds)
        if seed is not None:
            sampler = Sobol(d=dim, scramble=scramble, seed=seed)
        else:
            sampler = Sobol(d=dim, scramble=scramble)
        n_pow2 = 1
        while n_pow2 < n_samples:
            n_pow2 *= 2
        raw = sampler.random(n_pow2)[:n_samples]
        samples = np.zeros_like(raw)
        for d in range(dim):
            low, high = bounds[d]
            samples[:, d] = low + raw[:, d] * (high - low)
        return samples
    except ImportError:
        return latin_hypercube_sampling(n_samples, bounds)


def halton_sampling(n_samples, bounds, seed=None):
    try:
        from scipy.stats.qmc import Halton
        dim = len(bounds)
        if seed is not None:
            sampler = Halton(d=dim, scramble=True, seed=seed)
        else:
            sampler = Halton(d=dim, scramble=True)
        raw = sampler.random(n_samples)
        samples = np.zeros_like(raw)
        for d in range(dim):
            low, high = bounds[d]
            samples[:, d] = low + raw[:, d] * (high - low)
        return samples
    except ImportError:
        return latin_hypercube_sampling(n_samples, bounds)


def uniform_random_sampling(n_samples, bounds):
    dim = len(bounds)
    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        low, high = bounds[d]
        samples[:, d] = np.random.uniform(low, high, n_samples)
    return samples


def stratified_sampling(n_samples, bounds, n_strata=10):
    dim = len(bounds)
    samples_per_stratum = max(1, n_samples // (n_strata ** min(dim, 3)))
    all_samples = []
    if dim <= 3:
        strata_grids = []
        for d in range(dim):
            low, high = bounds[d]
            edges = np.linspace(low, high, n_strata + 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            strata_grids.append(centers)
        mesh = np.meshgrid(*strata_grids, indexing='ij')
        strata_points = np.stack([m.ravel() for m in mesh], axis=-1)
        n_strata_total = len(strata_points)
        samples_per_stratum = max(1, n_samples // n_strata_total)
        for sp in strata_points:
            jitter = np.zeros(dim)
            for d in range(dim):
                low, high = bounds[d]
                delta = (high - low) / n_strata / 2.0
                jitter[d] = np.random.uniform(-delta, delta)
            for _ in range(samples_per_stratum):
                all_samples.append(sp + np.random.uniform(0, 0.01, dim))
    else:
        return latin_hypercube_sampling(n_samples, bounds)
    result = np.array(all_samples)
    if len(result) > n_samples:
        indices = np.random.choice(len(result), n_samples, replace=False)
        result = result[indices]
    return result


class CoordinateTransformer:
    def __init__(self, source_pos=None):
        if source_pos is None:
            source_pos = [0.0, 0.0, 0.0]
        self.source_pos = np.array(source_pos)

    def cartesian_to_cylindrical(self, x, y, z):
        dx = x - self.source_pos[0]
        dy = y - self.source_pos[1]
        rho = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)
        z_shifted = z - self.source_pos[2]
        return rho, theta, z_shifted

    def cylindrical_to_cartesian(self, rho, theta, z):
        x = rho * np.cos(theta) + self.source_pos[0]
        y = rho * np.sin(theta) + self.source_pos[1]
        z_shifted = z + self.source_pos[2]
        return x, y, z_shifted

    def batch_cartesian_to_cylindrical(self, points):
        rho, theta, z = self.cartesian_to_cylindrical(
            points[:, 0], points[:, 1], points[:, 2]
        )
        result = points.copy()
        result[:, 0] = rho
        result[:, 1] = theta
        result[:, 2] = z
        return result

    def batch_cylindrical_to_cartesian(self, points):
        x, y, z = self.cylindrical_to_cartesian(
            points[:, 0], points[:, 1], points[:, 2]
        )
        result = points.copy()
        result[:, 0] = x
        result[:, 1] = y
        result[:, 2] = z
        return result

    def compute_rho_from_points(self, points):
        dx = points[:, 0] - points[:, 3]
        dy = points[:, 1] - points[:, 4]
        return np.sqrt(dx ** 2 + dy ** 2)


class DataNormalizer:
    def __init__(self, method='standard', eps=1e-8):
        self.method = method
        self.eps = eps
        self.mean_x = None
        self.std_x = None
        self.min_x = None
        self.max_x = None
        self.mean_y = None
        self.std_y = None
        self.min_y = None
        self.max_y = None
        self._fitted = False

    def fit(self, data_x, data_y):
        if isinstance(data_x, torch.Tensor):
            data_x = data_x.cpu().numpy()
        if isinstance(data_y, torch.Tensor):
            data_y = data_y.cpu().numpy()

        self.mean_x = np.mean(data_x, axis=0)
        self.std_x = np.std(data_x, axis=0) + self.eps
        self.min_x = np.min(data_x, axis=0)
        self.max_x = np.max(data_x, axis=0)

        self.mean_y = np.mean(data_y, axis=0)
        self.std_y = np.std(data_y, axis=0) + self.eps
        self.min_y = np.min(data_y, axis=0)
        self.max_y = np.max(data_y, axis=0)

        self._fitted = True
        return self

    def transform_x(self, data_x):
        if not self._fitted:
            return data_x
        if isinstance(data_x, torch.Tensor):
            data_x_np = data_x.cpu().numpy()
        else:
            data_x_np = data_x

        if self.method == 'standard':
            result = (data_x_np - self.mean_x) / self.std_x
        elif self.method == 'minmax':
            result = (data_x_np - self.min_x) / (self.max_x - self.min_x + self.eps)
        elif self.method == 'robust':
            q75_x = np.percentile(data_x_np, 75, axis=0)
            q25_x = np.percentile(data_x_np, 25, axis=0)
            iqr_x = q75_x - q25_x + self.eps
            median_x = np.median(data_x_np, axis=0)
            result = (data_x_np - median_x) / iqr_x
        else:
            result = data_x_np

        if isinstance(data_x, torch.Tensor):
            return torch.tensor(result, dtype=data_x.dtype, device=data_x.device)
        return result

    def transform_y(self, data_y):
        if not self._fitted:
            return data_y
        if isinstance(data_y, torch.Tensor):
            data_y_np = data_y.cpu().numpy()
        else:
            data_y_np = data_y

        if self.method == 'standard':
            result = (data_y_np - self.mean_y) / self.std_y
        elif self.method == 'minmax':
            result = (data_y_np - self.min_y) / (self.max_y - self.min_y + self.eps)
        elif self.method == 'robust':
            q75_y = np.percentile(data_y_np, 75, axis=0)
            q25_y = np.percentile(data_y_np, 25, axis=0)
            iqr_y = q75_y - q25_y + self.eps
            median_y = np.median(data_y_np, axis=0)
            result = (data_y_np - median_y) / iqr_y
        else:
            result = data_y_np

        if isinstance(data_y, torch.Tensor):
            return torch.tensor(result, dtype=data_y.dtype, device=data_y.device)
        return result

    def inverse_transform_y(self, data_y):
        if not self._fitted:
            return data_y
        if isinstance(data_y, torch.Tensor):
            data_y_np = data_y.cpu().numpy()
        else:
            data_y_np = data_y

        if self.method == 'standard':
            result = data_y_np * self.std_y + self.mean_y
        elif self.method == 'minmax':
            result = data_y_np * (self.max_y - self.min_y + self.eps) + self.min_y
        else:
            result = data_y_np

        if isinstance(data_y, torch.Tensor):
            return torch.tensor(result, dtype=data_y.dtype, device=data_y.device)
        return result

    def get_state(self):
        return {
            'method': self.method,
            'mean_x': self.mean_x.tolist() if self.mean_x is not None else None,
            'std_x': self.std_x.tolist() if self.std_x is not None else None,
            'min_x': self.min_x.tolist() if self.min_x is not None else None,
            'max_x': self.max_x.tolist() if self.max_x is not None else None,
            'mean_y': self.mean_y.tolist() if self.mean_y is not None else None,
            'std_y': self.std_y.tolist() if self.std_y is not None else None,
            'min_y': self.min_y.tolist() if self.min_y is not None else None,
            'max_y': self.max_y.tolist() if self.max_y is not None else None,
            'fitted': self._fitted,
        }

    def load_state(self, state):
        self.method = state['method']
        if state.get('mean_x') is not None:
            self.mean_x = np.array(state['mean_x'])
            self.std_x = np.array(state['std_x'])
            self.min_x = np.array(state['min_x'])
            self.max_x = np.array(state['max_x'])
            self.mean_y = np.array(state['mean_y'])
            self.std_y = np.array(state['std_y'])
            self.min_y = np.array(state['min_y'])
            self.max_y = np.array(state['max_y'])
            self._fitted = state.get('fitted', True)


class DataAugmentor:
    def __init__(self, noise_level=0.01, rotate_angle_range=0.0,
                 translate_range=0.0, mirror_prob=0.0):
        self.noise_level = noise_level
        self.rotate_angle_range = rotate_angle_range
        self.translate_range = translate_range
        self.mirror_prob = mirror_prob

    def add_gaussian_noise(self, data_y, noise_level=None):
        if noise_level is None:
            noise_level = self.noise_level
        if isinstance(data_y, torch.Tensor):
            noise = torch.randn_like(data_y) * noise_level
            return data_y + noise
        else:
            noise = np.random.randn(*data_y.shape) * noise_level
            return data_y + noise

    def add_relative_noise(self, data_y, relative_level=0.01):
        if isinstance(data_y, torch.Tensor):
            scale = data_y.abs() + 1e-10
            noise = torch.randn_like(data_y) * scale * relative_level
            return data_y + noise
        else:
            scale = np.abs(data_y) + 1e-10
            noise = np.random.randn(*data_y.shape) * scale * relative_level
            return data_y + noise

    def mirror_x(self, data_x, prob=None):
        if prob is None:
            prob = self.mirror_prob
        if isinstance(data_x, torch.Tensor):
            mask = torch.rand(data_x.shape[0]) < prob
            result = data_x.clone()
            result[mask, 0] = -result[mask, 0]
            return result
        else:
            mask = np.random.rand(data_x.shape[0]) < prob
            result = data_x.copy()
            result[mask, 0] = -result[mask, 0]
            return result

    def rotate_xy(self, data_x, angle_range=None):
        if angle_range is None:
            angle_range = self.rotate_angle_range
        if angle_range <= 0:
            return data_x
        if isinstance(data_x, torch.Tensor):
            angles = torch.rand(data_x.shape[0]) * 2 * angle_range - angle_range
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            result = data_x.clone()
            result[:, 0] = data_x[:, 0] * cos_a - data_x[:, 1] * sin_a
            result[:, 1] = data_x[:, 0] * sin_a + data_x[:, 1] * cos_a
            return result
        else:
            angles = np.random.uniform(-angle_range, angle_range, data_x.shape[0])
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            result = data_x.copy()
            result[:, 0] = data_x[:, 0] * cos_a - data_x[:, 1] * sin_a
            result[:, 1] = data_x[:, 0] * sin_a + data_x[:, 1] * cos_a
            return result

    def augment_batch(self, data_x, data_y):
        data_x = self.mirror_x(data_x)
        data_x = self.rotate_xy(data_x)
        data_y = self.add_gaussian_noise(data_y)
        return data_x, data_y


class ResidualBasedSampler:
    def __init__(self, n_candidates_factor=5, top_k_ratio=0.5,
                 resample_ratio=0.3, eps=1e-8):
        self.n_candidates_factor = n_candidates_factor
        self.top_k_ratio = top_k_ratio
        self.resample_ratio = resample_ratio
        self.eps = eps
        self._residual_cache = None

    def compute_residuals(self, model, points, freq_idx, device='cpu'):
        model.eval()
        with torch.no_grad():
            x = points.to(device)
            pred = model(x)
        residuals = pred.abs().mean(dim=-1).cpu().numpy()
        return residuals

    def select_points(self, model, bounds, n_points, freq_idx,
                      layer_info, source_pos, rho_range, z_range,
                      frequencies, device='cpu'):
        n_candidates = n_points * self.n_candidates_factor
        sampler = PhysicsSampler(
            layer_info=layer_info,
            source_pos=source_pos,
            rho_range=rho_range,
            z_range=z_range,
            frequencies=frequencies,
        )
        candidates = sampler.sample_interior(n_candidates, freq_idx, device)
        residuals = self.compute_residuals(model, candidates, freq_idx, device)
        n_select = int(n_points * self.top_k_ratio)
        n_random = n_points - n_select
        sorted_indices = np.argsort(residuals)[::-1]
        top_indices = sorted_indices[:n_select]
        selected = candidates[top_indices].clone()
        if n_random > 0:
            random_indices = np.random.choice(n_candidates, n_random, replace=False)
            random_points = candidates[random_indices].clone()
            selected = torch.cat([selected, random_points], dim=0)
        self._residual_cache = residuals
        return selected[:n_points]

    def get_residual_statistics(self):
        if self._residual_cache is None:
            return {}
        return {
            'mean': float(np.mean(self._residual_cache)),
            'std': float(np.std(self._residual_cache)),
            'max': float(np.max(self._residual_cache)),
            'min': float(np.min(self._residual_cache)),
            'median': float(np.median(self._residual_cache)),
            'p90': float(np.percentile(self._residual_cache, 90)),
            'p95': float(np.percentile(self._residual_cache, 95)),
            'p99': float(np.percentile(self._residual_cache, 99)),
        }


class GradientBasedSampler:
    def __init__(self, n_candidates_factor=5, top_k_ratio=0.5, eps=1e-8):
        self.n_candidates_factor = n_candidates_factor
        self.top_k_ratio = top_k_ratio
        self.eps = eps

    def compute_gradient_magnitude(self, model, points, device='cpu'):
        model.eval()
        x = points.to(device).detach().requires_grad_(True)
        pred = model(x)
        grad_magnitudes = []
        for i in range(pred.shape[-1]):
            grad_i = torch.autograd.grad(
                pred[:, i].sum(), x, create_graph=False, retain_graph=True
            )[0]
            grad_mag = grad_i.abs().mean(dim=-1)
            grad_magnitudes.append(grad_mag)
        total_grad_mag = torch.stack(grad_magnitudes, dim=-1).mean(dim=-1)
        return total_grad_mag.cpu().numpy()

    def select_points(self, model, bounds, n_points, freq_idx,
                      layer_info, source_pos, rho_range, z_range,
                      frequencies, device='cpu'):
        n_candidates = n_points * self.n_candidates_factor
        sampler = PhysicsSampler(
            layer_info=layer_info,
            source_pos=source_pos,
            rho_range=rho_range,
            z_range=z_range,
            frequencies=frequencies,
        )
        candidates = sampler.sample_interior(n_candidates, freq_idx, device)
        grad_mags = self.compute_gradient_magnitude(model, candidates, device)
        n_select = int(n_points * self.top_k_ratio)
        n_random = n_points - n_select
        sorted_indices = np.argsort(grad_mags)[::-1]
        top_indices = sorted_indices[:n_select]
        selected = candidates[top_indices].clone()
        if n_random > 0:
            random_indices = np.random.choice(n_candidates, n_random, replace=False)
            random_points = candidates[random_indices].clone()
            selected = torch.cat([selected, random_points], dim=0)
        return selected[:n_points]


class CurriculumSampler:
    def __init__(self, total_epochs=10000, initial_domain_scale=0.3,
                 final_domain_scale=1.0, schedule='linear'):
        self.total_epochs = total_epochs
        self.initial_domain_scale = initial_domain_scale
        self.final_domain_scale = final_domain_scale
        self.schedule = schedule
        self.current_epoch = 0

    def get_domain_scale(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
        progress = min(1.0, epoch / max(1, self.total_epochs))
        if self.schedule == 'linear':
            scale = self.initial_domain_scale + progress * (
                self.final_domain_scale - self.initial_domain_scale
            )
        elif self.schedule == 'cosine':
            scale = self.initial_domain_scale + 0.5 * (
                self.final_domain_scale - self.initial_domain_scale
            ) * (1 - math.cos(math.pi * progress))
        elif self.schedule == 'exponential':
            scale = self.initial_domain_scale * (
                self.final_domain_scale / self.initial_domain_scale
            ) ** progress
        elif self.schedule == 'step':
            n_stages = 5
            stage = min(int(progress * n_stages), n_stages - 1)
            stage_progress = stage / n_stages
            scale = self.initial_domain_scale + stage_progress * (
                self.final_domain_scale - self.initial_domain_scale
            )
        else:
            scale = self.final_domain_scale
        return scale

    def get_scaled_bounds(self, base_bounds, epoch=None):
        scale = self.get_domain_scale(epoch)
        scaled_bounds = []
        for low, high in base_bounds:
            center = (low + high) / 2.0
            half_range = (high - low) / 2.0 * scale
            scaled_bounds.append((center - half_range, center + half_range))
        return scaled_bounds

    def get_frequency_mask(self, frequencies, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
        progress = min(1.0, epoch / max(1, self.total_epochs))
        n_freq = len(frequencies)
        n_active = max(1, int(progress * n_freq))
        mask = [i < n_active for i in range(n_freq)]
        return mask

    def step(self):
        self.current_epoch += 1


class PointQualityMetrics:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute_min_distance(self, points):
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
        n = len(points_np)
        if n < 2:
            return float('inf')
        if n > 5000:
            indices = np.random.choice(n, 5000, replace=False)
            points_sub = points_np[indices]
        else:
            points_sub = points_np
        from scipy.spatial.distance import cdist
        dists = cdist(points_sub, points_sub)
        np.fill_diagonal(dists, np.inf)
        return float(np.min(dists))

    def compute_coverage_ratio(self, points, bounds):
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
        dim = len(bounds)
        n_bins = 10
        total_bins = n_bins ** min(dim, 3)
        occupied = set()
        for p in points_np:
            bin_idx = []
            for d in range(min(dim, 3)):
                low, high = bounds[d]
                bin_d = int((p[d] - low) / (high - low + self.eps) * n_bins)
                bin_d = min(max(bin_d, 0), n_bins - 1)
                bin_idx.append(bin_d)
            occupied.add(tuple(bin_idx))
        return len(occupied) / total_bins

    def compute_discrepancy(self, points, bounds):
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
        dim = len(bounds)
        normalized = np.zeros_like(points_np)
        for d in range(dim):
            low, high = bounds[d]
            normalized[:, d] = (points_np[:, d] - low) / (high - low + self.eps)
        n = len(normalized)
        if n > 2000:
            indices = np.random.choice(n, 2000, replace=False)
            normalized = normalized[indices]
            n = 2000
        discrepancy = 0.0
        n_test = min(100, n)
        for _ in range(n_test):
            test_point = np.random.rand(dim)
            count_in = np.all(normalized <= test_point, axis=1).sum()
            volume = np.prod(test_point)
            discrepancy += abs(count_in / n - volume)
        return discrepancy / n_test

    def compute_uniformity_score(self, points, bounds):
        coverage = self.compute_coverage_ratio(points, bounds)
        discrepancy = self.compute_discrepancy(points, bounds)
        uniformity = coverage * (1.0 - min(discrepancy, 1.0))
        return uniformity

    def compute_all_metrics(self, points, bounds):
        return {
            'coverage': self.compute_coverage_ratio(points, bounds),
            'discrepancy': self.compute_discrepancy(points, bounds),
            'uniformity': self.compute_uniformity_score(points, bounds),
            'min_distance': self.compute_min_distance(points),
            'n_points': len(points),
        }


class DataValidator:
    def __init__(self, input_dim=8, output_dim=14, eps=1e-8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps

    def validate_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return {'valid': False, 'error': f'File not found: {csv_path}'}
        try:
            data = []
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    data.append([float(v) for v in row])
            if len(data) == 0:
                return {'valid': False, 'error': 'No data rows found'}
            data = np.array(data)
            expected_cols = self.input_dim + self.output_dim
            if data.shape[1] < expected_cols:
                return {
                    'valid': False,
                    'error': f'Expected at least {expected_cols} columns, got {data.shape[1]}'
                }
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            if has_nan:
                return {'valid': False, 'error': 'Data contains NaN values'}
            if has_inf:
                return {'valid': False, 'error': 'Data contains Inf values'}
            data_x = data[:, :self.input_dim]
            data_y = data[:, self.input_dim:self.input_dim + self.output_dim]
            x_stats = {
                'mean': np.mean(data_x, axis=0).tolist(),
                'std': np.std(data_x, axis=0).tolist(),
                'min': np.min(data_x, axis=0).tolist(),
                'max': np.max(data_x, axis=0).tolist(),
            }
            y_stats = {
                'mean': np.mean(data_y, axis=0).tolist(),
                'std': np.std(data_y, axis=0).tolist(),
                'min': np.min(data_y, axis=0).tolist(),
                'max': np.max(data_y, axis=0).tolist(),
            }
            return {
                'valid': True,
                'n_samples': len(data),
                'n_features': data.shape[1],
                'x_stats': x_stats,
                'y_stats': y_stats,
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def validate_batch(self, batch):
        issues = []
        for key in ['interior', 'gauge', 'interface', 'radiation', 'symmetry']:
            if key not in batch:
                issues.append(f'Missing key: {key}')
                continue
            points = batch[key]
            if points is None:
                issues.append(f'{key} is None')
                continue
            if isinstance(points, torch.Tensor):
                if torch.any(torch.isnan(points)):
                    issues.append(f'{key} contains NaN')
                if torch.any(torch.isinf(points)):
                    issues.append(f'{key} contains Inf')
                if points.shape[-1] != self.input_dim:
                    issues.append(
                        f'{key} has wrong input dim: {points.shape[-1]} != {self.input_dim}'
                    )
        if 'data_x' in batch and batch['data_x'] is not None:
            dx = batch['data_x']
            dy = batch['data_y']
            if isinstance(dx, torch.Tensor) and isinstance(dy, torch.Tensor):
                if dx.shape[0] != dy.shape[0]:
                    issues.append(f'data_x and data_y have different lengths')
                if dy.shape[-1] != self.output_dim:
                    issues.append(
                        f'data_y has wrong output dim: {dy.shape[-1]} != {self.output_dim}'
                    )
        return {'valid': len(issues) == 0, 'issues': issues}


class CSVExporter:
    def __init__(self, input_names=None, output_names=None):
        if input_names is None:
            self.input_names = ['x', 'y', 'z', 'xp', 'yp', 'zp', 'rho', 'f_log']
        else:
            self.input_names = input_names
        if output_names is None:
            self.output_names = [
                'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
                'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
                'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
                'Kphi_Re', 'Kphi_Im',
            ]
        else:
            self.output_names = output_names

    def export_predictions(self, data_x, predictions, path, medium_indices=None):
        if isinstance(data_x, torch.Tensor):
            data_x = data_x.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(medium_indices, torch.Tensor):
            medium_indices = medium_indices.cpu().numpy()

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = self.input_names + self.output_names
            if medium_indices is not None:
                header.append('medium_idx')
            writer.writerow(header)
            for i in range(len(data_x)):
                row = list(data_x[i]) + list(predictions[i])
                if medium_indices is not None:
                    row.append(int(medium_indices[i]))
                writer.writerow(row)

    def export_sampling_points(self, points, point_type, path):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = self.input_names + ['point_type']
            writer.writerow(header)
            for i in range(len(points)):
                row = list(points[i]) + [point_type]
                writer.writerow(row)

    def export_loss_history(self, loss_history, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if len(loss_history) == 0:
            return
        keys = list(loss_history[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + keys)
            for i, entry in enumerate(loss_history):
                row = [i] + [entry.get(k, 0.0) for k in keys]
                writer.writerow(row)


class LMGFDataset(Dataset):
    def __init__(self, csv_path=None, data_x=None, data_y=None,
                 normalizer=None, augmentor=None):
        self.normalizer = normalizer
        self.augmentor = augmentor

        if csv_path is not None and os.path.exists(csv_path):
            self._load_csv(csv_path)
        elif data_x is not None and data_y is not None:
            self.data_x = torch.tensor(data_x, dtype=torch.float32)
            self.data_y = torch.tensor(data_y, dtype=torch.float32)
        else:
            self.data_x = torch.empty(0, 8)
            self.data_y = torch.empty(0, 14)

        if self.normalizer is not None and len(self.data_x) > 0:
            self.normalizer.fit(self.data_x, self.data_y)
            self.data_x = self.normalizer.transform_x(self.data_x)
            self.data_y = self.normalizer.transform_y(self.data_y)

    def _load_csv(self, csv_path):
        data = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                data.append([float(v) for v in row])
        data = np.array(data)
        self.data_x = torch.tensor(data[:, :8], dtype=torch.float32)
        self.data_y = torch.tensor(data[:, 8:], dtype=torch.float32)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        if self.augmentor is not None and self.training:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            x, y = self.augmentor.augment_batch(x, y)
            x = x.squeeze(0)
            y = y.squeeze(0)
        return x, y

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class PhysicsSampler:
    def __init__(self, layer_info, source_pos, rho_range, z_range,
                 frequencies, delta_z=0.5, R_min=0.1,
                 sampling_method='lhs', seed=None):
        self.layer_info = layer_info
        self.source_pos = source_pos
        self.rho_range = rho_range
        self.z_range = z_range
        self.frequencies = frequencies
        self.delta_z = delta_z
        self.R_min = R_min
        self.sampling_method = sampling_method
        self.seed = seed
        self.interface_z = self._get_interfaces()
        self._coord_transformer = CoordinateTransformer(source_pos)

        if seed is not None:
            np.random.seed(seed)

    def _get_interfaces(self):
        interfaces = []
        for i in range(len(self.layer_info) - 1):
            interfaces.append(self.layer_info[i]['z_max'])
        return interfaces

    def _get_sampling_function(self):
        if self.sampling_method == 'sobol':
            return sobol_sampling
        elif self.sampling_method == 'halton':
            return halton_sampling
        elif self.sampling_method == 'stratified':
            return stratified_sampling
        elif self.sampling_method == 'uniform':
            return uniform_random_sampling
        else:
            return latin_hypercube_sampling

    def _compute_rho(self, samples):
        x = samples[:, 0]
        y = samples[:, 1]
        xp = samples[:, 3]
        yp = samples[:, 4]
        return np.sqrt((x - xp) ** 2 + (y - yp) ** 2)

    def _filter_valid_points(self, samples, n_target):
        z_vals = samples[:, 2]
        valid_mask = np.ones(len(samples), dtype=bool)
        for z_if in self.interface_z:
            valid_mask &= np.abs(z_vals - z_if) > self.delta_z

        R_vals = np.sqrt(
            (samples[:, 0] - samples[:, 3]) ** 2 +
            (samples[:, 1] - samples[:, 4]) ** 2 +
            (samples[:, 2] - samples[:, 5]) ** 2
        )
        valid_mask &= R_vals > self.R_min

        return valid_mask

    def _resample_to_target(self, samples, valid_mask, n_target, bounds, freq_idx):
        valid_samples = samples[valid_mask]
        if len(valid_samples) >= n_target:
            return valid_samples[:n_target]

        n_extra = n_target - len(valid_samples)
        sample_fn = self._get_sampling_function()
        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq)

        extra = sample_fn(n_extra * 3, bounds)
        for i in range(len(extra)):
            x = extra[i, 0]
            y = extra[i, 1]
            xp = extra[i, 3]
            yp = extra[i, 4]
            extra[i, 6] = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
            extra[i, 7] = f_log

        extra_valid = self._filter_valid_points(extra, n_extra)
        extra_samples = extra[extra_valid]

        combined = np.vstack([valid_samples, extra_samples])
        if len(combined) < n_target:
            remaining = n_target - len(combined)
            random_extra = latin_hypercube_sampling(remaining, bounds)
            for i in range(len(random_extra)):
                x = random_extra[i, 0]
                y = random_extra[i, 1]
                xp = random_extra[i, 3]
                yp = random_extra[i, 4]
                random_extra[i, 6] = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
                random_extra[i, 7] = f_log
            combined = np.vstack([combined, random_extra])

        return combined[:n_target]

    def sample_interior(self, n_points, freq_idx, device='cpu'):
        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq)

        bounds = [
            (-self.rho_range[1], self.rho_range[1]),
            (-self.rho_range[1], self.rho_range[1]),
            (self.z_range[0], self.z_range[1]),
            (self.source_pos[0] - 1.0, self.source_pos[0] + 1.0),
            (self.source_pos[1] - 1.0, self.source_pos[1] + 1.0),
            (self.source_pos[2] - 1.0, self.source_pos[2] + 1.0),
            (self.rho_range[0], self.rho_range[1]),
            (f_log - 0.01, f_log + 0.01),
        ]

        sample_fn = self._get_sampling_function()
        samples = sample_fn(n_points * 2, bounds)

        for i in range(len(samples)):
            x = samples[i, 0]
            y = samples[i, 1]
            xp = samples[i, 3]
            yp = samples[i, 4]
            samples[i, 6] = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
            samples[i, 7] = f_log

        valid_mask = self._filter_valid_points(samples, n_points)
        samples = self._resample_to_target(samples, valid_mask, n_points, bounds, freq_idx)

        return torch.tensor(samples[:n_points], dtype=torch.float32, device=device)

    def sample_interface(self, n_points, freq_idx, device='cpu'):
        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq)

        points = []
        n_per_interface = n_points // max(len(self.interface_z), 1)

        for z_if in self.interface_z:
            rho_vals = np.random.uniform(self.rho_range[0], self.rho_range[1], n_per_interface)
            theta_vals = np.random.uniform(0, 2 * np.pi, n_per_interface)
            x_vals = self.source_pos[0] + rho_vals * np.cos(theta_vals)
            y_vals = self.source_pos[1] + rho_vals * np.sin(theta_vals)

            batch = np.zeros((n_per_interface, 8))
            batch[:, 0] = x_vals
            batch[:, 1] = y_vals
            batch[:, 2] = z_if
            batch[:, 3] = self.source_pos[0]
            batch[:, 4] = self.source_pos[1]
            batch[:, 5] = self.source_pos[2]
            batch[:, 6] = rho_vals
            batch[:, 7] = f_log
            points.append(batch)

        if len(points) > 0:
            points = np.vstack(points)
        else:
            points = np.zeros((0, 8))

        return torch.tensor(points, dtype=torch.float32, device=device)

    def sample_radiation(self, n_points, freq_idx, device='cpu'):
        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq)

        rho_far = self.rho_range[1] * 0.95
        theta_vals = np.random.uniform(0, 2 * np.pi, n_points)
        z_vals = np.random.uniform(self.z_range[0], self.z_range[1], n_points)

        batch = np.zeros((n_points, 8))
        batch[:, 0] = self.source_pos[0] + rho_far * np.cos(theta_vals)
        batch[:, 1] = self.source_pos[1] + rho_far * np.sin(theta_vals)
        batch[:, 2] = z_vals
        batch[:, 3] = self.source_pos[0]
        batch[:, 4] = self.source_pos[1]
        batch[:, 5] = self.source_pos[2]
        batch[:, 6] = rho_far
        batch[:, 7] = f_log

        return torch.tensor(batch, dtype=torch.float32, device=device)

    def sample_gauge(self, n_points, freq_idx, device='cpu'):
        return self.sample_interior(n_points, freq_idx, device)

    def sample_symmetry(self, n_points, freq_idx, device='cpu'):
        return self.sample_interior(n_points, freq_idx, device)

    def sample_near_source(self, n_points, freq_idx, device='cpu', radius=5.0):
        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq)

        rho_vals = np.random.uniform(0, radius, n_points)
        theta_vals = np.random.uniform(0, 2 * np.pi, n_points)
        z_vals = np.random.uniform(
            self.source_pos[2] - radius,
            self.source_pos[2] + radius,
            n_points
        )

        batch = np.zeros((n_points, 8))
        batch[:, 0] = self.source_pos[0] + rho_vals * np.cos(theta_vals)
        batch[:, 1] = self.source_pos[1] + rho_vals * np.sin(theta_vals)
        batch[:, 2] = z_vals
        batch[:, 3] = self.source_pos[0]
        batch[:, 4] = self.source_pos[1]
        batch[:, 5] = self.source_pos[2]
        batch[:, 6] = rho_vals
        batch[:, 7] = f_log

        return torch.tensor(batch, dtype=torch.float32, device=device)

    def sample_layer_centers(self, n_points_per_layer, freq_idx, device='cpu'):
        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq)

        points = []
        for layer in self.layer_info:
            z_center = (layer['z_min'] + layer['z_max']) / 2.0
            rho_vals = np.random.uniform(
                self.rho_range[0], self.rho_range[1], n_points_per_layer
            )
            theta_vals = np.random.uniform(0, 2 * np.pi, n_points_per_layer)
            x_vals = self.source_pos[0] + rho_vals * np.cos(theta_vals)
            y_vals = self.source_pos[1] + rho_vals * np.sin(theta_vals)

            batch = np.zeros((n_points_per_layer, 8))
            batch[:, 0] = x_vals
            batch[:, 1] = y_vals
            batch[:, 2] = z_center
            batch[:, 3] = self.source_pos[0]
            batch[:, 4] = self.source_pos[1]
            batch[:, 5] = self.source_pos[2]
            batch[:, 6] = rho_vals
            batch[:, 7] = f_log
            points.append(batch)

        if len(points) > 0:
            points = np.vstack(points)
        else:
            points = np.zeros((0, 8))

        return torch.tensor(points, dtype=torch.float32, device=device)

    def sample_multi_resolution(self, n_points, freq_idx, device='cpu',
                                 n_levels=3):
        points_per_level = n_points // n_levels
        all_points = []

        for level in range(n_levels):
            scale = 2.0 ** (-level)
            n_level = points_per_level
            if level == 0:
                n_level = n_points - points_per_level * (n_levels - 1)

            rho_max = self.rho_range[1] * scale
            freq = self.frequencies[freq_idx]
            f_log = np.log10(freq)

            bounds = [
                (-rho_max, rho_max),
                (-rho_max, rho_max),
                (self.z_range[0], self.z_range[1]),
                (self.source_pos[0] - 1.0, self.source_pos[0] + 1.0),
                (self.source_pos[1] - 1.0, self.source_pos[1] + 1.0),
                (self.source_pos[2] - 1.0, self.source_pos[2] + 1.0),
                (0, rho_max),
                (f_log - 0.01, f_log + 0.01),
            ]

            samples = latin_hypercube_sampling(n_level, bounds)
            for i in range(len(samples)):
                x = samples[i, 0]
                y = samples[i, 1]
                xp = samples[i, 3]
                yp = samples[i, 4]
                samples[i, 6] = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
                samples[i, 7] = f_log
            all_points.append(samples)

        combined = np.vstack(all_points)
        return torch.tensor(combined[:n_points], dtype=torch.float32, device=device)


class FRPINNsBatchBuilder:
    def __init__(self, sampler, dataset, n_interior=4000, n_gauge=2000,
                 n_interface=700, n_radiation=700, n_data=2000, n_sym=1000,
                 n_near_source=0, n_layer_centers=0):
        self.sampler = sampler
        self.dataset = dataset
        self.n_interior = n_interior
        self.n_gauge = n_gauge
        self.n_interface = n_interface
        self.n_radiation = n_radiation
        self.n_data = n_data
        self.n_sym = n_sym
        self.n_near_source = n_near_source
        self.n_layer_centers = n_layer_centers

    def build_batch(self, freq_idx, device='cpu'):
        batch = {}

        batch['interior'] = self.sampler.sample_interior(
            self.n_interior, freq_idx, device
        )
        batch['gauge'] = self.sampler.sample_gauge(
            self.n_gauge, freq_idx, device
        )
        batch['interface'] = self.sampler.sample_interface(
            self.n_interface, freq_idx, device
        )
        batch['radiation'] = self.sampler.sample_radiation(
            self.n_radiation, freq_idx, device
        )
        batch['symmetry'] = self.sampler.sample_symmetry(
            self.n_sym, freq_idx, device
        )

        if self.n_near_source > 0:
            batch['near_source'] = self.sampler.sample_near_source(
                self.n_near_source, freq_idx, device
            )

        if self.n_layer_centers > 0:
            batch['layer_centers'] = self.sampler.sample_layer_centers(
                self.n_layer_centers, freq_idx, device
            )

        if self.dataset is not None and len(self.dataset) > 0:
            n_avail = len(self.dataset)
            indices = np.random.choice(n_avail, min(self.n_data, n_avail), replace=False)
            data_x = self.dataset.data_x[indices].to(device)
            data_y = self.dataset.data_y[indices].to(device)
            batch['data_x'] = data_x
            batch['data_y'] = data_y
        else:
            batch['data_x'] = None
            batch['data_y'] = None

        return batch

    def build_multi_freq_batch(self, freq_indices, device='cpu'):
        multi_batch = {}
        for freq_idx in freq_indices:
            multi_batch[freq_idx] = self.build_batch(freq_idx, device)
        return multi_batch


class AdaptiveBatchBuilder:
    def __init__(self, sampler, dataset, initial_n_interior=4000,
                 initial_n_gauge=2000, initial_n_interface=700,
                 initial_n_radiation=700, initial_n_data=2000,
                 initial_n_sym=1000, adaptation_interval=500,
                 adaptation_factor=1.5):
        self.sampler = sampler
        self.dataset = dataset
        self.n_interior = initial_n_interior
        self.n_gauge = initial_n_gauge
        self.n_interface = initial_n_interface
        self.n_radiation = initial_n_radiation
        self.n_data = initial_n_data
        self.n_sym = initial_n_sym
        self.adaptation_interval = adaptation_interval
        self.adaptation_factor = adaptation_factor
        self._loss_history = {'pde': [], 'gauge': [], 'bc': [], 'data': [], 'sym': []}
        self._step_count = 0

    def update_loss_history(self, loss_dict):
        for key in self._loss_history:
            if key in loss_dict:
                self._loss_history[key].append(loss_dict[key])
        self._step_count += 1

    def _adapt_point_counts(self):
        if self._step_count < self.adaptation_interval:
            return
        if self._step_count % self.adaptation_interval != 0:
            return

        for key, history in self._loss_history.items():
            if len(history) < self.adaptation_interval:
                continue
            recent = history[-self.adaptation_interval:]
            early = history[:self.adaptation_interval]
            recent_mean = np.mean(recent)
            early_mean = np.mean(early)
            if early_mean > 0:
                ratio = recent_mean / early_mean
            else:
                ratio = 1.0

            if key == 'pde' and ratio > 0.5:
                self.n_interior = int(self.n_interior * self.adaptation_factor)
            elif key == 'gauge' and ratio > 0.5:
                self.n_gauge = int(self.n_gauge * self.adaptation_factor)
            elif key == 'bc' and ratio > 0.5:
                self.n_interface = int(self.n_interface * self.adaptation_factor)
                self.n_radiation = int(self.n_radiation * self.adaptation_factor)
            elif key == 'data' and ratio > 0.5:
                self.n_data = int(self.n_data * self.adaptation_factor)
            elif key == 'sym' and ratio > 0.5:
                self.n_sym = int(self.n_sym * self.adaptation_factor)

        max_points = 20000
        self.n_interior = min(self.n_interior, max_points)
        self.n_gauge = min(self.n_gauge, max_points // 2)
        self.n_interface = min(self.n_interface, max_points // 4)
        self.n_radiation = min(self.n_radiation, max_points // 4)
        self.n_data = min(self.n_data, max_points // 2)
        self.n_sym = min(self.n_sym, max_points // 4)

    def build_batch(self, freq_idx, device='cpu'):
        self._adapt_point_counts()

        batch = {}
        batch['interior'] = self.sampler.sample_interior(
            self.n_interior, freq_idx, device
        )
        batch['gauge'] = self.sampler.sample_gauge(
            self.n_gauge, freq_idx, device
        )
        batch['interface'] = self.sampler.sample_interface(
            self.n_interface, freq_idx, device
        )
        batch['radiation'] = self.sampler.sample_radiation(
            self.n_radiation, freq_idx, device
        )
        batch['symmetry'] = self.sampler.sample_symmetry(
            self.n_sym, freq_idx, device
        )

        if self.dataset is not None and len(self.dataset) > 0:
            n_avail = len(self.dataset)
            indices = np.random.choice(n_avail, min(self.n_data, n_avail), replace=False)
            data_x = self.dataset.data_x[indices].to(device)
            data_y = self.dataset.data_y[indices].to(device)
            batch['data_x'] = data_x
            batch['data_y'] = data_y
        else:
            batch['data_x'] = None
            batch['data_y'] = None

        return batch

    def get_current_counts(self):
        return {
            'n_interior': self.n_interior,
            'n_gauge': self.n_gauge,
            'n_interface': self.n_interface,
            'n_radiation': self.n_radiation,
            'n_data': self.n_data,
            'n_sym': self.n_sym,
        }


class MultiFreqBatchCache:
    def __init__(self, builder, n_freqs, resample_interval=10, device='cpu'):
        self.builder = builder
        self.n_freqs = n_freqs
        self.resample_interval = resample_interval
        self.device = device
        self._cache = {}
        self._epoch_count = 0

    def get_batch(self, freq_idx):
        return self._cache.get(freq_idx)

    def resample(self, epoch=None):
        if epoch is not None:
            self._epoch_count = epoch
        if self._epoch_count % self.resample_interval == 0:
            for freq_idx in range(self.n_freqs):
                self._cache[freq_idx] = self.builder.build_batch(
                    freq_idx, self.device
                )
        self._epoch_count += 1

    def invalidate(self):
        self._cache.clear()

    def get_cache_info(self):
        return {
            'n_cached': len(self._cache),
            'freq_indices': list(self._cache.keys()),
            'epoch_count': self._epoch_count,
        }


def create_default_layer_info():
    return [
        {'z_min': 0.0, 'z_max': 20.0, 'eps_r': 1.0, 'mu_r': 1.0, 'sigma': 0.01},
        {'z_min': 20.0, 'z_max': 60.0, 'eps_r': 1.0, 'mu_r': 1.0, 'sigma': 0.1},
        {'z_min': 60.0, 'z_max': 100.0, 'eps_r': 1.0, 'mu_r': 1.0, 'sigma': 0.001},
    ]


def create_default_config():
    layer_info = create_default_layer_info()
    source_pos = [0.0, 0.0, -25.0]
    rho_range = (0.0, 50.0)
    z_range = (0.0, 100.0)
    frequencies = [0.1, 100.0, 2000.0, 1e7, 3e10]

    return {
        'layer_info': layer_info,
        'source_pos': source_pos,
        'rho_range': rho_range,
        'z_range': z_range,
        'frequencies': frequencies,
    }


def create_synthetic_dataset(n_samples=1000, layer_info=None, source_pos=None,
                              rho_range=None, z_range=None, frequencies=None,
                              noise_level=0.0, seed=42):
    np.random.seed(seed)

    if layer_info is None:
        layer_info = create_default_layer_info()
    if source_pos is None:
        source_pos = [0.0, 0.0, -25.0]
    if rho_range is None:
        rho_range = (0.0, 50.0)
    if z_range is None:
        z_range = (0.0, 100.0)
    if frequencies is None:
        frequencies = [100.0]

    sampler = PhysicsSampler(
        layer_info=layer_info,
        source_pos=source_pos,
        rho_range=rho_range,
        z_range=z_range,
        frequencies=frequencies,
    )

    all_x = []
    all_y = []
    for freq_idx in range(len(frequencies)):
        points = sampler.sample_interior(n_samples // len(frequencies), freq_idx)
        all_x.append(points.numpy())

        n = points.shape[0]
        y = np.random.randn(n, 14) * 0.01
        all_y.append(y)

    data_x = np.vstack(all_x)
    data_y = np.vstack(all_y)

    if noise_level > 0:
        data_y += np.random.randn(*data_y.shape) * noise_level

    return LMGFDataset(data_x=data_x, data_y=data_y)


def merge_datasets(datasets):
    all_x = []
    all_y = []
    for ds in datasets:
        if len(ds) > 0:
            all_x.append(ds.data_x)
            all_y.append(ds.data_y)

    if len(all_x) == 0:
        return LMGFDataset()

    merged_x = torch.cat(all_x, dim=0)
    merged_y = torch.cat(all_y, dim=0)
    return LMGFDataset(data_x=merged_x.numpy(), data_y=merged_y.numpy())


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                  seed=42):
    n = len(dataset)
    if n == 0:
        return dataset, LMGFDataset(), LMGFDataset()

    np.random.seed(seed)
    indices = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_x = dataset.data_x[train_idx]
    train_y = dataset.data_y[train_idx]
    val_x = dataset.data_x[val_idx]
    val_y = dataset.data_y[val_idx]
    test_x = dataset.data_x[test_idx]
    test_y = dataset.data_y[test_idx]

    train_ds = LMGFDataset(data_x=train_x.numpy(), data_y=train_y.numpy())
    val_ds = LMGFDataset(data_x=val_x.numpy(), data_y=val_y.numpy())
    test_ds = LMGFDataset(data_x=test_x.numpy(), data_y=test_y.numpy())

    return train_ds, val_ds, test_ds
