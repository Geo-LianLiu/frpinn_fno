import numpy as np
import torch
from torch.utils.data import Dataset
import os
import csv
import json
import copy
from typing import Optional, List, Dict, Tuple, Union


class MediumProfile:
    def __init__(self, layers, z_min=0.0, z_max=800e3, nz=64,
                 eps_r_range=(1.0, 10.0), mu_r_range=(1.0, 1.0),
                 sigma_range=(1e-4, 1.0), name=None):
        self.layers = layers
        self.z_min = z_min
        self.z_max = z_max
        self.nz = nz
        self.eps_r_range = eps_r_range
        self.mu_r_range = mu_r_range
        self.sigma_range = sigma_range
        self.name = name

        self.layer_info = self._build_layer_info()
        self.interface_z = self._get_interfaces()

    def _build_layer_info(self):
        layer_info = []
        z_current = self.z_min
        for layer in self.layers:
            thickness = layer.get('thickness', 0)
            z_max_layer = z_current + thickness
            layer_info.append({
                'z_min': z_current,
                'z_max': z_max_layer,
                'eps_r': layer.get('eps_r', 1.0),
                'mu_r': layer.get('mu_r', 1.0),
                'sigma': layer.get('sigma', 0.01),
            })
            z_current = z_max_layer
        if len(layer_info) > 0:
            layer_info[-1]['z_max'] = self.z_max
        return layer_info

    def _get_interfaces(self):
        interfaces = []
        for i in range(len(self.layer_info) - 1):
            interfaces.append(self.layer_info[i]['z_max'])
        return interfaces

    def discretize(self, freq=None):
        z_grid = np.linspace(self.z_min, self.z_max, self.nz)

        eps_r_arr = np.zeros(self.nz)
        mu_r_arr = np.zeros(self.nz)
        sigma_arr = np.zeros(self.nz)

        for i, z in enumerate(z_grid):
            for layer in self.layer_info:
                if layer['z_min'] <= z < layer['z_max']:
                    eps_r_arr[i] = layer['eps_r']
                    mu_r_arr[i] = layer['mu_r']
                    sigma_arr[i] = layer['sigma']
                    break
            else:
                eps_r_arr[i] = self.layer_info[-1]['eps_r']
                mu_r_arr[i] = self.layer_info[-1]['mu_r']
                sigma_arr[i] = self.layer_info[-1]['sigma']

        eps_r_norm = (eps_r_arr - self.eps_r_range[0]) / (
            self.eps_r_range[1] - self.eps_r_range[0] + 1e-30
        )
        mu_r_norm = (mu_r_arr - self.mu_r_range[0]) / (
            self.mu_r_range[1] - self.mu_r_range[0] + 1e-30
        )

        sigma0 = 1e-8
        sigma_log = np.log10(sigma_arr + sigma0)
        sigma_log_min = np.log10(self.sigma_range[0] + sigma0)
        sigma_log_max = np.log10(self.sigma_range[1] + sigma0)
        sigma_norm = (sigma_log - sigma_log_min) / (
            sigma_log_max - sigma_log_min + 1e-30
        )

        profile = np.stack([eps_r_norm, mu_r_norm, sigma_norm], axis=1)

        if freq is not None:
            f_log = np.log10(freq)
            freq_channel = np.full((self.nz, 1), f_log)
            profile = np.concatenate([profile, freq_channel], axis=1)

        return profile.astype(np.float32)

    def discretize_raw(self):
        z_grid = np.linspace(self.z_min, self.z_max, self.nz)

        eps_r_arr = np.zeros(self.nz)
        mu_r_arr = np.zeros(self.nz)
        sigma_arr = np.zeros(self.nz)

        for i, z in enumerate(z_grid):
            for layer in self.layer_info:
                if layer['z_min'] <= z < layer['z_max']:
                    eps_r_arr[i] = layer['eps_r']
                    mu_r_arr[i] = layer['mu_r']
                    sigma_arr[i] = layer['sigma']
                    break
            else:
                eps_r_arr[i] = self.layer_info[-1]['eps_r']
                mu_r_arr[i] = self.layer_info[-1]['mu_r']
                sigma_arr[i] = self.layer_info[-1]['sigma']

        profile = np.stack([eps_r_arr, mu_r_arr, sigma_arr], axis=1)
        return profile.astype(np.float32), z_grid

    def get_layer_at_depth(self, z):
        for layer in self.layer_info:
            if layer['z_min'] <= z < layer['z_max']:
                return layer
        return self.layer_info[-1]

    def get_km2_at_depth(self, z, freq, mu0=4e-7 * np.pi, eps0=8.854e-12):
        layer = self.get_layer_at_depth(z)
        omega = 2 * np.pi * freq
        eps_r = layer['eps_r']
        mu_r = layer['mu_r']
        sigma = layer['sigma']
        eps_complex = eps_r * eps0 - 1j * sigma / omega
        km2 = omega ** 2 * mu_r * mu0 * eps_complex
        return km2

    def get_n_layers(self):
        return len(self.layer_info)

    def get_total_thickness(self):
        return self.z_max - self.z_min

    def get_layer_thicknesses(self):
        return [layer['z_max'] - layer['z_min'] for layer in self.layer_info]

    def get_conductivity_profile(self):
        z_grid = np.linspace(self.z_min, self.z_max, self.nz)
        sigma_arr = np.zeros(self.nz)
        for i, z in enumerate(z_grid):
            layer = self.get_layer_at_depth(z)
            sigma_arr[i] = layer['sigma']
        return sigma_arr, z_grid

    def get_permittivity_profile(self):
        z_grid = np.linspace(self.z_min, self.z_max, self.nz)
        eps_arr = np.zeros(self.nz)
        for i, z in enumerate(z_grid):
            layer = self.get_layer_at_depth(z)
            eps_arr[i] = layer['eps_r']
        return eps_arr, z_grid

    def to_dict(self):
        return {
            'name': self.name,
            'z_min': self.z_min,
            'z_max': self.z_max,
            'nz': self.nz,
            'eps_r_range': list(self.eps_r_range),
            'mu_r_range': list(self.mu_r_range),
            'sigma_range': list(self.sigma_range),
            'layers': self.layers,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            layers=d['layers'],
            z_min=d.get('z_min', 0.0),
            z_max=d.get('z_max', 800e3),
            nz=d.get('nz', 64),
            eps_r_range=tuple(d.get('eps_r_range', (1.0, 10.0))),
            mu_r_range=tuple(d.get('mu_r_range', (1.0, 1.0))),
            sigma_range=tuple(d.get('sigma_range', (1e-4, 1.0))),
            name=d.get('name', None),
        )

    def copy(self):
        return MediumProfile(
            layers=copy.deepcopy(self.layers),
            z_min=self.z_min,
            z_max=self.z_max,
            nz=self.nz,
            eps_r_range=self.eps_r_range,
            mu_r_range=self.mu_r_range,
            sigma_range=self.sigma_range,
            name=self.name,
        )

    def __repr__(self):
        n_layers = len(self.layer_info)
        name_str = f" '{self.name}'" if self.name else ""
        return f"MediumProfile{name_str}: {n_layers} layers, z=[{self.z_min:.0f}, {self.z_max:.0f}]"


class GradientMediumProfile(MediumProfile):
    def __init__(self, layers, z_min=0.0, z_max=800e3, nz=64,
                 eps_r_range=(1.0, 10.0), mu_r_range=(1.0, 1.0),
                 sigma_range=(1e-4, 1.0), name=None,
                 gradient_type='linear'):
        super().__init__(layers, z_min, z_max, nz,
                         eps_r_range, mu_r_range, sigma_range, name)
        self.gradient_type = gradient_type

    def discretize(self, freq=None):
        z_grid = np.linspace(self.z_min, self.z_max, self.nz)

        eps_r_arr = np.zeros(self.nz)
        mu_r_arr = np.zeros(self.nz)
        sigma_arr = np.zeros(self.nz)

        for i, z in enumerate(z_grid):
            for j, layer in enumerate(self.layer_info):
                if layer['z_min'] <= z < layer['z_max']:
                    progress = (z - layer['z_min']) / (layer['z_max'] - layer['z_min'] + 1e-30)
                    if j < len(self.layer_info) - 1:
                        next_layer = self.layer_info[j + 1]
                        if self.gradient_type == 'linear':
                            eps_r_arr[i] = layer['eps_r'] + progress * (next_layer['eps_r'] - layer['eps_r'])
                            mu_r_arr[i] = layer['mu_r'] + progress * (next_layer['mu_r'] - layer['mu_r'])
                            sigma_arr[i] = layer['sigma'] * (next_layer['sigma'] / (layer['sigma'] + 1e-30)) ** progress
                        elif self.gradient_type == 'exponential':
                            eps_r_arr[i] = layer['eps_r'] * (next_layer['eps_r'] / (layer['eps_r'] + 1e-30)) ** progress
                            mu_r_arr[i] = layer['mu_r']
                            sigma_arr[i] = layer['sigma'] * (next_layer['sigma'] / (layer['sigma'] + 1e-30)) ** progress
                        else:
                            eps_r_arr[i] = layer['eps_r']
                            mu_r_arr[i] = layer['mu_r']
                            sigma_arr[i] = layer['sigma']
                    else:
                        eps_r_arr[i] = layer['eps_r']
                        mu_r_arr[i] = layer['mu_r']
                        sigma_arr[i] = layer['sigma']
                    break
            else:
                eps_r_arr[i] = self.layer_info[-1]['eps_r']
                mu_r_arr[i] = self.layer_info[-1]['mu_r']
                sigma_arr[i] = self.layer_info[-1]['sigma']

        eps_r_norm = (eps_r_arr - self.eps_r_range[0]) / (
            self.eps_r_range[1] - self.eps_r_range[0] + 1e-30
        )
        mu_r_norm = (mu_r_arr - self.mu_r_range[0]) / (
            self.mu_r_range[1] - self.mu_r_range[0] + 1e-30
        )

        sigma0 = 1e-8
        sigma_log = np.log10(sigma_arr + sigma0)
        sigma_log_min = np.log10(self.sigma_range[0] + sigma0)
        sigma_log_max = np.log10(self.sigma_range[1] + sigma0)
        sigma_norm = (sigma_log - sigma_log_min) / (
            sigma_log_max - sigma_log_min + 1e-30
        )

        profile = np.stack([eps_r_norm, mu_r_norm, sigma_norm], axis=1)

        if freq is not None:
            f_log = np.log10(freq)
            freq_channel = np.full((self.nz, 1), f_log)
            profile = np.concatenate([profile, freq_channel], axis=1)

        return profile.astype(np.float32)


class AnisotropicMediumProfile(MediumProfile):
    def __init__(self, layers, z_min=0.0, z_max=800e3, nz=64,
                 eps_r_range=(1.0, 10.0), mu_r_range=(1.0, 1.0),
                 sigma_range=(1e-4, 1.0), name=None,
                 anisotropy_ratio=1.0):
        super().__init__(layers, z_min, z_max, nz,
                         eps_r_range, mu_r_range, sigma_range, name)
        self.anisotropy_ratio = anisotropy_ratio

    def discretize(self, freq=None):
        z_grid = np.linspace(self.z_min, self.z_max, self.nz)

        eps_r_h_arr = np.zeros(self.nz)
        eps_r_v_arr = np.zeros(self.nz)
        mu_r_arr = np.zeros(self.nz)
        sigma_h_arr = np.zeros(self.nz)
        sigma_v_arr = np.zeros(self.nz)

        for i, z in enumerate(z_grid):
            layer = self.get_layer_at_depth(z)
            eps_r_h_arr[i] = layer['eps_r']
            eps_r_v_arr[i] = layer['eps_r'] * self.anisotropy_ratio
            mu_r_arr[i] = layer['mu_r']
            sigma_h_arr[i] = layer['sigma']
            sigma_v_arr[i] = layer['sigma'] * self.anisotropy_ratio

        eps_r_h_norm = (eps_r_h_arr - self.eps_r_range[0]) / (
            self.eps_r_range[1] - self.eps_r_range[0] + 1e-30
        )
        eps_r_v_norm = (eps_r_v_arr - self.eps_r_range[0]) / (
            self.eps_r_range[1] - self.eps_r_range[0] + 1e-30
        )
        mu_r_norm = (mu_r_arr - self.mu_r_range[0]) / (
            self.mu_r_range[1] - self.mu_r_range[0] + 1e-30
        )

        sigma0 = 1e-8
        sigma_h_log = np.log10(sigma_h_arr + sigma0)
        sigma_v_log = np.log10(sigma_v_arr + sigma0)
        sigma_log_min = np.log10(self.sigma_range[0] + sigma0)
        sigma_log_max = np.log10(self.sigma_range[1] + sigma0)
        sigma_h_norm = (sigma_h_log - sigma_log_min) / (sigma_log_max - sigma_log_min + 1e-30)
        sigma_v_norm = (sigma_v_log - sigma_log_min) / (sigma_log_max - sigma_log_min + 1e-30)

        profile = np.stack([eps_r_h_norm, eps_r_v_norm, mu_r_norm,
                            sigma_h_norm, sigma_v_norm], axis=1)

        if freq is not None:
            f_log = np.log10(freq)
            freq_channel = np.full((self.nz, 1), f_log)
            profile = np.concatenate([profile, freq_channel], axis=1)

        return profile.astype(np.float32)


class MediumProfileAugmentor:
    def __init__(self, sigma_perturbation=0.1, eps_perturbation=0.05,
                 thickness_perturbation=0.1, n_perturbed_copies=5,
                 seed=42):
        self.sigma_perturbation = sigma_perturbation
        self.eps_perturbation = eps_perturbation
        self.thickness_perturbation = thickness_perturbation
        self.n_perturbed_copies = n_perturbed_copies
        self.seed = seed

    def perturb_sigma(self, profile, perturbation=None):
        if perturbation is None:
            perturbation = self.sigma_perturbation
        new_layers = []
        for layer in profile.layers:
            new_layer = copy.deepcopy(layer)
            sigma = new_layer.get('sigma', 0.01)
            log_sigma = np.log10(sigma + 1e-30)
            log_sigma += np.random.uniform(-perturbation, perturbation)
            new_layer['sigma'] = 10.0 ** log_sigma
            new_layers.append(new_layer)
        return MediumProfile(
            layers=new_layers,
            z_min=profile.z_min,
            z_max=profile.z_max,
            nz=profile.nz,
            eps_r_range=profile.eps_r_range,
            mu_r_range=profile.mu_r_range,
            sigma_range=profile.sigma_range,
        )

    def perturb_eps(self, profile, perturbation=None):
        if perturbation is None:
            perturbation = self.eps_perturbation
        new_layers = []
        for layer in profile.layers:
            new_layer = copy.deepcopy(layer)
            eps_r = new_layer.get('eps_r', 1.0)
            eps_r *= (1.0 + np.random.uniform(-perturbation, perturbation))
            eps_r = max(1.0, eps_r)
            new_layer['eps_r'] = eps_r
            new_layers.append(new_layer)
        return MediumProfile(
            layers=new_layers,
            z_min=profile.z_min,
            z_max=profile.z_max,
            nz=profile.nz,
            eps_r_range=profile.eps_r_range,
            mu_r_range=profile.mu_r_range,
            sigma_range=profile.sigma_range,
        )

    def perturb_thickness(self, profile, perturbation=None):
        if perturbation is None:
            perturbation = self.thickness_perturbation
        new_layers = []
        total_thickness = profile.z_max - profile.z_min
        remaining = total_thickness
        for i, layer in enumerate(profile.layers):
            new_layer = copy.deepcopy(layer)
            if i < len(profile.layers) - 1:
                thickness = new_layer.get('thickness', 100e3)
                thickness *= (1.0 + np.random.uniform(-perturbation, perturbation))
                thickness = max(1e3, thickness)
                thickness = min(thickness, remaining * 0.9)
                new_layer['thickness'] = thickness
                remaining -= thickness
            else:
                new_layer['thickness'] = remaining
            new_layers.append(new_layer)
        return MediumProfile(
            layers=new_layers,
            z_min=profile.z_min,
            z_max=profile.z_max,
            nz=profile.nz,
            eps_r_range=profile.eps_r_range,
            mu_r_range=profile.mu_r_range,
            sigma_range=profile.sigma_range,
        )

    def add_layer(self, profile, z_insert, sigma, eps_r=1.0, mu_r=1.0,
                  min_thickness=10e3):
        new_layers = []
        inserted = False
        z_current = profile.z_min
        for layer in profile.layers:
            thickness = layer.get('thickness', 100e3)
            z_layer_end = z_current + thickness
            if not inserted and z_insert < z_layer_end and z_insert > z_current + min_thickness:
                if z_layer_end - z_insert > min_thickness:
                    new_layers.append({
                        'thickness': z_insert - z_current,
                        'eps_r': layer.get('eps_r', 1.0),
                        'mu_r': layer.get('mu_r', 1.0),
                        'sigma': layer.get('sigma', 0.01),
                    })
                    new_layers.append({
                        'thickness': min_thickness,
                        'eps_r': eps_r,
                        'mu_r': mu_r,
                        'sigma': sigma,
                    })
                    new_layers.append({
                        'thickness': z_layer_end - z_insert - min_thickness,
                        'eps_r': layer.get('eps_r', 1.0),
                        'mu_r': layer.get('mu_r', 1.0),
                        'sigma': layer.get('sigma', 0.01),
                    })
                    inserted = True
                else:
                    new_layers.append(copy.deepcopy(layer))
            else:
                new_layers.append(copy.deepcopy(layer))
            z_current = z_layer_end
        return MediumProfile(
            layers=new_layers,
            z_min=profile.z_min,
            z_max=profile.z_max,
            nz=profile.nz,
            eps_r_range=profile.eps_r_range,
            mu_r_range=profile.mu_r_range,
            sigma_range=profile.sigma_range,
        )

    def remove_layer(self, profile, layer_idx):
        if layer_idx < 0 or layer_idx >= len(profile.layers):
            return profile.copy()
        new_layers = []
        for i, layer in enumerate(profile.layers):
            if i == layer_idx:
                continue
            new_layers.append(copy.deepcopy(layer))
        if len(new_layers) > 0:
            total_new = sum(l.get('thickness', 0) for l in new_layers[:-1])
            new_layers[-1]['thickness'] = profile.z_max - profile.z_min - total_new
        return MediumProfile(
            layers=new_layers,
            z_min=profile.z_min,
            z_max=profile.z_max,
            nz=profile.nz,
            eps_r_range=profile.eps_r_range,
            mu_r_range=profile.mu_r_range,
            sigma_range=profile.sigma_range,
        )

    def generate_perturbed_copies(self, profile, n_copies=None):
        if n_copies is None:
            n_copies = self.n_perturbed_copies
        np.random.seed(self.seed)
        copies = []
        for _ in range(n_copies):
            p = self.perturb_sigma(profile)
            p = self.perturb_eps(p)
            p = self.perturb_thickness(p)
            copies.append(p)
        return copies


class MediumProfileComparator:
    def __init__(self, nz=64, z_min=0.0, z_max=800e3):
        self.nz = nz
        self.z_min = z_min
        self.z_max = z_max

    def compute_profile_distance(self, profile1, profile2, metric='l2'):
        p1 = profile1.discretize()
        p2 = profile2.discretize()
        if p1.shape != p2.shape:
            nz_min = min(p1.shape[0], p2.shape[0])
            nc_min = min(p1.shape[1], p2.shape[1])
            p1 = p1[:nz_min, :nc_min]
            p2 = p2[:nz_min, :nc_min]

        if metric == 'l2':
            return float(np.sqrt(np.mean((p1 - p2) ** 2)))
        elif metric == 'l1':
            return float(np.mean(np.abs(p1 - p2)))
        elif metric == 'linf':
            return float(np.max(np.abs(p1 - p2)))
        elif metric == 'cosine':
            p1_flat = p1.flatten()
            p2_flat = p2.flatten()
            cos_sim = np.dot(p1_flat, p2_flat) / (
                np.linalg.norm(p1_flat) * np.linalg.norm(p2_flat) + 1e-30
            )
            return float(1.0 - cos_sim)
        else:
            return float(np.sqrt(np.mean((p1 - p2) ** 2)))

    def compute_interface_distance(self, profile1, profile2):
        if1 = profile1.interface_z
        if2 = profile2.interface_z
        if len(if1) != len(if2):
            return float('inf')
        if len(if1) == 0:
            return 0.0
        return float(np.sqrt(np.mean((np.array(if1) - np.array(if2)) ** 2)))

    def compute_layer_count_distance(self, profile1, profile2):
        return abs(profile1.get_n_layers() - profile2.get_n_layers())

    def compute_composite_distance(self, profile1, profile2,
                                    w_profile=0.5, w_interface=0.3,
                                    w_layers=0.2):
        d_profile = self.compute_profile_distance(profile1, profile2)
        d_interface = self.compute_interface_distance(profile1, profile2)
        d_layers = self.compute_layer_count_distance(profile1, profile2)
        max_interface = self.z_max - self.z_min
        d_interface_norm = d_interface / max_interface if max_interface > 0 else 0.0
        d_layers_norm = d_layers / 10.0
        return w_profile * d_profile + w_interface * d_interface_norm + w_layers * d_layers_norm

    def find_most_similar(self, target, candidates, metric='composite'):
        best_idx = -1
        best_dist = float('inf')
        for i, candidate in enumerate(candidates):
            if metric == 'composite':
                dist = self.compute_composite_distance(target, candidate)
            elif metric == 'l2':
                dist = self.compute_profile_distance(target, candidate, 'l2')
            elif metric == 'interface':
                dist = self.compute_interface_distance(target, candidate)
            else:
                dist = self.compute_profile_distance(target, candidate)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, best_dist

    def compute_distance_matrix(self, profiles, metric='l2'):
        n = len(profiles)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.compute_profile_distance(profiles[i], profiles[j], metric)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix


class MediumProfileValidator:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def validate_profile(self, profile):
        issues = []
        if len(profile.layers) == 0:
            issues.append('Profile has no layers')
            return {'valid': False, 'issues': issues}

        total_thickness = sum(l.get('thickness', 0) for l in profile.layers[:-1])
        expected_total = profile.z_max - profile.z_min
        if abs(total_thickness - expected_total) > 1.0:
            issues.append(
                f'Layer thicknesses sum to {total_thickness:.1f}, '
                f'expected {expected_total:.1f}'
            )

        for i, layer in enumerate(profile.layers):
            sigma = layer.get('sigma', 0.01)
            if sigma <= 0:
                issues.append(f'Layer {i}: sigma={sigma} <= 0')
            eps_r = layer.get('eps_r', 1.0)
            if eps_r < 1.0:
                issues.append(f'Layer {i}: eps_r={eps_r} < 1.0')
            mu_r = layer.get('mu_r', 1.0)
            if mu_r <= 0:
                issues.append(f'Layer {i}: mu_r={mu_r} <= 0')
            thickness = layer.get('thickness', 0)
            if i < len(profile.layers) - 1 and thickness <= 0:
                issues.append(f'Layer {i}: thickness={thickness} <= 0')

        for i in range(len(profile.layer_info) - 1):
            if abs(profile.layer_info[i]['z_max'] - profile.layer_info[i + 1]['z_min']) > self.eps:
                issues.append(
                    f'Gap between layer {i} and {i + 1}: '
                    f'{profile.layer_info[i]["z_max"]:.2f} != {profile.layer_info[i + 1]["z_min"]:.2f}'
                )

        discretized = profile.discretize()
        if np.any(np.isnan(discretized)):
            issues.append('Discretized profile contains NaN')
        if np.any(np.isinf(discretized)):
            issues.append('Discretized profile contains Inf')

        return {'valid': len(issues) == 0, 'issues': issues}

    def validate_library(self, library):
        results = {}
        all_valid = True
        for i, model in enumerate(library.models):
            result = self.validate_profile(model)
            results[i] = result
            if not result['valid']:
                all_valid = False
        return {'valid': all_valid, 'per_model': results}

    def check_profile_diversity(self, profiles, min_distance=0.01):
        comparator = MediumProfileComparator()
        n = len(profiles)
        duplicates = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = comparator.compute_profile_distance(profiles[i], profiles[j])
                if dist < min_distance:
                    duplicates.append((i, j, dist))
        return {
            'n_profiles': n,
            'n_duplicate_pairs': len(duplicates),
            'duplicates': duplicates,
            'diverse': len(duplicates) == 0,
        }


class StrataInterface:
    def __init__(self, strata_path=None, z_min=0.0, z_max=800e3, nz=64):
        self.strata_path = strata_path
        self.z_min = z_min
        self.z_max = z_max
        self.nz = nz
        self._raw_data = None

    def load_strata_csv(self, path=None):
        if path is None:
            path = self.strata_path
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Strata CSV file not found: {path}")

        data = []
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                data.append([float(v) for v in row])

        self._raw_data = np.array(data) if len(data) > 0 else None
        return self._raw_data

    def parse_strata_output(self, data=None):
        if data is None:
            data = self._raw_data
        if data is None:
            return None

        if data.shape[1] >= 3:
            z_vals = data[:, 0]
            sigma_vals = data[:, 1]
            eps_r_vals = data[:, 2] if data.shape[1] > 2 else np.ones_like(z_vals)
        else:
            return None

        return {
            'z': z_vals,
            'sigma': sigma_vals,
            'eps_r': eps_r_vals,
        }

    def strata_to_medium_profile(self, parsed_data=None, name=None):
        if parsed_data is None:
            parsed_data = self.parse_strata_output()
        if parsed_data is None:
            return None

        z_vals = parsed_data['z']
        sigma_vals = parsed_data['sigma']
        eps_r_vals = parsed_data['eps_r']

        layers = []
        current_sigma = sigma_vals[0]
        current_eps_r = eps_r_vals[0]
        z_start = z_vals[0]

        for i in range(1, len(z_vals)):
            if (abs(sigma_vals[i] - current_sigma) > 1e-6 or
                    abs(eps_r_vals[i] - current_eps_r) > 1e-6):
                layers.append({
                    'thickness': z_vals[i] - z_start,
                    'eps_r': float(current_eps_r),
                    'mu_r': 1.0,
                    'sigma': float(current_sigma),
                })
                current_sigma = sigma_vals[i]
                current_eps_r = eps_r_vals[i]
                z_start = z_vals[i]

        if z_start < z_vals[-1]:
            layers.append({
                'thickness': z_vals[-1] - z_start,
                'eps_r': float(current_eps_r),
                'mu_r': 1.0,
                'sigma': float(current_sigma),
            })

        if len(layers) > 0:
            total = sum(l['thickness'] for l in layers[:-1])
            layers[-1]['thickness'] = self.z_max - self.z_min - total

        sigma_min = min(l['sigma'] for l in layers) if layers else 1e-4
        sigma_max = max(l['sigma'] for l in layers) if layers else 1.0
        eps_min = min(l['eps_r'] for l in layers) if layers else 1.0
        eps_max = max(l['eps_r'] for l in layers) if layers else 10.0

        return MediumProfile(
            layers=layers,
            z_min=self.z_min,
            z_max=self.z_max,
            nz=self.nz,
            eps_r_range=(min(1.0, eps_min * 0.9), max(10.0, eps_max * 1.1)),
            mu_r_range=(1.0, 1.0),
            sigma_range=(min(1e-4, sigma_min * 0.5), max(1.0, sigma_max * 2.0)),
            name=name or 'Strata',
        )

    def load_multiple_strata(self, paths, names=None):
        if names is None:
            names = [f'Strata_{i}' for i in range(len(paths))]
        profiles = []
        for path, name in zip(paths, names):
            self.load_strata_csv(path)
            parsed = self.parse_strata_output()
            profile = self.strata_to_medium_profile(parsed, name=name)
            if profile is not None:
                profiles.append(profile)
        return profiles


class MediumModelLibrary:
    def __init__(self, z_min=0.0, z_max=800e3, nz=64,
                 eps_r_range=(1.0, 10.0), mu_r_range=(1.0, 1.0),
                 sigma_range=(1e-4, 1.0)):
        self.z_min = z_min
        self.z_max = z_max
        self.nz = nz
        self.eps_r_range = eps_r_range
        self.mu_r_range = mu_r_range
        self.sigma_range = sigma_range
        self.models = []

    def add_uniform_model(self, sigma, eps_r=1.0, mu_r=1.0, name=None):
        layers = [{
            'thickness': self.z_max - self.z_min,
            'eps_r': eps_r,
            'mu_r': mu_r,
            'sigma': sigma,
        }]
        self.models.append(MediumProfile(
            layers, self.z_min, self.z_max, self.nz,
            self.eps_r_range, self.mu_r_range, self.sigma_range,
            name=name or f'Uniform_sigma{sigma:.4f}'
        ))

    def add_h_type_model(self, layer_params, name=None):
        layers = []
        for params in layer_params:
            layers.append({
                'thickness': params.get('thickness', 100e3),
                'eps_r': params.get('eps_r', 1.0),
                'mu_r': params.get('mu_r', 1.0),
                'sigma': params.get('sigma', 0.01),
            })
        self.models.append(MediumProfile(
            layers, self.z_min, self.z_max, self.nz,
            self.eps_r_range, self.mu_r_range, self.sigma_range,
            name=name or f'H-type_{len(layer_params)}L'
        ))

    def add_hk_type_model(self, layer_params, name=None):
        layers = []
        for params in layer_params:
            layers.append({
                'thickness': params.get('thickness', 100e3),
                'eps_r': params.get('eps_r', 1.0),
                'mu_r': params.get('mu_r', 1.0),
                'sigma': params.get('sigma', 0.01),
            })
        self.models.append(MediumProfile(
            layers, self.z_min, self.z_max, self.nz,
            self.eps_r_range, self.mu_r_range, self.sigma_range,
            name=name or f'HK-type_{len(layer_params)}L'
        ))

    def add_gradient_model(self, layer_params, gradient_type='linear', name=None):
        layers = []
        for params in layer_params:
            layers.append({
                'thickness': params.get('thickness', 100e3),
                'eps_r': params.get('eps_r', 1.0),
                'mu_r': params.get('mu_r', 1.0),
                'sigma': params.get('sigma', 0.01),
            })
        self.models.append(GradientMediumProfile(
            layers, self.z_min, self.z_max, self.nz,
            self.eps_r_range, self.mu_r_range, self.sigma_range,
            name=name or f'Gradient-{gradient_type}_{len(layer_params)}L',
            gradient_type=gradient_type,
        ))

    def add_anisotropic_model(self, layer_params, anisotropy_ratio=1.2, name=None):
        layers = []
        for params in layer_params:
            layers.append({
                'thickness': params.get('thickness', 100e3),
                'eps_r': params.get('eps_r', 1.0),
                'mu_r': params.get('mu_r', 1.0),
                'sigma': params.get('sigma', 0.01),
            })
        self.models.append(AnisotropicMediumProfile(
            layers, self.z_min, self.z_max, self.nz,
            self.eps_r_range, self.mu_r_range, self.sigma_range,
            name=name or f'Aniso-{anisotropy_ratio:.1f}_{len(layer_params)}L',
            anisotropy_ratio=anisotropy_ratio,
        ))

    def add_profile(self, profile):
        self.models.append(profile)

    def generate_perturbed_library(self, base_type='all', n_models=30,
                                    seed=42):
        np.random.seed(seed)

        if base_type in ['all', 'uniform']:
            for _ in range(n_models // 3):
                sigma = 10 ** np.random.uniform(
                    np.log10(self.sigma_range[0]),
                    np.log10(self.sigma_range[1])
                )
                self.add_uniform_model(sigma)

        if base_type in ['all', 'h_type']:
            for _ in range(n_models // 3):
                n_layers = np.random.randint(3, 6)
                layer_params = []
                z_remaining = self.z_max - self.z_min
                for j in range(n_layers):
                    if j < n_layers - 1:
                        thickness = np.random.uniform(
                            0.1 * z_remaining / n_layers,
                            2.0 * z_remaining / n_layers
                        )
                    else:
                        thickness = z_remaining
                    sigma = 10 ** np.random.uniform(
                        np.log10(self.sigma_range[0]),
                        np.log10(self.sigma_range[1])
                    )
                    layer_params.append({
                        'thickness': thickness,
                        'sigma': sigma,
                    })
                    z_remaining -= thickness
                self.add_h_type_model(layer_params)

        if base_type in ['all', 'hk_type']:
            for _ in range(n_models - len(self.models)):
                n_layers = np.random.randint(3, 6)
                layer_params = []
                z_remaining = self.z_max - self.z_min
                for j in range(n_layers):
                    if j < n_layers - 1:
                        thickness = np.random.uniform(
                            0.1 * z_remaining / n_layers,
                            2.0 * z_remaining / n_layers
                        )
                    else:
                        thickness = z_remaining
                    sigma = 10 ** np.random.uniform(
                        np.log10(self.sigma_range[0]),
                        np.log10(self.sigma_range[1])
                    )
                    layer_params.append({
                        'thickness': thickness,
                        'sigma': sigma,
                    })
                    z_remaining -= thickness
                self.add_hk_type_model(layer_params)

        return self.models

    def generate_augmented_library(self, base_profiles, augmentor=None,
                                    n_copies_per_profile=3):
        if augmentor is None:
            augmentor = MediumProfileAugmentor()
        for profile in base_profiles:
            self.add_profile(profile)
            copies = augmentor.generate_perturbed_copies(profile, n_copies_per_profile)
            for copy_profile in copies:
                self.add_profile(copy_profile)
        return self.models

    def get_profile(self, idx):
        if 0 <= idx < len(self.models):
            return self.models[idx]
        return None

    def get_all_profiles(self):
        return self.models

    def get_n_models(self):
        return len(self.models)

    def sample_random_profiles(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(len(self.models), min(n, len(self.models)), replace=False)
        return [self.models[i] for i in indices]

    def to_dict_list(self):
        return [m.to_dict() for m in self.models]

    def save_library(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'z_min': self.z_min,
            'z_max': self.z_max,
            'nz': self.nz,
            'eps_r_range': list(self.eps_r_range),
            'mu_r_range': list(self.mu_r_range),
            'sigma_range': list(self.sigma_range),
            'models': self.to_dict_list(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_library(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        library = cls(
            z_min=data.get('z_min', 0.0),
            z_max=data.get('z_max', 800e3),
            nz=data.get('nz', 64),
            eps_r_range=tuple(data.get('eps_r_range', (1.0, 10.0))),
            mu_r_range=tuple(data.get('mu_r_range', (1.0, 1.0))),
            sigma_range=tuple(data.get('sigma_range', (1e-4, 1.0))),
        )
        for model_dict in data.get('models', []):
            library.add_profile(MediumProfile.from_dict(model_dict))
        return library

    def get_library_statistics(self):
        if len(self.models) == 0:
            return {'n_models': 0}
        n_layers_list = [m.get_n_layers() for m in self.models]
        sigma_list = []
        eps_list = []
        for m in self.models:
            for layer in m.layer_info:
                sigma_list.append(layer['sigma'])
                eps_list.append(layer['eps_r'])
        return {
            'n_models': len(self.models),
            'n_layers': {
                'min': min(n_layers_list),
                'max': max(n_layers_list),
                'mean': np.mean(n_layers_list),
            },
            'sigma': {
                'min': min(sigma_list),
                'max': max(sigma_list),
                'mean': np.mean(sigma_list),
                'std': np.std(sigma_list),
            },
            'eps_r': {
                'min': min(eps_list),
                'max': max(eps_list),
                'mean': np.mean(eps_list),
            },
        }


class MultiMediumDataset(Dataset):
    def __init__(self, csv_path=None):
        self.samples = []
        self.medium_profiles = []
        self.data_x = torch.empty(0, 8)
        self.data_y = torch.empty(0, 14)
        self.medium_indices = torch.empty(0, dtype=torch.long)

        if csv_path is not None and os.path.exists(csv_path):
            self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        data = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                data.append([float(v) for v in row])
        if len(data) > 0:
            data = np.array(data)
            self.data_x = torch.tensor(data[:, :8], dtype=torch.float32)
            self.data_y = torch.tensor(data[:, 8:22], dtype=torch.float32)
            if data.shape[1] > 22:
                self.medium_indices = torch.tensor(
                    data[:, 22].astype(int), dtype=torch.long
                )
            else:
                self.medium_indices = torch.zeros(len(data), dtype=torch.long)
        else:
            self.data_x = torch.empty(0, 8)
            self.data_y = torch.empty(0, 14)
            self.medium_indices = torch.empty(0, dtype=torch.long)

    def add_medium_data(self, medium_idx, x_data, y_data):
        if not hasattr(self, 'data_x') or len(self.data_x) == 0:
            self.data_x = torch.tensor(x_data, dtype=torch.float32)
            self.data_y = torch.tensor(y_data, dtype=torch.float32)
            self.medium_indices = torch.full(
                (len(x_data),), medium_idx, dtype=torch.long
            )
        else:
            new_x = torch.tensor(x_data, dtype=torch.float32)
            new_y = torch.tensor(y_data, dtype=torch.float32)
            new_idx = torch.full((len(x_data),), medium_idx, dtype=torch.long)
            self.data_x = torch.cat([self.data_x, new_x], dim=0)
            self.data_y = torch.cat([self.data_y, new_y], dim=0)
            self.medium_indices = torch.cat(
                [self.medium_indices, new_idx], dim=0
            )

    def get_data_for_medium(self, medium_idx):
        if not hasattr(self, 'data_x') or len(self.data_x) == 0:
            return None, None
        mask = self.medium_indices == medium_idx
        if not mask.any():
            return None, None
        return self.data_x[mask], self.data_y[mask]

    def get_n_mediums(self):
        if not hasattr(self, 'medium_indices') or len(self.medium_indices) == 0:
            return 0
        return len(torch.unique(self.medium_indices))

    def get_medium_counts(self):
        if not hasattr(self, 'medium_indices') or len(self.medium_indices) == 0:
            return {}
        unique, counts = torch.unique(self.medium_indices, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def __len__(self):
        return len(self.data_x) if hasattr(self, 'data_x') else 0

    def __getitem__(self, idx):
        return (self.data_x[idx], self.data_y[idx], self.medium_indices[idx])


def create_default_medium_library(z_min=0.0, z_max=800e3, nz=64,
                                  eps_r_range=(1.0, 10.0),
                                  mu_r_range=(1.0, 1.0),
                                  sigma_range=(1e-4, 1.0),
                                  n_perturbed=30, seed=42,
                                  base_models=None):
    library = MediumModelLibrary(
        z_min=z_min, z_max=z_max, nz=nz,
        eps_r_range=eps_r_range,
        mu_r_range=mu_r_range,
        sigma_range=sigma_range,
    )

    if base_models is not None:
        for bm in base_models:
            bm_type = bm.get('type', 'uniform')
            if bm_type == 'uniform':
                library.add_uniform_model(
                    sigma=bm.get('sigma', 0.01),
                    eps_r=bm.get('eps_r', 1.0),
                    mu_r=bm.get('mu_r', 1.0),
                    name=bm.get('name', None),
                )
            elif bm_type == 'h_type':
                library.add_h_type_model(
                    bm.get('layers', []),
                    name=bm.get('name', None),
                )
            elif bm_type == 'hk_type':
                library.add_hk_type_model(
                    bm.get('layers', []),
                    name=bm.get('name', None),
                )
    else:
        library.add_uniform_model(sigma=0.01, name='Uniform_0.01')
        library.add_uniform_model(sigma=0.1, name='Uniform_0.1')

        library.add_h_type_model([
            {'thickness': 200e3, 'sigma': 0.01},
            {'thickness': 100e3, 'sigma': 0.1},
            {'thickness': 500e3, 'sigma': 0.001},
        ], name='H-type_3L_base')

        library.add_hk_type_model([
            {'thickness': 100e3, 'sigma': 0.01},
            {'thickness': 150e3, 'sigma': 0.1},
            {'thickness': 100e3, 'sigma': 0.001},
            {'thickness': 450e3, 'sigma': 0.05},
        ], name='HK-type_4L_base')

    library.generate_perturbed_library(base_type='all', n_models=n_perturbed, seed=seed)

    return library


def create_default_fno_config():
    return {
        'z_min': 0.0,
        'z_max': 800e3,
        'nz': 64,
        'source_pos': [0.0, 0.0, -10e3],
        'rho_range': (0.0, 800e3),
        'z_range': (0.0, 800e3),
        'frequencies': [0.0001, 0.1, 1.0, 10.0, 100.0],
        'eps_r_range': (1.0, 10.0),
        'mu_r_range': (1.0, 1.0),
        'sigma_range': (1e-4, 1.0),
        'n_interior': 3000,
        'n_gauge': 1500,
        'n_interface': 500,
        'n_radiation': 300,
        'n_data': 800,
        'n_sym': 500,
    }


def create_classic_geophysical_models(z_min=0.0, z_max=800e3, nz=64):
    library = MediumModelLibrary(
        z_min=z_min, z_max=z_max, nz=nz,
        eps_r_range=(1.0, 10.0),
        mu_r_range=(1.0, 1.0),
        sigma_range=(1e-4, 1.0),
    )

    library.add_uniform_model(sigma=0.001, name='Half-space_0.001')
    library.add_uniform_model(sigma=0.01, name='Half-space_0.01')
    library.add_uniform_model(sigma=0.1, name='Half-space_0.1')

    library.add_h_type_model([
        {'thickness': 100e3, 'sigma': 0.01},
        {'thickness': 700e3, 'sigma': 0.001},
    ], name='H-type_2L_conductor')

    library.add_h_type_model([
        {'thickness': 200e3, 'sigma': 0.001},
        {'thickness': 600e3, 'sigma': 0.01},
    ], name='H-type_2L_resistor')

    library.add_h_type_model([
        {'thickness': 100e3, 'sigma': 0.01},
        {'thickness': 200e3, 'sigma': 0.1},
        {'thickness': 500e3, 'sigma': 0.001},
    ], name='H-type_3L')

    library.add_hk_type_model([
        {'thickness': 100e3, 'sigma': 0.001},
        {'thickness': 100e3, 'sigma': 0.1},
        {'thickness': 600e3, 'sigma': 0.01},
    ], name='HK-type_3L')

    library.add_hk_type_model([
        {'thickness': 50e3, 'sigma': 0.01},
        {'thickness': 100e3, 'sigma': 0.1},
        {'thickness': 150e3, 'sigma': 0.001},
        {'thickness': 500e3, 'sigma': 0.05},
    ], name='HK-type_4L')

    library.add_h_type_model([
        {'thickness': 50e3, 'sigma': 0.01},
        {'thickness': 100e3, 'sigma': 0.1},
        {'thickness': 150e3, 'sigma': 0.001},
        {'thickness': 200e3, 'sigma': 0.05},
        {'thickness': 300e3, 'sigma': 0.01},
    ], name='H-type_5L')

    return library
