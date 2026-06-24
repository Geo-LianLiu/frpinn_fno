import os
import math
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm


class PlotConfig:
    def __init__(self, figsize=(12, 8), dpi=150, style='seaborn-v0_8-whitegrid',
                 font_size=12, title_size=14, label_size=11, tick_size=10,
                 legend_size=10, linewidth=1.5, alpha=0.8,
                 color_palette=None, save_format='png'):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.font_size = font_size
        self.title_size = title_size
        self.label_size = label_size
        self.tick_size = tick_size
        self.legend_size = legend_size
        self.linewidth = linewidth
        self.alpha = alpha
        self.save_format = save_format

        if color_palette is None:
            self.color_palette = [
                '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
                '#2980b9', '#27ae60', '#f1c40f', '#8e44ad', '#d35400',
            ]
        else:
            self.color_palette = color_palette

    def apply_style(self):
        available_styles = plt.style.available
        if self.style in available_styles:
            plt.style.use(self.style)
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            'figure.figsize': self.figsize,
            'figure.dpi': self.dpi,
        })


class LossConvergencePlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(16, 10))
        else:
            self.config = config

    def plot_loss_components(self, loss_history, save_path=None,
                             loss_names=None, loss_labels=None,
                             title='Training Loss Convergence',
                             log_scale=True, show_weights=False):
        self.config.apply_style()

        if loss_names is None:
            loss_names = ['pde', 'gauge', 'bc', 'data', 'sym']
        if loss_labels is None:
            loss_labels = ['PDE Residual', 'Lorenz Gauge', 'Boundary Cond.',
                           'Data Fitting', 'Symmetry']

        n_components = len(loss_names)
        n_cols = 3
        n_rows = (n_components + 1 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        epochs = list(range(len(loss_history)))
        colors = self.config.color_palette[:n_components]

        for i, (name, label, color) in enumerate(zip(loss_names, loss_labels, colors)):
            ax = axes[i]
            values = [h.get(name, 0.0) for h in loss_history]
            if log_scale:
                ax.semilogy(epochs, values, color=color,
                            linewidth=self.config.linewidth, alpha=self.config.alpha)
            else:
                ax.plot(epochs, values, color=color,
                        linewidth=self.config.linewidth, alpha=self.config.alpha)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

            if show_weights:
                w_key = f'w_{name}'
                w_values = [h.get(w_key, None) for h in loss_history]
                valid_w = [(e, w) for e, w in zip(epochs, w_values) if w is not None]
                if valid_w:
                    w_epochs, w_vals = zip(*valid_w)
                    ax2 = ax.twinx()
                    ax2.plot(w_epochs, w_vals, color='gray', linestyle='--',
                             linewidth=0.8, alpha=0.6)
                    ax2.set_ylabel('Weight', fontsize=9, color='gray')
                    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

        ax_total = axes[n_components]
        total_values = [h.get('total', 0.0) for h in loss_history]
        if log_scale:
            ax_total.semilogy(epochs, total_values, color='#2c3e50',
                              linewidth=2.0)
        else:
            ax_total.plot(epochs, total_values, color='#2c3e50', linewidth=2.0)
        ax_total.set_xlabel('Epoch')
        ax_total.set_ylabel('Loss')
        ax_total.set_title('Total Loss')
        ax_total.grid(True, alpha=0.3)

        for j in range(n_components + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight',
                        format=self.config.save_format)
        plt.close()
        return fig

    def plot_loss_comparison(self, loss_histories_dict, save_path=None,
                             title='Loss Comparison Across Methods'):
        self.config.apply_style()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        loss_names = ['pde', 'gauge', 'bc', 'data', 'sym', 'total']
        loss_labels = ['PDE Residual', 'Lorenz Gauge', 'Boundary Cond.',
                       'Data Fitting', 'Symmetry', 'Total Loss']

        for idx, (name, label) in enumerate(zip(loss_names, loss_labels)):
            ax = axes[idx]
            for i, (method_name, history) in enumerate(loss_histories_dict.items()):
                values = [h.get(name, 0.0) for h in history]
                epochs = list(range(len(values)))
                color = self.config.color_palette[i % len(self.config.color_palette)]
                ax.semilogy(epochs, values, color=color, label=method_name,
                            linewidth=self.config.linewidth, alpha=self.config.alpha)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=self.config.legend_size - 1)

        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_loss_weights_evolution(self, loss_history, save_path=None,
                                    title='Adaptive Loss Weights Evolution'):
        self.config.apply_style()

        weight_names = ['w_pde', 'w_gauge', 'w_bc', 'w_data', 'w_sym']
        weight_labels = ['PDE Weight', 'Gauge Weight', 'BC Weight',
                         'Data Weight', 'Sym Weight']

        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = list(range(len(loss_history)))

        for name, label, color in zip(weight_names, weight_labels,
                                       self.config.color_palette[:5]):
            values = [h.get(name, None) for h in loss_history]
            valid = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid:
                e_vals, v_vals = zip(*valid)
                ax.plot(e_vals, v_vals, color=color, label=label,
                        linewidth=self.config.linewidth, alpha=self.config.alpha)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title(title)
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_loss_ratio_pie(self, loss_history, epoch_idx=-1, save_path=None,
                            title='Loss Component Ratios'):
        self.config.apply_style()

        if abs(epoch_idx) > len(loss_history):
            epoch_idx = -1

        loss_dict = loss_history[epoch_idx]
        names = ['pde', 'gauge', 'bc', 'data', 'sym']
        labels = ['PDE', 'Gauge', 'BC', 'Data', 'Sym']
        values = [max(loss_dict.get(n, 0.0), 1e-30) for n in names]

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = self.config.color_palette[:5]
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85
        )
        for text in autotexts:
            text.set_fontsize(10)
        ax.set_title(title)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class FieldComponentPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(16, 12))
        else:
            self.config = config

        self.component_names = [
            'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
            'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
            'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
            'Kphi_Re', 'Kphi_Im'
        ]

    def plot_component_comparison(self, pred, target, coordinates=None,
                                  save_path=None, n_components=14,
                                  title='Predicted vs Reference Field Components'):
        self.config.apply_style()

        n_cols = 4
        n_rows = (n_components + n_cols - 1) // n_cols

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for i in range(min(n_components, pred.shape[-1])):
            ax = axes[i]
            pred_i = pred[:, i]
            target_i = target[:, i]

            sort_idx = np.argsort(target_i)
            x_axis = np.arange(len(target_i))

            ax.plot(x_axis, target_i[sort_idx], 'b-', label='Reference',
                    linewidth=1.0, alpha=0.7)
            ax.plot(x_axis, pred_i[sort_idx], 'r--', label='Predicted',
                    linewidth=1.0, alpha=0.7)

            name = self.component_names[i] if i < len(self.component_names) else f'Comp_{i}'
            ax.set_title(name, fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        for j in range(n_components, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_component_scatter(self, pred, target, save_path=None,
                               title='Predicted vs Reference Scatter'):
        self.config.apply_style()

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        n_components = min(14, pred.shape[-1])
        n_cols = 4
        n_rows = (n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for i in range(n_components):
            ax = axes[i]
            pred_i = pred[:, i]
            target_i = target[:, i]

            ax.scatter(target_i, pred_i, s=1, alpha=0.3, c=self.config.color_palette[0])

            vmin = min(target_i.min(), pred_i.min())
            vmax = max(target_i.max(), pred_i.max())
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=0.8, alpha=0.5)

            name = self.component_names[i] if i < len(self.component_names) else f'Comp_{i}'
            ax.set_title(name, fontsize=10)
            ax.set_xlabel('Reference', fontsize=8)
            ax.set_ylabel('Predicted', fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.2)

        for j in range(n_components, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_component_error_bar(self, component_metrics, save_path=None,
                                 metric='rel_l2',
                                 title='Component-wise Error'):
        self.config.apply_style()

        names = list(component_metrics.keys())
        values = [component_metrics[n].get(metric, 0.0) for n in names]

        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(names))
        colors = [self.config.color_palette[i % len(self.config.color_palette)]
                  for i in range(len(names))]

        bars = ax.bar(x_pos, values, color=colors, alpha=self.config.alpha)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=7)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class SpatialErrorPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(14, 6))
        else:
            self.config = config

    def plot_radial_error_profile(self, radial_profile, save_path=None,
                                  title='Radial Error Profile'):
        self.config.apply_style()

        rho_centers = [v['rho_center'] for v in radial_profile.values()]
        mean_abs = [v['mean_abs_error'] for v in radial_profile.values()]
        mean_rel = [v['mean_rel_error'] for v in radial_profile.values()]
        max_abs = [v['max_abs_error'] for v in radial_profile.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.semilogy(rho_centers, mean_abs, 'b-o', markersize=4,
                     label='Mean Abs Error', linewidth=self.config.linewidth)
        ax1.semilogy(rho_centers, max_abs, 'r--s', markersize=4,
                     label='Max Abs Error', linewidth=self.config.linewidth)
        ax1.set_xlabel('Radial Distance (rho)')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Absolute Error vs Radial Distance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(rho_centers, mean_rel, 'b-o', markersize=4,
                     label='Mean Rel Error', linewidth=self.config.linewidth)
        ax2.set_xlabel('Radial Distance (rho)')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Relative Error vs Radial Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_depth_error_profile(self, depth_profile, save_path=None,
                                 title='Depth Error Profile'):
        self.config.apply_style()

        z_centers = [v['z_center'] for v in depth_profile.values()]
        mean_abs = [v['mean_abs_error'] for v in depth_profile.values()]
        mean_rel = [v['mean_rel_error'] for v in depth_profile.values()]
        max_abs = [v['max_abs_error'] for v in depth_profile.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.semilogy(z_centers, mean_abs, 'b-o', markersize=4,
                     label='Mean Abs Error', linewidth=self.config.linewidth)
        ax1.semilogy(z_centers, max_abs, 'r--s', markersize=4,
                     label='Max Abs Error', linewidth=self.config.linewidth)
        ax1.set_xlabel('Depth (z)')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Absolute Error vs Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(z_centers, mean_rel, 'b-o', markersize=4,
                     label='Mean Rel Error', linewidth=self.config.linewidth)
        ax2.set_xlabel('Depth (z)')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Relative Error vs Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_error_heatmap_2d(self, pred, target, x_coords, z_coords,
                              save_path=None, component_idx=0,
                              title='2D Error Heatmap'):
        self.config.apply_style()

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        error = np.abs(pred[:, component_idx] - target[:, component_idx])

        fig, ax = plt.subplots(figsize=(10, 8))

        x_unique = np.sort(np.unique(x_coords))
        z_unique = np.sort(np.unique(z_coords))

        if len(x_unique) > 1 and len(z_unique) > 1:
            from scipy.interpolate import griddata
            xi = np.linspace(x_unique.min(), x_unique.max(), 100)
            zi = np.linspace(z_unique.min(), z_unique.max(), 100)
            Xi, Zi = np.meshgrid(xi, zi)
            Ei = griddata((x_coords, z_coords), error, (Xi, Zi), method='cubic')

            im = ax.pcolormesh(Xi, Zi, Ei, cmap='hot_r', shading='auto')
            plt.colorbar(im, ax=ax, label='Absolute Error')
        else:
            scatter = ax.scatter(x_coords, z_coords, c=error, cmap='hot_r', s=10)
            plt.colorbar(scatter, ax=ax, label='Absolute Error')

        ax.set_xlabel('x / rho')
        ax.set_ylabel('z')
        comp_name = f'Component {component_idx}'
        ax.set_title(f'{title} - {comp_name}')

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class FrequencyAnalysisPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(12, 6))
        else:
            self.config = config

    def plot_frequency_error(self, freq_results, save_path=None,
                             metric='rel_l2',
                             title='Error vs Frequency'):
        self.config.apply_style()

        freqs = []
        errors = []
        for freq, res in freq_results.items():
            if isinstance(res, dict) and 'overall_rel_l2' in res:
                freqs.append(res.get('freq_hz', freq))
                errors.append(res['overall_rel_l2'])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogy(freqs, errors, 'bo-', markersize=8,
                    linewidth=self.config.linewidth)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'Relative L2 Error')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if len(freqs) > 1:
            coeffs = np.polyfit(np.log10(freqs), np.log10(errors), 1)
            fit_line = 10 ** (coeffs[0] * np.log10(freqs) + coeffs[1])
            ax.semilogy(freqs, fit_line, 'r--', alpha=0.5,
                        label=f'Slope: {coeffs[0]:.2f}')
            ax.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_component_frequency_heatmap(self, freq_results, save_path=None,
                                         metric='rel_l2',
                                         title='Component Error vs Frequency'):
        self.config.apply_style()

        component_names = list(list(freq_results.values())[0]['component_metrics'].keys())
        freqs = sorted([res['freq_hz'] for res in freq_results.values()])

        error_matrix = np.zeros((len(component_names), len(freqs)))
        for j, (freq_key, res) in enumerate(freq_results.items()):
            for i, name in enumerate(component_names):
                error_matrix[i, j] = res['component_metrics'][name].get(metric, 0.0)

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(error_matrix, aspect='auto', cmap='YlOrRd',
                        norm=LogNorm(vmin=max(error_matrix.min(), 1e-10),
                                     vmax=error_matrix.max()))
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([f'{f:.0f}' for f in freqs], rotation=45, fontsize=8)
        ax.set_yticks(range(len(component_names)))
        ax.set_yticklabels(component_names, fontsize=8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Component')
        ax.set_title(title)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class MediumProfilePlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(14, 8))
        else:
            self.config = config

    def plot_medium_profile(self, medium_profile_np, z_values=None,
                            save_path=None, title='Medium Profile',
                            param_names=None):
        self.config.apply_style()

        if isinstance(medium_profile_np, torch.Tensor):
            medium_profile_np = medium_profile_np.detach().cpu().numpy()

        if z_values is None:
            z_values = np.arange(medium_profile_np.shape[0])

        if param_names is None:
            param_names = ['eps_r', 'mu_r', 'sigma', 'freq_weight']

        n_params = medium_profile_np.shape[1]
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 8))
        if n_params == 1:
            axes = [axes]

        for i in range(n_params):
            ax = axes[i]
            ax.plot(medium_profile_np[:, i], z_values,
                    color=self.config.color_palette[i],
                    linewidth=self.config.linewidth)
            name = param_names[i] if i < len(param_names) else f'Param {i}'
            ax.set_xlabel(name)
            ax.set_ylabel('Depth (z)')
            ax.set_title(name)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_medium_comparison(self, profiles_dict, save_path=None,
                               title='Medium Profile Comparison'):
        self.config.apply_style()

        n_profiles = len(profiles_dict)
        fig, axes = plt.subplots(1, n_profiles, figsize=(5 * n_profiles, 8))
        if n_profiles == 1:
            axes = [axes]

        for idx, (name, profile) in enumerate(profiles_dict.items()):
            ax = axes[idx]
            if isinstance(profile, torch.Tensor):
                profile = profile.detach().cpu().numpy()

            z_values = np.arange(profile.shape[0])
            for j in range(profile.shape[1]):
                ax.plot(profile[:, j], z_values,
                        color=self.config.color_palette[j],
                        linewidth=self.config.linewidth,
                        label=['eps_r', 'mu_r', 'sigma', 'freq'][j] if j < 4 else f'ch{j}')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Depth (z)')
            ax.set_title(name)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_multi_medium_errors(self, multi_medium_results, save_path=None,
                                 metric='rel_l2',
                                 title='Error Across Medium Models'):
        self.config.apply_style()

        medium_names = []
        mean_errors = []
        std_errors = []

        for medium_key, medium_res in multi_medium_results.items():
            if 'per_medium' in medium_res:
                for mk, mv in medium_res['per_medium'].items():
                    medium_names.append(mk)
                    mean_errors.append(mv.get('mean_rel_l2', 0.0))
                    std_errors.append(mv.get('std_rel_l2', 0.0))

        if not medium_names:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(medium_names))
        bars = ax.bar(x_pos, mean_errors, yerr=std_errors,
                      color=self.config.color_palette[0], alpha=self.config.alpha,
                      capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(medium_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Relative L2 Error')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class AttentionWeightPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(12, 6))
        else:
            self.config = config

    def plot_attention_weights(self, weights, z_values=None, save_path=None,
                               title='FNO Attention Weights'):
        self.config.apply_style()

        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        if weights.ndim == 3:
            weights = weights[0]

        if z_values is None:
            z_values = np.arange(weights.shape[-2])

        fig, ax = plt.subplots(figsize=(12, 6))
        if weights.ndim == 2:
            for h in range(weights.shape[0]):
                ax.plot(z_values, weights[h],
                        color=self.config.color_palette[h % len(self.config.color_palette)],
                        linewidth=self.config.linewidth, alpha=self.config.alpha,
                        label=f'Head {h}')
            ax.legend(fontsize=self.config.legend_size)
        else:
            ax.plot(z_values, weights,
                    color=self.config.color_palette[0],
                    linewidth=self.config.linewidth)

        ax.set_xlabel('Depth Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_spectral_features(self, spectral_acts, save_path=None,
                               title='FNO Spectral Features'):
        self.config.apply_style()

        n_layers = len(spectral_acts)
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
        if n_layers == 1:
            axes = [axes]

        for i, spec in enumerate(spectral_acts):
            ax = axes[i]
            if isinstance(spec, torch.Tensor):
                spec = spec.detach().cpu().numpy()
            if spec.ndim > 1:
                spec = spec.mean(axis=0)
            ax.plot(spec,
                    color=self.config.color_palette[i % len(self.config.color_palette)],
                    linewidth=self.config.linewidth)
            ax.set_xlabel('Frequency Mode')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'FNO Layer {i + 1}')
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class FiLMParameterPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(14, 6))
        else:
            self.config = config

    def plot_film_parameters(self, gammas, betas, save_path=None,
                            title='FiLM Modulation Parameters'):
        self.config.apply_style()

        n_layers = len(gammas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        gamma_norms = [g.norm().item() for g in gammas]
        beta_norms = [b.norm().item() for b in betas]
        gamma_means = [g.mean().item() for g in gammas]
        beta_means = [b.mean().item() for b in betas]

        layers = [f'Layer {i}' for i in range(n_layers)]
        x_pos = np.arange(n_layers)

        ax1.bar(x_pos - 0.2, gamma_norms, 0.4, label='Gamma Norm',
                color=self.config.color_palette[0], alpha=self.config.alpha)
        ax1.bar(x_pos + 0.2, beta_norms, 0.4, label='Beta Norm',
                color=self.config.color_palette[1], alpha=self.config.alpha)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_ylabel('L2 Norm')
        ax1.set_title('FiLM Parameter Norms')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(x_pos - 0.2, gamma_means, 0.4, label='Gamma Mean',
                color=self.config.color_palette[2], alpha=self.config.alpha)
        ax2.bar(x_pos + 0.2, beta_means, 0.4, label='Beta Mean',
                color=self.config.color_palette[3], alpha=self.config.alpha)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(layers, rotation=45, ha='right')
        ax2.set_ylabel('Mean Value')
        ax2.set_title('FiLM Parameter Means')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class GradientMonitorPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(14, 8))
        else:
            self.config = config

    def plot_gradient_norms(self, gradient_history, save_path=None,
                            title='Gradient Norms During Training'):
        self.config.apply_style()

        if isinstance(gradient_history, dict):
            fig, ax = plt.subplots(figsize=(14, 8))
            for name, norms in gradient_history.items():
                if len(norms) > 0:
                    ax.semilogy(norms, label=name, linewidth=0.8, alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(title)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.semilogy(gradient_history, linewidth=self.config.linewidth)
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_weight_norms(self, weight_history, save_path=None,
                          title='Weight Norms During Training'):
        self.config.apply_style()

        fig, ax = plt.subplots(figsize=(14, 8))
        for name, norms in weight_history.items():
            if len(norms) > 0:
                ax.semilogy(norms, label=name, linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Weight Norm')
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class ConvergencePlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(12, 6))
        else:
            self.config = config

    def plot_convergence_rate(self, loss_history, window=500, save_path=None,
                              title='Convergence Rate Analysis'):
        self.config.apply_style()

        values = [h.get('total', 0.0) for h in loss_history]
        if len(values) < window:
            window = len(values) // 2

        log_values = np.log10(np.array(values) + 1e-30)
        epochs = np.arange(len(log_values))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.semilogy(values, color=self.config.color_palette[0],
                     linewidth=0.5, alpha=0.5, label='Raw')
        ema = [values[0]]
        alpha = 0.01
        for v in values[1:]:
            ema.append(alpha * v + (1 - alpha) * ema[-1])
        ax1.semilogy(ema, color=self.config.color_palette[1],
                     linewidth=2.0, label='EMA')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Loss with EMA')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if len(values) >= window:
            recent = log_values[-window:]
            x = np.arange(len(recent))
            slope, intercept = np.polyfit(x, recent, 1)
            fit_line = slope * x + intercept
            ax2.plot(x, recent, color=self.config.color_palette[0],
                     linewidth=0.5, alpha=0.5, label='Log Loss')
            ax2.plot(x, fit_line, 'r--', linewidth=2.0,
                     label=f'Fit: slope={slope:.4f}')
            ax2.set_xlabel(f'Epoch (last {window})')
            ax2.set_ylabel('Log10(Loss)')
            ax2.set_title(f'Convergence Rate (last {window} epochs)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Not enough data', ha='center', va='center',
                     transform=ax2.transAxes)

        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_loss_phase_transition(self, loss_history, phase1_epochs=8000,
                                   save_path=None,
                                   title='Training Phase Transition'):
        self.config.apply_style()

        values = [h.get('total', 0.0) for h in loss_history]
        epochs = np.arange(len(values))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.semilogy(epochs, values, color=self.config.color_palette[0],
                    linewidth=0.8, alpha=0.7)

        if phase1_epochs < len(values):
            ax.axvline(x=phase1_epochs, color='red', linestyle='--',
                       linewidth=2, label='Phase Transition (AdamW → L-BFGS)')
            ax.fill_between(epochs[:phase1_epochs], min(values) * 0.1,
                            max(values) * 10, alpha=0.05, color='blue',
                            label='Phase 1 (AdamW)')
            ax.fill_between(epochs[phase1_epochs:], min(values) * 0.1,
                            max(values) * 10, alpha=0.05, color='green',
                            label='Phase 2 (L-BFGS)')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class PaperFigureGenerator:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(16, 12), dpi=300)
        else:
            self.config = config
        self.loss_plotter = LossConvergencePlotter(config)
        self.field_plotter = FieldComponentPlotter(config)
        self.spatial_plotter = SpatialErrorPlotter(config)
        self.freq_plotter = FrequencyAnalysisPlotter(config)
        self.medium_plotter = MediumProfilePlotter(config)
        self.attention_plotter = AttentionWeightPlotter(config)
        self.film_plotter = FiLMParameterPlotter(config)
        self.gradient_plotter = GradientMonitorPlotter(config)
        self.convergence_plotter = ConvergencePlotter(config)

    def generate_all_figures(self, output_dir, loss_history=None,
                             pred=None, target=None, coordinates=None,
                             freq_results=None, radial_profile=None,
                             depth_profile=None, medium_profiles=None,
                             attention_weights=None, spectral_features=None,
                             film_gammas=None, film_betas=None,
                             gradient_history=None,
                             component_metrics=None,
                             phase1_epochs=8000):
        os.makedirs(output_dir, exist_ok=True)

        generated = OrderedDict()

        if loss_history is not None:
            path = os.path.join(output_dir, 'fig_loss_convergence.png')
            self.loss_plotter.plot_loss_components(
                loss_history, save_path=path,
                title='FRPINNs Training Loss Convergence'
            )
            generated['loss_convergence'] = path

            path = os.path.join(output_dir, 'fig_loss_weights.png')
            self.loss_plotter.plot_loss_weights_evolution(
                loss_history, save_path=path
            )
            generated['loss_weights'] = path

            path = os.path.join(output_dir, 'fig_convergence_rate.png')
            self.convergence_plotter.plot_convergence_rate(
                loss_history, save_path=path
            )
            generated['convergence_rate'] = path

            path = os.path.join(output_dir, 'fig_phase_transition.png')
            self.convergence_plotter.plot_loss_phase_transition(
                loss_history, phase1_epochs=phase1_epochs, save_path=path
            )
            generated['phase_transition'] = path

        if pred is not None and target is not None:
            path = os.path.join(output_dir, 'fig_component_comparison.png')
            self.field_plotter.plot_component_comparison(
                pred, target, save_path=path
            )
            generated['component_comparison'] = path

            path = os.path.join(output_dir, 'fig_component_scatter.png')
            self.field_plotter.plot_component_scatter(
                pred, target, save_path=path
            )
            generated['component_scatter'] = path

        if component_metrics is not None:
            path = os.path.join(output_dir, 'fig_component_error_bar.png')
            self.field_plotter.plot_component_error_bar(
                component_metrics, save_path=path
            )
            generated['component_error_bar'] = path

        if radial_profile is not None:
            path = os.path.join(output_dir, 'fig_radial_error.png')
            self.spatial_plotter.plot_radial_error_profile(
                radial_profile, save_path=path
            )
            generated['radial_error'] = path

        if depth_profile is not None:
            path = os.path.join(output_dir, 'fig_depth_error.png')
            self.spatial_plotter.plot_depth_error_profile(
                depth_profile, save_path=path
            )
            generated['depth_error'] = path

        if freq_results is not None:
            path = os.path.join(output_dir, 'fig_frequency_error.png')
            self.freq_plotter.plot_frequency_error(
                freq_results, save_path=path
            )
            generated['frequency_error'] = path

            path = os.path.join(output_dir, 'fig_freq_component_heatmap.png')
            self.freq_plotter.plot_component_frequency_heatmap(
                freq_results, save_path=path
            )
            generated['freq_heatmap'] = path

        if medium_profiles is not None:
            path = os.path.join(output_dir, 'fig_medium_profiles.png')
            self.medium_plotter.plot_medium_comparison(
                medium_profiles, save_path=path
            )
            generated['medium_profiles'] = path

        if attention_weights is not None:
            path = os.path.join(output_dir, 'fig_attention_weights.png')
            self.attention_plotter.plot_attention_weights(
                attention_weights, save_path=path
            )
            generated['attention_weights'] = path

        if spectral_features is not None:
            path = os.path.join(output_dir, 'fig_spectral_features.png')
            self.attention_plotter.plot_spectral_features(
                spectral_features, save_path=path
            )
            generated['spectral_features'] = path

        if film_gammas is not None and film_betas is not None:
            path = os.path.join(output_dir, 'fig_film_parameters.png')
            self.film_plotter.plot_film_parameters(
                film_gammas, film_betas, save_path=path
            )
            generated['film_parameters'] = path

        if gradient_history is not None:
            path = os.path.join(output_dir, 'fig_gradient_norms.png')
            self.gradient_plotter.plot_gradient_norms(
                gradient_history, save_path=path
            )
            generated['gradient_norms'] = path

        return generated

    def generate_paper_figure_3_1(self, loss_history, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'paper_fig_3_1_loss_convergence.png')
        self.loss_plotter.plot_loss_components(
            loss_history, save_path=path,
            title='Figure 3-1: FRPINNs Loss Convergence',
            show_weights=True
        )
        return path

    def generate_paper_figure_3_2(self, pred, target, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'paper_fig_3_2_field_comparison.png')
        self.field_plotter.plot_component_comparison(
            pred, target, save_path=path,
            title='Figure 3-2: Field Component Comparison'
        )
        return path

    def generate_paper_figure_4_1(self, loss_history, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'paper_fig_4_1_fno_loss_convergence.png')
        self.loss_plotter.plot_loss_components(
            loss_history, save_path=path,
            title='Figure 4-1: FRPINNs-FNO Loss Convergence',
            show_weights=True
        )
        return path

    def generate_paper_figure_4_2(self, medium_profiles, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'paper_fig_4_2_medium_profiles.png')
        self.medium_plotter.plot_medium_comparison(
            medium_profiles, save_path=path,
            title='Figure 4-2: Medium Profile Comparison'
        )
        return path

    def generate_paper_figure_4_3(self, attention_weights, spectral_features,
                                  save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path1 = os.path.join(save_dir, 'paper_fig_4_3a_attention.png')
        self.attention_plotter.plot_attention_weights(
            attention_weights, save_path=path1,
            title='Figure 4-3a: FNO Attention Weights'
        )
        path2 = os.path.join(save_dir, 'paper_fig_4_3b_spectral.png')
        self.attention_plotter.plot_spectral_features(
            spectral_features, save_path=path2,
            title='Figure 4-3b: FNO Spectral Features'
        )
        return path1, path2


def create_visualizer(config=None):
    if config is None:
        config = PlotConfig()
    return PaperFigureGenerator(config)


class SubnetAnalysisPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(14, 8))
        else:
            self.config = config

    def plot_subnet_diversity(self, diversity_history, save_path=None,
                              title='Subnet Output Diversity'):
        self.config.apply_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(diversity_history, color=self.config.color_palette[0],
                linewidth=self.config.linewidth)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Variance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_subnet_correlation_matrix(self, correlations, save_path=None,
                                       title='Subnet Output Correlations'):
        self.config.apply_style()
        if isinstance(correlations, torch.Tensor):
            correlations = correlations.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Correlation')
        n = correlations.shape[0]
        ax.set_xticks(range(n))
        ax.set_xticklabels([f'S{i}' for i in range(n)], fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'S{i}' for i in range(n)], fontsize=8)
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{correlations[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if abs(correlations[i, j]) > 0.5 else 'black')
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig

    def plot_subnet_individual_outputs(self, outputs, target=None, save_path=None,
                                      component_idx=0, title='Individual Subnet Outputs'):
        self.config.apply_style()
        n_subnets = len(outputs)
        fig, axes = plt.subplots(2, (n_subnets + 1) // 2, figsize=(4 * n_subnets // 2 + 2, 8))
        axes = axes.flatten()
        for i, out in enumerate(outputs):
            ax = axes[i]
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            ax.plot(out[:, component_idx], color=self.config.color_palette[i % len(self.config.color_palette)],
                    linewidth=0.8, alpha=0.7, label=f'Subnet {i}')
            if target is not None:
                if isinstance(target, torch.Tensor):
                    target = target.detach().cpu().numpy()
                ax.plot(target[:, component_idx], 'k--', linewidth=0.5, alpha=0.3, label='Target')
            ax.set_title(f'Subnet {i} (scale={i})', fontsize=9)
            ax.grid(True, alpha=0.2)
        for j in range(n_subnets, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title, fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig


class TrainingProgressPlotter:
    def __init__(self, config=None):
        if config is None:
            self.config = PlotConfig(figsize=(16, 10))
        else:
            self.config = config

    def plot_training_dashboard(self, loss_history, lr_history=None,
                                gradient_norms=None, weight_norms=None,
                                eval_metrics=None, save_path=None,
                                title='Training Dashboard'):
        self.config.apply_style()
        n_plots = 2
        if lr_history is not None:
            n_plots += 1
        if gradient_norms is not None:
            n_plots += 1
        if eval_metrics is not None:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0
        loss_names = ['pde', 'gauge', 'bc', 'data', 'sym', 'total']
        loss_labels = ['PDE', 'Gauge', 'BC', 'Data', 'Sym', 'Total']
        epochs = list(range(len(loss_history)))
        for name, label, color in zip(loss_names, loss_labels, self.config.color_palette[:6]):
            values = [h.get(name, 0.0) for h in loss_history]
            axes[plot_idx].semilogy(epochs, values, color=color, label=label,
                                    linewidth=0.8, alpha=0.7)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Loss Components')
        axes[plot_idx].legend(fontsize=8, ncol=3)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        w_names = ['w_pde', 'w_gauge', 'w_bc', 'w_data', 'w_sym']
        w_labels = ['PDE W', 'Gauge W', 'BC W', 'Data W', 'Sym W']
        for name, label, color in zip(w_names, w_labels, self.config.color_palette[:5]):
            values = [h.get(name, None) for h in loss_history]
            valid = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid:
                e_vals, v_vals = zip(*valid)
                axes[plot_idx].plot(e_vals, v_vals, color=color, label=label,
                                    linewidth=0.8, alpha=0.7)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Weight')
        axes[plot_idx].set_title('Adaptive Weights')
        axes[plot_idx].legend(fontsize=8, ncol=3)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        if lr_history is not None:
            axes[plot_idx].plot(lr_history, color=self.config.color_palette[0],
                                linewidth=self.config.linewidth)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Learning Rate')
            axes[plot_idx].set_title('Learning Rate Schedule')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        if gradient_norms is not None:
            axes[plot_idx].semilogy(gradient_norms, color=self.config.color_palette[1],
                                    linewidth=0.8, alpha=0.7)
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Gradient Norm')
            axes[plot_idx].set_title('Gradient Norm')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        if eval_metrics is not None:
            for name, values in eval_metrics.items():
                axes[plot_idx].plot(values, label=name, linewidth=0.8, alpha=0.7)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Metric')
            axes[plot_idx].set_title('Evaluation Metrics')
            axes[plot_idx].legend(fontsize=8)
            axes[plot_idx].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return fig
