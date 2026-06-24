import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import json
import csv
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.frpinn import FRPINNs
from models.frpinn_fno import FRPINNFNO
from data.medium_profile import MediumProfile, MediumModelLibrary


COMPONENT_NAMES = [
    'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
    'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
    'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
    'Kphi_Re', 'Kphi_Im'
]


def build_frpinn_model(config):
    model_cfg = config.get('model', {})
    model = FRPINNs(
        num_subnets=model_cfg.get('num_subnets', 10),
        input_dim=model_cfg.get('input_dim', 8),
        fourier_dim=model_cfg.get('fourier_dim', 64),
        hidden_dims=model_cfg.get('hidden_dims', [128, 200, 200, 200, 200, 128]),
        output_dim=model_cfg.get('output_dim', 14),
        scale_factors=model_cfg.get('scale_factors',
                                   [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
    )
    return model


def build_frpinn_fno_model(config):
    model_cfg = config.get('model', {})
    model = FRPINNFNO(
        num_subnets=model_cfg.get('num_subnets', 10),
        input_dim=model_cfg.get('input_dim', 8),
        fourier_dim=model_cfg.get('fourier_dim', 64),
        hidden_dims=model_cfg.get('hidden_dims', [128, 200, 200, 200, 200, 128]),
        output_dim=model_cfg.get('output_dim', 14),
        scale_factors=model_cfg.get('scale_factors',
                                   [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
        nz=model_cfg.get('nz', 64),
        medium_channels=model_cfg.get('medium_channels', 4),
        fno_hidden_dim=model_cfg.get('fno_hidden_dim', 64),
        num_fno_layers=model_cfg.get('num_fno_layers', 4),
        num_modes=model_cfg.get('num_modes', 12),
        film_dim=model_cfg.get('film_dim', 128),
        pooling_type=model_cfg.get('pooling_type', 'attention'),
        num_attention_heads=model_cfg.get('num_attention_heads', 4),
        activation_name_fno=model_cfg.get('activation_name_fno', 'gelu'),
        use_fno_residual=model_cfg.get('use_fno_residual', True),
        aggregation=model_cfg.get('aggregation', 'weighted_mean'),
        encoder_type=model_cfg.get('encoder_type', 'standard'),
    )
    return model


def load_checkpoint(checkpoint_path, device='cpu'):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    return checkpoint, config, device


def build_model_from_checkpoint(checkpoint, config, device):
    has_medium = 'n_medium_models' in checkpoint
    if has_medium:
        model = build_frpinn_fno_model(config)
    else:
        model = build_frpinn_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, has_medium


def create_medium_profile_from_layers(layers, z_min=0.0, z_max=800e3, nz=64):
    profile = MediumProfile(
        layers=layers,
        z_min=z_min,
        z_max=z_max,
        nz=nz,
    )
    return profile


def create_medium_profile_from_dict(medium_dict):
    layers = []
    for layer in medium_dict.get('layers', []):
        layers.append({
            'thickness': layer.get('thickness', 100e3),
            'eps_r': layer.get('eps_r', 1.0),
            'mu_r': layer.get('mu_r', 1.0),
            'sigma': layer.get('sigma', 0.01),
        })
    return create_medium_profile_from_layers(
        layers,
        z_min=medium_dict.get('z_min', 0.0),
        z_max=medium_dict.get('z_max', 800e3),
        nz=medium_dict.get('nz', 64),
    )


def create_medium_profile_from_library(library_path, profile_idx=0):
    library = MediumModelLibrary.load_library(library_path)
    if profile_idx >= len(library.models):
        raise ValueError(f"Profile index {profile_idx} out of range (max {len(library.models)-1})")
    return library.models[profile_idx]


def build_input_tensor(x, y, z, xp, yp, zp, rho, freq):
    f_log = np.log10(freq) if freq > 0 else 0.0
    coords = np.array([x, y, z, xp, yp, zp, rho, f_log], dtype=np.float32)
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)


def build_grid_input_tensor(rho_range, z_range, source_pos, freq,
                            n_rho=100, n_z=100):
    rho_vals = np.linspace(rho_range[0], rho_range[1], n_rho)
    z_vals = np.linspace(z_range[0], z_range[1], n_z)
    rho_grid, z_grid = np.meshgrid(rho_vals, z_vals)

    f_log = np.log10(freq) if freq > 0 else 0.0
    n_pts = n_rho * n_z
    x_flat = np.zeros((n_pts, 8), dtype=np.float32)
    x_flat[:, 0] = rho_grid.flatten()
    x_flat[:, 2] = z_grid.flatten()
    x_flat[:, 3] = source_pos[0]
    x_flat[:, 4] = source_pos[1]
    x_flat[:, 5] = source_pos[2]
    x_flat[:, 6] = rho_grid.flatten()
    x_flat[:, 7] = f_log

    return torch.tensor(x_flat, dtype=torch.float32), rho_grid, z_grid


def build_medium_profile_tensor(medium_profile, freq, device='cpu'):
    profile_np = medium_profile.discretize(freq=freq)
    profile_tensor = torch.tensor(profile_np, dtype=torch.float32, device=device)
    if profile_tensor.dim() == 2:
        profile_tensor = profile_tensor.unsqueeze(0)
    return profile_tensor


class FRPINNPredictor:
    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.checkpoint, self.config, _ = load_checkpoint(
            checkpoint_path, str(self.device)
        )
        self.model, self.is_fno = build_model_from_checkpoint(
            self.checkpoint, self.config, self.device
        )
        self.frequencies = self.config.get('frequencies', [0.1, 100.0])
        self.source_pos = self.config.get('source_pos', [0.0, 0.0, 0.0])
        self.rho_range = self.config.get('rho_range', [0.1, 800e3])
        self.z_range = self.config.get('z_range', [0.0, 800e3])

        self._medium_profiles_cache = {}

    def _get_medium_profile_tensor(self, medium_profile, freq):
        key = id(medium_profile), freq
        if key not in self._medium_profiles_cache:
            self._medium_profiles_cache[key] = build_medium_profile_tensor(
                medium_profile, freq, self.device
            )
        return self._medium_profiles_cache[key]

    def predict_point(self, x, y, z, xp, yp, zp, rho, freq,
                      medium_profile=None):
        x_tensor = build_input_tensor(x, y, z, xp, yp, zp, rho, freq)
        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            if self.is_fno:
                if medium_profile is None:
                    raise ValueError("FRPINN-FNO model requires a medium_profile")
                mp_tensor = self._get_medium_profile_tensor(medium_profile, freq)
                pred = self.model(x_tensor, mp_tensor)
            else:
                pred = self.model(x_tensor)

        pred_np = pred.cpu().numpy().flatten()
        result = OrderedDict()
        for name, val in zip(COMPONENT_NAMES, pred_np):
            result[name] = float(val)
        return result

    def predict_points(self, coords_array, freq, medium_profile=None):
        if isinstance(coords_array, np.ndarray):
            x_tensor = torch.tensor(coords_array, dtype=torch.float32)
        else:
            x_tensor = coords_array
        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            if self.is_fno:
                if medium_profile is None:
                    raise ValueError("FRPINN-FNO model requires a medium_profile")
                mp_tensor = self._get_medium_profile_tensor(medium_profile, freq)
                pred = self.model(x_tensor, mp_tensor)
            else:
                pred = self.model(x_tensor)

        pred_np = pred.cpu().numpy()
        return pred_np

    def predict_grid(self, freq, medium_profile=None,
                     rho_range=None, z_range=None,
                     source_pos=None, n_rho=100, n_z=100):
        if rho_range is None:
            rho_range = self.rho_range
        if z_range is None:
            z_range = self.z_range
        if source_pos is None:
            source_pos = self.source_pos

        x_tensor, rho_grid, z_grid = build_grid_input_tensor(
            rho_range, z_range, source_pos, freq, n_rho, n_z
        )
        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            if self.is_fno:
                if medium_profile is None:
                    raise ValueError("FRPINN-FNO model requires a medium_profile")
                mp_tensor = self._get_medium_profile_tensor(medium_profile, freq)
                pred = self.model(x_tensor, mp_tensor)
            else:
                pred = self.model(x_tensor)

        pred_grid = pred.cpu().numpy().reshape(n_z, n_rho, 14)
        result = OrderedDict()
        result['rho_grid'] = rho_grid
        result['z_grid'] = z_grid
        for i, name in enumerate(COMPONENT_NAMES):
            result[name] = pred_grid[:, :, i]
        return result

    def predict_multi_frequency(self, frequencies=None, medium_profile=None,
                                rho_range=None, z_range=None,
                                source_pos=None, n_rho=50, n_z=50):
        if frequencies is None:
            frequencies = self.frequencies

        all_results = OrderedDict()
        for freq in frequencies:
            all_results[freq] = self.predict_grid(
                freq=freq,
                medium_profile=medium_profile,
                rho_range=rho_range,
                z_range=z_range,
                source_pos=source_pos,
                n_rho=n_rho,
                n_z=n_z,
            )
        return all_results

    def predict_from_csv(self, csv_path, freq, medium_profile=None):
        coords_list = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                vals = [float(v) for v in row[:8]]
                coords_list.append(vals)
        coords_array = np.array(coords_list, dtype=np.float32)
        pred_np = self.predict_points(coords_array, freq, medium_profile)
        return coords_array, pred_np

    def get_component_real(self, result, component_base):
        key = f'{component_base}_Re'
        if key in result:
            return result[key]
        raise KeyError(f"Component {key} not found in result")

    def get_component_imag(self, result, component_base):
        key = f'{component_base}_Im'
        if key in result:
            return result[key]
        raise KeyError(f"Component {key} not found in result")

    def get_component_amplitude(self, result, component_base):
        re = self.get_component_real(result, component_base)
        im = self.get_component_imag(result, component_base)
        if isinstance(re, np.ndarray):
            return np.sqrt(re ** 2 + im ** 2)
        return math.sqrt(re ** 2 + im ** 2)

    def get_component_phase(self, result, component_base):
        re = self.get_component_real(result, component_base)
        im = self.get_component_imag(result, component_base)
        if isinstance(re, np.ndarray):
            return np.arctan2(im, re)
        return math.atan2(im, re)

    def get_all_components_summary(self, result):
        summary = OrderedDict()
        component_bases = ['Kxx', 'Kxz', 'Kyz', 'Kzx', 'Kzy', 'Kzz', 'Kphi']
        for base in component_bases:
            re = result.get(f'{base}_Re', None)
            im = result.get(f'{base}_Im', None)
            if re is not None and im is not None:
                if isinstance(re, np.ndarray):
                    amp = np.sqrt(re ** 2 + im ** 2)
                    phase = np.arctan2(im, re)
                    summary[base] = {
                        'Re_min': float(re.min()),
                        'Re_max': float(re.max()),
                        'Re_mean': float(re.mean()),
                        'Im_min': float(im.min()),
                        'Im_max': float(im.max()),
                        'Im_mean': float(im.mean()),
                        'Amp_max': float(amp.max()),
                        'Amp_mean': float(amp.mean()),
                    }
                else:
                    amp = math.sqrt(re ** 2 + im ** 2)
                    phase = math.atan2(im, re)
                    summary[base] = {
                        'Re': float(re),
                        'Im': float(im),
                        'Amplitude': float(amp),
                        'Phase_rad': float(phase),
                        'Phase_deg': float(math.degrees(phase)),
                    }
        return summary

    def evaluate_against_ground_truth(self, pred_np, gt_np, component_names=None):
        if component_names is None:
            component_names = COMPONENT_NAMES
        from evaluation import RelativeErrorMetrics
        metrics = RelativeErrorMetrics()
        results = OrderedDict()
        all_pred = pred_np.flatten()
        all_gt = gt_np.flatten()
        results['overall'] = metrics.all_metrics(all_pred, all_gt)
        per_component = OrderedDict()
        n_components = pred_np.shape[1] if pred_np.ndim > 1 else 1
        for i in range(min(n_components, len(component_names))):
            name = component_names[i]
            if pred_np.ndim > 1:
                p = pred_np[:, i]
                g = gt_np[:, i]
            else:
                p = pred_np
                g = gt_np
            per_component[name] = metrics.all_metrics(p, g)
        results['per_component'] = per_component
        component_bases = ['Kxx', 'Kxz', 'Kyz', 'Kzx', 'Kzy', 'Kzz', 'Kphi']
        per_base = OrderedDict()
        for base in component_bases:
            re_idx = component_names.index(f'{base}_Re') if f'{base}_Re' in component_names else None
            im_idx = component_names.index(f'{base}_Im') if f'{base}_Im' in component_names else None
            if re_idx is not None and im_idx is not None and pred_np.ndim > 1:
                p_amp = np.sqrt(pred_np[:, re_idx] ** 2 + pred_np[:, im_idx] ** 2)
                g_amp = np.sqrt(gt_np[:, re_idx] ** 2 + gt_np[:, im_idx] ** 2)
                per_base[base] = metrics.all_metrics(
                    torch.tensor(p_amp, dtype=torch.float32),
                    torch.tensor(g_amp, dtype=torch.float32)
                )
        results['per_base_amplitude'] = per_base
        return results

    def predict_and_compare(self, coords_array, gt_array, freq,
                            medium_profile=None):
        pred_np = self.predict_points(coords_array, freq, medium_profile)
        eval_results = self.evaluate_against_ground_truth(pred_np, gt_array)
        return pred_np, eval_results


def load_ground_truth_csv(csv_path, has_coords=True):
    coords_list = []
    gt_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            vals = [float(v) for v in row]
            if has_coords and len(vals) > 14:
                coords_list.append(vals[:8])
                gt_list.append(vals[8:22])
            elif has_coords and len(vals) == 22:
                coords_list.append(vals[:8])
                gt_list.append(vals[8:22])
            elif not has_coords and len(vals) >= 14:
                gt_list.append(vals[:14])
            else:
                gt_list.append(vals)
    coords = np.array(coords_list, dtype=np.float32) if coords_list else None
    gt = np.array(gt_list, dtype=np.float32)
    return coords, gt


def print_evaluation_results(eval_results):
    print("\n" + "=" * 70)
    print("EVALUATION: Predicted vs Ground Truth")
    print("=" * 70)
    overall = eval_results['overall']
    print(f"\n{'Overall Metrics':<20}")
    print("-" * 50)
    for metric_name, value in overall.items():
        print(f"  {metric_name:<15}: {value:.6e}")
    per_comp = eval_results.get('per_component', {})
    if per_comp:
        print(f"\n{'Per-Component Relative L2 Error':<40}")
        print("-" * 50)
        for name, metrics in per_comp.items():
            print(f"  {name:<12}: rel_L2={metrics['rel_l2']:.6e}, "
                  f"rel_Linf={metrics['rel_linf']:.6e}, R²={metrics['r2']:.6f}")
    per_base = eval_results.get('per_base_amplitude', {})
    if per_base:
        print(f"\n{'Per-Base Amplitude Error':<40}")
        print("-" * 50)
        for name, metrics in per_base.items():
            print(f"  {name:<8}: rel_L2={metrics['rel_l2']:.6e}, "
                  f"RMSE={metrics['rmse']:.6e}, R²={metrics['r2']:.6f}")


def save_evaluation_results(eval_results, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    serializable = OrderedDict()
    serializable['overall'] = eval_results['overall']
    comp_dict = OrderedDict()
    for name, metrics in eval_results.get('per_component', {}).items():
        comp_dict[name] = metrics
    serializable['per_component'] = comp_dict
    base_dict = OrderedDict()
    for name, metrics in eval_results.get('per_base_amplitude', {}).items():
        base_dict[name] = metrics
    serializable['per_base_amplitude'] = base_dict
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {output_path}")


def save_predictions_csv(predictions, output_path, coords=None):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if coords is not None:
            coord_header = ['x', 'y', 'z', 'xp', 'yp', 'zp', 'rho', 'f_log']
        else:
            coord_header = []
        writer.writerow(coord_header + COMPONENT_NAMES)
        for i in range(predictions.shape[0]):
            row = []
            if coords is not None:
                row.extend(coords[i].tolist())
            row.extend(predictions[i].tolist())
            writer.writerow(row)
    print(f"Predictions saved to {output_path}")


def save_grid_predictions_npz(grid_result, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    save_dict = {
        'rho_grid': grid_result['rho_grid'],
        'z_grid': grid_result['z_grid'],
    }
    for name in COMPONENT_NAMES:
        if name in grid_result:
            save_dict[name] = grid_result[name]
    np.savez(output_path, **save_dict)
    print(f"Grid predictions saved to {output_path}")


def save_summary_json(summary, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {output_path}")


def parse_medium_args(args):
    if args.medium_json:
        with open(args.medium_json, 'r') as f:
            medium_dict = json.load(f)
        return create_medium_profile_from_dict(medium_dict)

    elif args.medium_library:
        return create_medium_profile_from_library(
            args.medium_library, args.medium_idx
        )

    elif args.medium_layers:
        layers = []
        layer_strs = args.medium_layers.split(';')
        for layer_str in layer_strs:
            parts = layer_str.split(',')
            layer = {
                'thickness': float(parts[0]),
                'eps_r': float(parts[1]) if len(parts) > 1 else 1.0,
                'mu_r': float(parts[2]) if len(parts) > 2 else 1.0,
                'sigma': float(parts[3]) if len(parts) > 3 else 0.01,
            }
            layers.append(layer)
        z_min = args.z_min if args.z_min is not None else 0.0
        z_max = args.z_max if args.z_max is not None else 800e3
        nz = args.nz if args.nz is not None else 64
        return create_medium_profile_from_layers(layers, z_min, z_max, nz)

    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='FRPINN / FRPINN-FNO Prediction Tool'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (auto-detect if not specified)')

    medium_group = parser.add_argument_group('Medium Profile')
    medium_group.add_argument('--medium_json', type=str, default=None,
                              help='Path to JSON file defining medium layers')
    medium_group.add_argument('--medium_library', type=str, default=None,
                              help='Path to saved medium library (.json)')
    medium_group.add_argument('--medium_idx', type=int, default=0,
                              help='Index of medium profile in library')
    medium_group.add_argument('--medium_layers', type=str, default=None,
                              help='Layer params: thickness,eps_r,mu_r,sigma;...')
    medium_group.add_argument('--z_min', type=float, default=None)
    medium_group.add_argument('--z_max', type=float, default=None)
    medium_group.add_argument('--nz', type=int, default=None)

    obs_group = parser.add_argument_group('Observation System')
    obs_group.add_argument('--source_x', type=float, default=0.0,
                           help='Source x position')
    obs_group.add_argument('--source_y', type=float, default=0.0,
                           help='Source y position')
    obs_group.add_argument('--source_z', type=float, default=0.0,
                           help='Source z position')
    obs_group.add_argument('--freq', type=float, default=None,
                           help='Single frequency (Hz)')
    obs_group.add_argument('--freqs', type=str, default=None,
                           help='Multiple frequencies: 0.1,100,2000')
    obs_group.add_argument('--rho_min', type=float, default=None)
    obs_group.add_argument('--rho_max', type=float, default=None)
    obs_group.add_argument('--z_obs_min', type=float, default=None)
    obs_group.add_argument('--z_obs_max', type=float, default=None)

    pred_group = parser.add_argument_group('Prediction Mode')
    pred_group.add_argument('--mode', type=str, default='grid',
                            choices=['point', 'grid', 'csv', 'multi_freq', 'evaluate'],
                            help='Prediction mode')
    pred_group.add_argument('--point_x', type=float, default=1000.0)
    pred_group.add_argument('--point_y', type=float, default=0.0)
    pred_group.add_argument('--point_z', type=float, default=50000.0)
    pred_group.add_argument('--n_rho', type=int, default=100)
    pred_group.add_argument('--n_z', type=int, default=100)
    pred_group.add_argument('--csv_input', type=str, default=None,
                            help='CSV file with input coordinates')
    pred_group.add_argument('--ground_truth', type=str, default=None,
                            help='CSV file with ground truth values (8 coords + 14 components)')
    pred_group.add_argument('--gt_has_coords', type=int, default=1,
                            help='Whether ground truth CSV has 8 coordinate columns before 14 components')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='predictions')
    output_group.add_argument('--output_prefix', type=str, default='pred')
    output_group.add_argument('--save_csv', action='store_true')
    output_group.add_argument('--save_npz', action='store_true')
    output_group.add_argument('--save_summary', action='store_true')
    output_group.add_argument('--save_component_plots', action='store_true')
    pred_group.add_argument('--save_error_csv', action='store_true',
                            help='Save per-point error details to CSV')

    args = parser.parse_args()

    predictor = FRPINNPredictor(args.checkpoint, args.device)

    medium_profile = parse_medium_args(args)
    if predictor.is_fno and medium_profile is None:
        print("ERROR: FRPINN-FNO model requires a medium profile.")
        print("Use --medium_json, --medium_library, or --medium_layers to specify.")
        return

    source_pos = [args.source_x, args.source_y, args.source_z]

    if args.freq is not None:
        frequencies = [args.freq]
    elif args.freqs is not None:
        frequencies = [float(f) for f in args.freqs.split(',')]
    else:
        frequencies = predictor.frequencies

    rho_range = (args.rho_min, args.rho_max) if args.rho_min is not None else predictor.rho_range
    z_range = (args.z_obs_min, args.z_obs_max) if args.z_obs_min is not None else predictor.z_range

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model type: {'FRPINN-FNO' if predictor.is_fno else 'FRPINN'}")
    print(f"Device: {predictor.device}")
    print(f"Frequencies: {frequencies}")
    print(f"Source position: {source_pos}")
    print(f"Rho range: {rho_range}")
    print(f"Z range: {z_range}")
    if medium_profile is not None:
        print(f"Medium profile: {medium_profile.name}")
        print(f"  Layers: {len(medium_profile.layers)}")
        print(f"  z range: [{medium_profile.z_min}, {medium_profile.z_max}]")
    print()

    for freq in frequencies:
        freq_str = f"{freq:.2e}".replace('.', 'p').replace('+', '').replace('-', 'n')
        prefix = f"{args.output_prefix}_f{freq_str}"

        if args.mode == 'point':
            rho = args.point_x
            result = predictor.predict_point(
                x=args.point_x, y=args.point_y, z=args.point_z,
                xp=source_pos[0], yp=source_pos[1], zp=source_pos[2],
                rho=rho, freq=freq,
                medium_profile=medium_profile,
            )
            print(f"\n=== Point Prediction at freq={freq} Hz ===")
            print(f"Point: x={args.point_x}, y={args.point_y}, z={args.point_z}")
            print(f"Source: ({source_pos[0]}, {source_pos[1]}, {source_pos[2]})")
            print()
            component_bases = ['Kxx', 'Kxz', 'Kyz', 'Kzx', 'Kzy', 'Kzz', 'Kphi']
            print(f"{'Component':<10} {'Real':>14} {'Imag':>14} {'Amplitude':>14} {'Phase(deg)':>12}")
            print("-" * 66)
            for base in component_bases:
                re_val = result.get(f'{base}_Re', 0.0)
                im_val = result.get(f'{base}_Im', 0.0)
                amp = math.sqrt(re_val ** 2 + im_val ** 2)
                phase = math.degrees(math.atan2(im_val, re_val))
                print(f"{base:<10} {re_val:>14.6e} {im_val:>14.6e} {amp:>14.6e} {phase:>12.4f}")

            if args.save_summary:
                summary = predictor.get_all_components_summary(result)
                summary_path = os.path.join(args.output_dir, f"{prefix}_point_summary.json")
                save_summary_json(summary, summary_path)

        elif args.mode == 'grid':
            grid_result = predictor.predict_grid(
                freq=freq,
                medium_profile=medium_profile,
                rho_range=rho_range,
                z_range=z_range,
                source_pos=source_pos,
                n_rho=args.n_rho,
                n_z=args.n_z,
            )

            summary = predictor.get_all_components_summary(grid_result)
            print(f"\n=== Grid Prediction at freq={freq} Hz ===")
            print(f"Grid: {args.n_rho} x {args.n_z} points")
            print(f"Rho: [{rho_range[0]}, {rho_range[1]}]")
            print(f"Z: [{z_range[0]}, {z_range[1]}]")
            print()
            component_bases = ['Kxx', 'Kxz', 'Kyz', 'Kzx', 'Kzy', 'Kzz', 'Kphi']
            print(f"{'Comp':<6} {'Re_min':>12} {'Re_max':>12} {'Im_min':>12} {'Im_max':>12} {'Amp_max':>12}")
            print("-" * 68)
            for base in component_bases:
                s = summary.get(base, {})
                print(f"{base:<6} {s.get('Re_min',0):>12.4e} {s.get('Re_max',0):>12.4e} "
                      f"{s.get('Im_min',0):>12.4e} {s.get('Im_max',0):>12.4e} "
                      f"{s.get('Amp_max',0):>12.4e}")

            if args.save_npz:
                npz_path = os.path.join(args.output_dir, f"{prefix}_grid.npz")
                save_grid_predictions_npz(grid_result, npz_path)

            if args.save_summary:
                summary_path = os.path.join(args.output_dir, f"{prefix}_grid_summary.json")
                save_summary_json(summary, summary_path)

            if args.save_component_plots:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    component_bases = ['Kxx', 'Kxz', 'Kyz', 'Kzx', 'Kzy', 'Kzz', 'Kphi']
                    rho_grid = grid_result['rho_grid']
                    z_grid = grid_result['z_grid']

                    fig, axes = plt.subplots(7, 2, figsize=(14, 28))
                    for i, base in enumerate(component_bases):
                        re_data = grid_result.get(f'{base}_Re')
                        im_data = grid_result.get(f'{base}_Im')

                        if re_data is not None:
                            ax = axes[i, 0]
                            vmax = np.abs(re_data).max()
                            if vmax > 0:
                                pcm = ax.pcolormesh(rho_grid, z_grid, re_data,
                                                    cmap='RdBu_r',
                                                    vmin=-vmax, vmax=vmax)
                                plt.colorbar(pcm, ax=ax)
                            ax.set_title(f'{base} Real')
                            ax.set_xlabel('rho')
                            ax.set_ylabel('z')

                        if im_data is not None:
                            ax = axes[i, 1]
                            vmax = np.abs(im_data).max()
                            if vmax > 0:
                                pcm = ax.pcolormesh(rho_grid, z_grid, im_data,
                                                    cmap='RdBu_r',
                                                    vmin=-vmax, vmax=vmax)
                                plt.colorbar(pcm, ax=ax)
                            ax.set_title(f'{base} Imag')
                            ax.set_xlabel('rho')
                            ax.set_ylabel('z')

                    fig.suptitle(f'Predicted Components at freq={freq} Hz', fontsize=14)
                    plt.tight_layout()
                    plot_path = os.path.join(args.output_dir, f"{prefix}_components.png")
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Component plots saved to {plot_path}")
                except ImportError:
                    print("matplotlib not available, skipping plots")

        elif args.mode == 'csv':
            if args.csv_input is None:
                print("ERROR: --csv_input required for csv mode")
                continue

            coords, predictions = predictor.predict_from_csv(
                args.csv_input, freq, medium_profile
            )

            if args.save_csv:
                csv_path = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
                save_predictions_csv(predictions, csv_path, coords=coords)

            summary = OrderedDict()
            for i, name in enumerate(COMPONENT_NAMES):
                vals = predictions[:, i]
                summary[name] = {
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                }
            print(f"\n=== CSV Prediction at freq={freq} Hz ===")
            print(f"Points: {predictions.shape[0]}")
            for name, stats in summary.items():
                print(f"  {name}: min={stats['min']:.4e}, max={stats['max']:.4e}, "
                      f"mean={stats['mean']:.4e}")

            if args.save_summary:
                summary_path = os.path.join(args.output_dir, f"{prefix}_csv_summary.json")
                save_summary_json(summary, summary_path)

        elif args.mode == 'multi_freq':
            multi_results = predictor.predict_multi_frequency(
                frequencies=frequencies,
                medium_profile=medium_profile,
                rho_range=rho_range,
                z_range=z_range,
                source_pos=source_pos,
                n_rho=args.n_rho,
                n_z=args.n_z,
            )

            print(f"\n=== Multi-Frequency Prediction ===")
            for f_val, f_result in multi_results.items():
                f_summary = predictor.get_all_components_summary(f_result)
                print(f"\nFreq = {f_val} Hz:")
                component_bases = ['Kxx', 'Kxz', 'Kyz', 'Kzx', 'Kzy', 'Kzz', 'Kphi']
                for base in component_bases:
                    s = f_summary.get(base, {})
                    print(f"  {base}: Re_mean={s.get('Re_mean',0):.4e}, "
                          f"Im_mean={s.get('Im_mean',0):.4e}, "
                          f"Amp_max={s.get('Amp_max',0):.4e}")

                if args.save_npz:
                    freq_str = f"{f_val:.2e}".replace('.', 'p').replace('+', '').replace('-', 'n')
                    npz_path = os.path.join(args.output_dir,
                                            f"{args.output_prefix}_f{freq_str}_grid.npz")
                    save_grid_predictions_npz(f_result, npz_path)

            if args.save_summary:
                all_summaries = OrderedDict()
                for f_val, f_result in multi_results.items():
                    all_summaries[str(f_val)] = predictor.get_all_components_summary(f_result)
                summary_path = os.path.join(args.output_dir,
                                            f"{args.output_prefix}_multi_freq_summary.json")
                save_summary_json(all_summaries, summary_path)

        elif args.mode == 'evaluate':
            gt_path = args.ground_truth
            if gt_path is None:
                print("ERROR: --ground_truth required for evaluate mode")
                print("  CSV format: 8 coordinate columns + 14 component columns")
                print("  Or use --gt_has_coords 0 if only 14 component columns")
                continue

            has_coords = bool(args.gt_has_coords)
            coords, gt_array = load_ground_truth_csv(gt_path, has_coords=has_coords)
            print(f"\n=== Evaluate Mode at freq={freq} Hz ===")
            print(f"Ground truth file: {gt_path}")
            print(f"Number of points: {gt_array.shape[0]}")
            print(f"Components: {gt_array.shape[1]}")

            if coords is not None:
                pred_np, eval_results = predictor.predict_and_compare(
                    coords, gt_array, freq, medium_profile
                )
            else:
                if args.csv_input is not None:
                    coords_eval, _ = predictor.predict_from_csv(
                        args.csv_input, freq, medium_profile
                    )
                    pred_np, eval_results = predictor.predict_and_compare(
                        coords_eval, gt_array, freq, medium_profile
                    )
                else:
                    print("ERROR: No coordinate data available. Provide --csv_input or ground truth with coords.")
                    continue

            print_evaluation_results(eval_results)

            if args.save_error_csv:
                error_csv_path = os.path.join(args.output_dir, f"{prefix}_errors.csv")
                with open(error_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = COMPONENT_NAMES[:]
                    header = [f'pred_{n}' for n in header] + [f'gt_{n}' for n in COMPONENT_NAMES] + [f'rel_err_{n}' for n in COMPONENT_NAMES]
                    writer.writerow(header)
                    for i in range(pred_np.shape[0]):
                        row = []
                        for j in range(14):
                            row.append(pred_np[i, j])
                        for j in range(14):
                            row.append(gt_array[i, j])
                        for j in range(14):
                            denom = abs(gt_array[i, j]) + 1e-30
                            row.append(abs(pred_np[i, j] - gt_array[i, j]) / denom)
                        writer.writerow(row)
                print(f"Per-point errors saved to {error_csv_path}")

            if args.save_csv:
                csv_path = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
                save_predictions_csv(pred_np, csv_path, coords=coords)

            if args.save_summary:
                eval_path = os.path.join(args.output_dir, f"{prefix}_evaluation.json")
                save_evaluation_results(eval_results, eval_path)

    print("\nPrediction completed.")


if __name__ == '__main__':
    import math
    main()
