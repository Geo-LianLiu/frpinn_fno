import yaml
import os


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def config_to_frpinn_args(config):
    layer_info = []
    for layer in config.get('layer_info', []):
        layer_info.append({
            'z_min': float(layer['z_min']),
            'z_max': float(layer['z_max']),
            'eps_r': float(layer.get('eps_r', 1.0)),
            'mu_r': float(layer.get('mu_r', 1.0)),
            'sigma': float(layer.get('sigma', 0.01)),
        })

    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    sampling_cfg = config.get('sampling', {})
    loss_cfg = config.get('loss', {})
    data_cfg = config.get('data', {})
    output_cfg = config.get('output', {})

    return {
        'layer_info': layer_info,
        'source_pos': config.get('source_pos', [0.0, 0.0, -25.0]),
        'rho_range': tuple(config.get('rho_range', [0.0, 50.0])),
        'z_range': tuple(config.get('z_range', [0.0, 100.0])),
        'frequencies': config.get('frequencies', [0.1, 100.0, 2000.0, 1e7, 3e10]),

        'model': {
            'num_subnets': model_cfg.get('num_subnets', 10),
            'input_dim': model_cfg.get('input_dim', 8),
            'fourier_dim': model_cfg.get('fourier_dim', 64),
            'hidden_dims': model_cfg.get('hidden_dims', [128, 200, 200, 200, 200, 128]),
            'output_dim': model_cfg.get('output_dim', 14),
            'scale_factors': model_cfg.get('scale_factors', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
        },

        'training': {
            'phase1_epochs': training_cfg.get('phase1_epochs', 8000),
            'phase2_epochs': training_cfg.get('phase2_epochs', 2000),
            'lr': training_cfg.get('lr', 0.001),
            'weight_decay': training_cfg.get('weight_decay', 1e-4),
            'warmup_epochs': training_cfg.get('warmup_epochs', 1000),
            'resample_interval': training_cfg.get('resample_interval', 10),
            'grad_clip': training_cfg.get('grad_clip', 1.0),
        },

        'sampling': {
            'n_interior': sampling_cfg.get('n_interior', 4000),
            'n_gauge': sampling_cfg.get('n_gauge', 2000),
            'n_interface': sampling_cfg.get('n_interface', 700),
            'n_radiation': sampling_cfg.get('n_radiation', 700),
            'n_data': sampling_cfg.get('n_data', 2000),
            'n_sym': sampling_cfg.get('n_sym', 1000),
            'delta_z': sampling_cfg.get('delta_z', 0.5),
            'R_min': sampling_cfg.get('R_min', 0.1),
        },

        'loss': {
            'adaptive': loss_cfg.get('adaptive', True),
            'init_weights': loss_cfg.get('init_weights', [1.0, 1.0, 1.0, 1.0, 1.0]),
            'beta1': loss_cfg.get('beta1', 1.0),
            'beta2': loss_cfg.get('beta2', 1.0),
            'eps': loss_cfg.get('eps', 1e-8),
        },

        'data': {
            'csv_path': data_cfg.get('csv_path', None),
        },

        'output': {
            'checkpoint_dir': output_cfg.get('checkpoint_dir', 'checkpoints'),
            'loss_history_dir': output_cfg.get('loss_history_dir', 'checkpoints'),
        },
    }


def config_to_frpinn_fno_args(config):
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    sampling_cfg = config.get('sampling', {})
    loss_cfg = config.get('loss', {})
    data_cfg = config.get('data', {})
    output_cfg = config.get('output', {})
    medium_cfg = config.get('medium_library', {})

    base_models = []
    for bm in medium_cfg.get('base_models', []):
        base_models.append(bm)

    return {
        'z_min': float(config.get('z_min', 0.0)),
        'z_max': float(config.get('z_max', 800e3)),
        'nz': int(config.get('nz', 64)),
        'source_pos': config.get('source_pos', [0.0, 0.0, -10e3]),
        'rho_range': tuple(config.get('rho_range', [0.0, 800e3])),
        'z_range': tuple(config.get('z_range', [0.0, 800e3])),
        'frequencies': config.get('frequencies', [0.0001, 0.1, 1.0, 10.0, 100.0]),
        'eps_r_range': tuple(config.get('eps_r_range', [1.0, 10.0])),
        'mu_r_range': tuple(config.get('mu_r_range', [1.0, 1.0])),
        'sigma_range': tuple(config.get('sigma_range', [1e-4, 1.0])),

        'model': {
            'num_subnets': model_cfg.get('num_subnets', 10),
            'input_dim': model_cfg.get('input_dim', 8),
            'fourier_dim': model_cfg.get('fourier_dim', 64),
            'hidden_dims': model_cfg.get('hidden_dims', [128, 200, 200, 200, 200, 128]),
            'output_dim': model_cfg.get('output_dim', 14),
            'scale_factors': model_cfg.get('scale_factors', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
            'nz': model_cfg.get('nz', 64),
            'medium_channels': model_cfg.get('medium_channels', 4),
            'fno_hidden_dim': model_cfg.get('fno_hidden_dim', 64),
            'num_fno_layers': model_cfg.get('num_fno_layers', 4),
            'num_modes': model_cfg.get('num_modes', 12),
            'film_dim': model_cfg.get('film_dim', 128),
        },

        'training': {
            'phase1_epochs': training_cfg.get('phase1_epochs', 8000),
            'phase2_epochs': training_cfg.get('phase2_epochs', 2000),
            'lr': training_cfg.get('lr', 0.001),
            'weight_decay': training_cfg.get('weight_decay', 1e-4),
            'warmup_epochs': training_cfg.get('warmup_epochs', 1000),
            'resample_interval': training_cfg.get('resample_interval', 10),
            'grad_clip': training_cfg.get('grad_clip', 1.0),
            'n_medium_per_epoch_phase1': training_cfg.get('n_medium_per_epoch_phase1', 5),
            'n_medium_per_epoch_phase2': training_cfg.get('n_medium_per_epoch_phase2', 3),
        },

        'sampling': {
            'n_interior': sampling_cfg.get('n_interior', 3000),
            'n_gauge': sampling_cfg.get('n_gauge', 1500),
            'n_interface': sampling_cfg.get('n_interface', 500),
            'n_radiation': sampling_cfg.get('n_radiation', 300),
            'n_data': sampling_cfg.get('n_data', 800),
            'n_sym': sampling_cfg.get('n_sym', 500),
            'delta_z': sampling_cfg.get('delta_z', 0.5),
            'R_min': sampling_cfg.get('R_min', 0.1),
        },

        'loss': {
            'adaptive': loss_cfg.get('adaptive', True),
            'init_weights': loss_cfg.get('init_weights', [1.0, 1.0, 1.0, 1.0, 1.0]),
            'beta1': loss_cfg.get('beta1', 1.0),
            'beta2': loss_cfg.get('beta2', 1.0),
            'eps': loss_cfg.get('eps', 1e-8),
        },

        'medium_library': {
            'n_perturbed_models': medium_cfg.get('n_perturbed_models', 30),
            'seed': medium_cfg.get('seed', 42),
            'base_models': base_models,
        },

        'data': {
            'csv_path': data_cfg.get('csv_path', None),
        },

        'output': {
            'checkpoint_dir': output_cfg.get('checkpoint_dir', 'checkpoints'),
            'loss_history_dir': output_cfg.get('loss_history_dir', 'checkpoints'),
        },
    }
