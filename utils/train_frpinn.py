import os
import sys
import time
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.frpinn import FRPINNs
from models.common import compute_gradient_norm, compute_weight_norm, check_model_health
from losses.frpinn_loss import FRPINNsLoss
from data.data_loader import (
    LMGFDataset, PhysicsSampler, FRPINNsBatchBuilder,
    create_default_config
)
from config_utils import load_config, config_to_frpinn_args


class EarlyStopping:
    def __init__(self, patience=500, min_delta=1e-8, mode='min',
                 restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_value = None
        self.best_state = None
        self.best_loss_state = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, value, model=None, loss_fn=None, epoch=0):
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            if model is not None:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if loss_fn is not None:
                self.best_loss_state = {k: v.clone() for k, v in loss_fn.state_dict().items()}
            return False

        improved = False
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            if model is not None:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if loss_fn is not None:
                self.best_loss_state = {k: v.clone() for k, v in loss_fn.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def restore_best_weights(self, model, loss_fn=None):
        if self.best_state is not None and self.restore_best:
            model.load_state_dict(self.best_state)
            if loss_fn is not None and self.best_loss_state is not None:
                loss_fn.load_state_dict(self.best_loss_state)
            print(f"Restored best model from epoch {self.best_epoch} "
                  f"(loss={self.best_value:.6e})")


class CheckpointManager:
    def __init__(self, save_dir='checkpoints', max_keep=5, save_best=True):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.save_best = save_best
        self.best_loss = float('inf')
        self.checkpoint_list = []
        os.makedirs(save_dir, exist_ok=True)

    def save(self, state, epoch, loss, is_best=False):
        filename = f'checkpoint_epoch_{epoch:06d}.pt'
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)
        self.checkpoint_list.append(filepath)

        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(state, best_path)

        while len(self.checkpoint_list) > self.max_keep:
            old_path = self.checkpoint_list.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        return filepath

    def load_latest(self):
        if not self.checkpoint_list:
            return None
        latest = self.checkpoint_list[-1]
        return torch.load(latest, map_location='cpu')

    def load_best(self):
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        if os.path.exists(best_path):
            return torch.load(best_path, map_location='cpu')
        return None

    def load_checkpoint(self, path):
        if os.path.exists(path):
            return torch.load(path, map_location='cpu')
        return None


class GradientMonitor:
    def __init__(self, log_interval=100, alert_threshold=100.0,
                 nan_alert=True):
        self.log_interval = log_interval
        self.alert_threshold = alert_threshold
        self.nan_alert = nan_alert
        self.gradient_norms_history = []
        self.weight_norms_history = []
        self.nan_count = 0
        self.inf_count = 0
        self.step_count = 0

    def log_gradients(self, model, epoch=None):
        self.step_count += 1
        if self.step_count % self.log_interval != 0:
            return

        total_norm, per_layer = compute_gradient_norm(model)
        self.gradient_norms_history.append({
            'step': self.step_count,
            'epoch': epoch,
            'total_norm': total_norm,
            'per_layer': per_layer,
        })

        if total_norm > self.alert_threshold:
            print(f"  [WARNING] Large gradient norm: {total_norm:.4e} at step {self.step_count}")

        for name, norm in per_layer.items():
            if math.isnan(norm):
                self.nan_count += 1
                if self.nan_alert:
                    print(f"  [ALERT] NaN gradient in: {name}")
            if math.isinf(norm):
                self.inf_count += 1
                if self.nan_alert:
                    print(f"  [ALERT] Inf gradient in: {name}")

    def log_weights(self, model, epoch=None):
        total_norm, per_layer = compute_weight_norm(model)
        self.weight_norms_history.append({
            'step': self.step_count,
            'epoch': epoch,
            'total_norm': total_norm,
            'per_layer': per_layer,
        })

    def get_gradient_summary(self):
        if not self.gradient_norms_history:
            return {}
        norms = [h['total_norm'] for h in self.gradient_norms_history]
        return {
            'mean_norm': np.mean(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'n_records': len(norms),
        }

    def get_weight_summary(self):
        if not self.weight_norms_history:
            return {}
        norms = [h['total_norm'] for h in self.weight_norms_history]
        return {
            'mean_norm': np.mean(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'n_records': len(norms),
        }

    def plot_gradient_norms(self, save_path=None):
        if not self.gradient_norms_history:
            return
        steps = [h['step'] for h in self.gradient_norms_history]
        norms = [h['total_norm'] for h in self.gradient_norms_history]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogy(steps, norms, linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm During Training')
        ax.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class TrainingLogger:
    def __init__(self, log_dir='logs', log_file='training.log'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self._file = open(self.log_path, 'a', encoding='utf-8')

    def log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {message}"
        self._file.write(line + '\n')
        self._file.flush()

    def log_config(self, config):
        self.log("Configuration:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")

    def log_epoch(self, epoch, total_epochs, loss_dict, elapsed=None,
                  extra_info=None):
        parts = [f"Epoch {epoch+1}/{total_epochs}"]
        parts.append(f"Loss={loss_dict.get('total', 0.0):.6e}")
        for name in ['pde', 'gauge', 'bc', 'data', 'sym']:
            if name in loss_dict:
                parts.append(f"{name}={loss_dict[name]:.6e}")
        if elapsed is not None:
            parts.append(f"Time={elapsed:.1f}s")
        if extra_info:
            parts.append(extra_info)
        self.log(" | ".join(parts))

    def close(self):
        self._file.close()


class LRSchedulerFactory:
    @staticmethod
    def create(optimizer, scheduler_type='cosine_warmup', **kwargs):
        if scheduler_type == 'cosine_warmup':
            warmup = kwargs.get('warmup_epochs', 1000)
            total = kwargs.get('total_epochs', 8000)
            def lr_lambda(epoch):
                if epoch < warmup:
                    return epoch / max(1, warmup)
                progress = (epoch - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return LambdaLR(optimizer, lr_lambda)

        elif scheduler_type == 'exponential_decay':
            decay_rate = kwargs.get('decay_rate', 0.9995)
            def lr_lambda(epoch):
                return decay_rate ** epoch
            return LambdaLR(optimizer, lr_lambda)

        elif scheduler_type == 'step_decay':
            step_size = kwargs.get('step_size', 2000)
            gamma = kwargs.get('gamma', 0.5)
            def lr_lambda(epoch):
                return gamma ** (epoch // step_size)
            return LambdaLR(optimizer, lr_lambda)

        elif scheduler_type == 'cosine_annealing':
            T_max = kwargs.get('T_max', 8000)
            eta_min = kwargs.get('eta_min', 1e-6)
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        elif scheduler_type == 'reduce_on_plateau':
            patience = kwargs.get('patience', 500)
            factor = kwargs.get('factor', 0.5)
            min_lr = kwargs.get('min_lr', 1e-7)
            return ReduceLROnPlateau(optimizer, mode='min', patience=patience,
                                     factor=factor, min_lr=min_lr)

        elif scheduler_type == 'onecycle':
            max_lr = kwargs.get('max_lr', 0.01)
            total_steps = kwargs.get('total_steps', 8000)
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=total_steps
            )

        else:
            warmup = kwargs.get('warmup_epochs', 1000)
            total = kwargs.get('total_epochs', 8000)
            def lr_lambda(epoch):
                if epoch < warmup:
                    return epoch / max(1, warmup)
                progress = (epoch - warmup) / max(1, total - warmup)
                return 0.975 ** (progress * (total - warmup) / 100)
            return LambdaLR(optimizer, lr_lambda)


class EvaluationCallback:
    def __init__(self, eval_interval=1000, save_dir='evaluation'):
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.eval_history = []
        os.makedirs(save_dir, exist_ok=True)

    def should_evaluate(self, epoch):
        return (epoch + 1) % self.eval_interval == 0

    def evaluate(self, model, dataset, epoch, device='cpu',
                 medium_profiles=None):
        if not self.should_evaluate(epoch):
            return None

        model.eval()
        if len(dataset) == 0:
            return None

        n_eval = min(1000, len(dataset))
        indices = torch.randperm(len(dataset))[:n_eval]
        x_eval = dataset.data_x[indices].to(device)
        y_eval = dataset.data_y[indices].to(device)

        with torch.no_grad():
            if medium_profiles is not None:
                first_key = list(medium_profiles.keys())[0]
                mp = medium_profiles[first_key]
                pred = model(x_eval, mp)
            else:
                pred = model(x_eval)

        rel_l2 = torch.norm(pred - y_eval, p=2) / (torch.norm(y_eval, p=2) + 1e-30)
        rel_l1 = torch.norm(pred - y_eval, p=1) / (torch.norm(y_eval, p=1) + 1e-30)
        mae = (pred - y_eval).abs().mean()

        result = {
            'epoch': epoch,
            'rel_l2': rel_l2.item(),
            'rel_l1': rel_l1.item(),
            'mae': mae.item(),
        }
        self.eval_history.append(result)
        return result


class FRPINNsTrainer:
    def __init__(self, config=None, csv_path=None, device=None):
        if config is None:
            config = create_default_config()

        self.config = config
        self.layer_info = config['layer_info']
        self.frequencies = config['frequencies']
        self.source_pos = config['source_pos']
        self.rho_range = config['rho_range']
        self.z_range = config['z_range']

        model_cfg = config.get('model', {})
        sampling_cfg = config.get('sampling', {})
        loss_cfg = config.get('loss', {})

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = FRPINNs(
            num_subnets=model_cfg.get('num_subnets', 10),
            input_dim=model_cfg.get('input_dim', 8),
            fourier_dim=model_cfg.get('fourier_dim', 64),
            hidden_dims=model_cfg.get('hidden_dims', [128, 200, 200, 200, 200, 128]),
            output_dim=model_cfg.get('output_dim', 14),
            scale_factors=model_cfg.get('scale_factors', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
        ).to(self.device)

        self.loss_fn = FRPINNsLoss(
            layer_info=self.layer_info,
            frequencies=self.frequencies,
            eps=loss_cfg.get('eps', 1e-8),
            beta1=loss_cfg.get('beta1', 1.0),
            beta2=loss_cfg.get('beta2', 1.0),
            adaptive=loss_cfg.get('adaptive', True),
            rho_range=self.rho_range,
            use_complex_km=loss_cfg.get('use_complex_km', False),
            weighting_strategy=loss_cfg.get('weighting_strategy', 'ema'),
        ).to(self.device)

        self.dataset = LMGFDataset(csv_path=csv_path)

        self.sampler = PhysicsSampler(
            layer_info=self.layer_info,
            source_pos=self.source_pos,
            rho_range=self.rho_range,
            z_range=self.z_range,
            frequencies=self.frequencies,
            delta_z=sampling_cfg.get('delta_z', 0.5),
            R_min=sampling_cfg.get('R_min', 0.1),
        )

        self.batch_builder = FRPINNsBatchBuilder(
            sampler=self.sampler,
            dataset=self.dataset,
            n_interior=sampling_cfg.get('n_interior', 4000),
            n_gauge=sampling_cfg.get('n_gauge', 2000),
            n_interface=sampling_cfg.get('n_interface', 700),
            n_radiation=sampling_cfg.get('n_radiation', 700),
            n_data=sampling_cfg.get('n_data', 2000),
            n_sym=sampling_cfg.get('n_sym', 1000),
        )

        self.loss_history = []
        self.current_epoch = 0
        self.lr_history = []

        training_cfg = config.get('training', {})
        self.gradient_monitor = GradientMonitor(
            log_interval=training_cfg.get('grad_log_interval', 100),
            alert_threshold=training_cfg.get('grad_alert_threshold', 100.0),
        )
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.get('output', {}).get('checkpoint_dir', 'checkpoints'),
            max_keep=training_cfg.get('max_checkpoints', 5),
        )
        self.eval_callback = EvaluationCallback(
            eval_interval=training_cfg.get('eval_interval', 1000),
            save_dir=config.get('output', {}).get('eval_dir', 'evaluation'),
        )
        self.logger = TrainingLogger(
            log_dir=config.get('output', {}).get('log_dir', 'logs'),
            log_file='frpinn_training.log',
        )

    def _get_adamw_optimizer(self, lr=0.001, weight_decay=1e-4):
        model_params = list(self.model.parameters())
        loss_params = list(self.loss_fn.parameters())
        all_params = model_params + loss_params
        return optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    def _get_lbfgs_optimizer(self, lr=1.0, max_iter=20):
        model_params = list(self.model.parameters())
        loss_params = list(self.loss_fn.parameters())
        all_params = model_params + loss_params
        return optim.LBFGS(all_params, lr=lr, max_iter=max_iter,
                           tolerance_grad=1e-9, tolerance_change=1e-12)

    def _get_scheduler(self, optimizer, scheduler_type='cosine_warmup', **kwargs):
        return LRSchedulerFactory.create(optimizer, scheduler_type, **kwargs)

    def _get_cosine_scheduler(self, optimizer, warmup_epochs=1000, total_epochs=8000):
        return self._get_scheduler(optimizer, 'cosine_warmup',
                                   warmup_epochs=warmup_epochs,
                                   total_epochs=total_epochs)

    def _check_model_health(self):
        health = check_model_health(self.model)
        if not health['healthy']:
            for issue in health['issues']:
                self.logger.log(f"[HEALTH] {issue}")
        return health

    def _log_epoch_info(self, epoch, n_epochs, loss_dict, start_time):
        total_loss = loss_dict.get('total', 0.0)
        elapsed = time.time() - start_time
        w_info = ""
        if 'w_pde' in loss_dict:
            w_info = (f" | w_pde={loss_dict['w_pde']:.4f}"
                      f" w_gauge={loss_dict['w_gauge']:.4f}"
                      f" w_bc={loss_dict['w_bc']:.4f}"
                      f" w_data={loss_dict['w_data']:.4f}"
                      f" w_sym={loss_dict['w_sym']:.4f}")
        msg = (f"Epoch {epoch+1}/{n_epochs} | "
               f"Loss={total_loss:.6e} | "
               f"PDE={loss_dict.get('pde',0):.6e} "
               f"Gauge={loss_dict.get('gauge',0):.6e} "
               f"BC={loss_dict.get('bc',0):.6e} "
               f"Data={loss_dict.get('data',0):.6e} "
               f"Sym={loss_dict.get('sym',0):.6e}"
               f"{w_info} | Time={elapsed:.1f}s")
        print(msg)
        self.logger.log(msg)

    def train_phase1(self, n_epochs=8000, lr=0.001, warmup_epochs=1000,
                     resample_interval=10, grad_clip=1.0,
                     scheduler_type='cosine_warmup',
                     early_stopping_patience=0,
                     checkpoint_interval=1000):
        print(f"Phase 1: AdamW training for {n_epochs} epochs")
        print(f"Device: {self.device}")

        optimizer = self._get_adamw_optimizer(lr=lr)
        scheduler = self._get_scheduler(
            optimizer, scheduler_type,
            warmup_epochs=warmup_epochs, total_epochs=n_epochs
        )

        early_stopper = None
        if early_stopping_patience > 0:
            early_stopper = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=1e-8,
                mode='min',
                restore_best=True,
            )

        start_time = time.time()
        for epoch in range(n_epochs):
            self.current_epoch = epoch

            if epoch % resample_interval == 0:
                batches = {}
                for freq_idx in range(len(self.frequencies)):
                    batches[freq_idx] = self.batch_builder.build_batch(
                        freq_idx, self.device
                    )

            total_loss_epoch = 0.0
            loss_dict_epoch = {}

            for freq_idx in range(len(self.frequencies)):
                optimizer.zero_grad()
                total_loss, loss_dict = self.loss_fn(
                    self.model, batches[freq_idx], freq_idx, epoch=epoch
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip
                )
                optimizer.step()

                total_loss_epoch += total_loss.item()
                for k, v in loss_dict.items():
                    loss_dict_epoch[k] = loss_dict_epoch.get(k, 0.0) + v

            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            else:
                scheduler.step(total_loss_epoch / len(self.frequencies))

            current_lr = optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

            n_freq = len(self.frequencies)
            for k in loss_dict_epoch:
                loss_dict_epoch[k] /= n_freq
            total_loss_epoch /= n_freq
            loss_dict_epoch['total'] = total_loss_epoch

            self.loss_history.append(loss_dict_epoch)

            self.gradient_monitor.log_gradients(self.model, epoch)

            if (epoch + 1) % 500 == 0 or epoch == 0:
                self._log_epoch_info(epoch, n_epochs, loss_dict_epoch, start_time)

            if (epoch + 1) % 2000 == 0:
                self.gradient_monitor.log_weights(self.model, epoch)

            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'loss_fn_state_dict': self.loss_fn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': self.loss_history,
                    'config': self.config,
                }
                self.checkpoint_manager.save(state, epoch, total_loss_epoch)

            eval_result = self.eval_callback.evaluate(
                self.model, self.dataset, epoch, self.device
            )
            if eval_result is not None:
                self.logger.log(
                    f"  [EVAL] Epoch {epoch+1}: "
                    f"rel_l2={eval_result['rel_l2']:.6e} "
                    f"rel_l1={eval_result['rel_l1']:.6e} "
                    f"mae={eval_result['mae']:.6e}"
                )

            if early_stopper is not None:
                if early_stopper.step(total_loss_epoch, self.model, self.loss_fn, epoch):
                    print(f"Early stopping at epoch {epoch+1}")
                    early_stopper.restore_best_weights(self.model, self.loss_fn)
                    break

        print(f"Phase 1 completed in {time.time() - start_time:.1f}s")

    def train_phase2(self, n_epochs=2000, lr=1.0, max_iter=20,
                     resample_interval=10, early_stopping_patience=0,
                     checkpoint_interval=500):
        print(f"Phase 2: L-BFGS fine-tuning for {n_epochs} epochs")

        optimizer = self._get_lbfgs_optimizer(lr=lr, max_iter=max_iter)

        early_stopper = None
        if early_stopping_patience > 0:
            early_stopper = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=1e-10,
                mode='min',
                restore_best=True,
            )

        start_time = time.time()
        for epoch in range(n_epochs):
            self.current_epoch = 8000 + epoch

            if epoch % resample_interval == 0:
                batches = {}
                for freq_idx in range(len(self.frequencies)):
                    batches[freq_idx] = self.batch_builder.build_batch(
                        freq_idx, self.device
                    )

            total_loss_epoch = 0.0
            loss_dict_epoch = {}

            for freq_idx in range(len(self.frequencies)):
                batch = batches[freq_idx]

                closure_loss_dict = {}

                def closure(b=batch, fi=freq_idx, ep=self.current_epoch,
                            cld=closure_loss_dict):
                    optimizer.zero_grad()
                    loss, ld = self.loss_fn(self.model, b, fi, epoch=ep)
                    loss.backward()
                    cld.update(ld)
                    return loss

                optimizer.step(closure)

                loss_dict = closure_loss_dict
                total_loss_epoch += loss_dict.get('total', 0.0)
                for k, v in loss_dict.items():
                    loss_dict_epoch[k] = loss_dict_epoch.get(k, 0.0) + v

            n_freq = len(self.frequencies)
            for k in loss_dict_epoch:
                loss_dict_epoch[k] /= n_freq
            total_loss_epoch /= n_freq
            loss_dict_epoch['total'] = total_loss_epoch

            self.loss_history.append(loss_dict_epoch)

            if (epoch + 1) % 200 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                self._log_epoch_info(self.current_epoch, 8000 + n_epochs,
                                     loss_dict_epoch, start_time)

            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                state = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'loss_fn_state_dict': self.loss_fn.state_dict(),
                    'loss_history': self.loss_history,
                    'config': self.config,
                }
                self.checkpoint_manager.save(state, self.current_epoch, total_loss_epoch)

            if early_stopper is not None:
                if early_stopper.step(total_loss_epoch, self.model, self.loss_fn,
                                      self.current_epoch):
                    print(f"Early stopping at L-BFGS epoch {epoch+1}")
                    early_stopper.restore_best_weights(self.model, self.loss_fn)
                    break

        print(f"Phase 2 completed in {time.time() - start_time:.1f}s")

    def train(self, phase1_epochs=8000, phase2_epochs=2000,
              lr=0.001, warmup_epochs=1000, resample_interval=10,
              grad_clip=1.0, scheduler_type='cosine_warmup',
              early_stopping_patience=0, checkpoint_interval=1000):
        self.train_phase1(n_epochs=phase1_epochs, lr=lr,
                          warmup_epochs=warmup_epochs,
                          resample_interval=resample_interval,
                          grad_clip=grad_clip,
                          scheduler_type=scheduler_type,
                          early_stopping_patience=early_stopping_patience,
                          checkpoint_interval=checkpoint_interval)
        self.train_phase2(n_epochs=phase2_epochs,
                          resample_interval=resample_interval,
                          early_stopping_patience=early_stopping_patience,
                          checkpoint_interval=checkpoint_interval)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
        return pred

    def predict_on_grid(self, rho_range, z_range, freq_idx, n_rho=100, n_z=100,
                        source_pos=None):
        self.model.eval()
        rho_vals = np.linspace(rho_range[0], rho_range[1], n_rho)
        z_vals = np.linspace(z_range[0], z_range[1], n_z)
        rho_grid, z_grid = np.meshgrid(rho_vals, z_vals)

        if source_pos is None:
            source_pos = self.source_pos

        freq = self.frequencies[freq_idx]
        f_log = np.log10(freq) if freq > 0 else 0.0

        x_flat = np.zeros((n_rho * n_z, 8))
        x_flat[:, 0] = rho_grid.flatten()
        x_flat[:, 2] = z_grid.flatten()
        x_flat[:, 3] = source_pos[0]
        x_flat[:, 4] = source_pos[1]
        x_flat[:, 5] = source_pos[2]
        x_flat[:, 6] = rho_grid.flatten()
        x_flat[:, 7] = f_log

        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(x_tensor)

        pred_grid = pred.cpu().numpy().reshape(n_z, n_rho, -1)
        return rho_grid, z_grid, pred_grid

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        self.lr_history = checkpoint.get('lr_history', [])
        print(f"Model loaded from {path}")

    def save_loss_history(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        save_data = {
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
            'gradient_summary': self.gradient_monitor.get_gradient_summary(),
            'weight_summary': self.gradient_monitor.get_weight_summary(),
            'eval_history': self.eval_callback.eval_history,
        }
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Loss history saved to {path}")

    def plot_loss_history(self, save_dir=None):
        if len(self.loss_history) == 0:
            print("No loss history to plot.")
            return

        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'checkpoints'
            )
        os.makedirs(save_dir, exist_ok=True)

        loss_names = ['pde', 'gauge', 'bc', 'data', 'sym']
        loss_labels = ['PDE Residual', 'Lorenz Gauge', 'Boundary Cond.', 'Data Fitting', 'Symmetry']
        loss_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        epochs = list(range(len(self.loss_history)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, (name, label, color) in enumerate(zip(loss_names, loss_labels, loss_colors)):
            ax = axes[i]
            values = [h.get(name, 0.0) for h in self.loss_history]
            ax.semilogy(epochs, values, color=color, linewidth=1.0, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        ax_total = axes[5]
        total_values = [h.get('total', 0.0) for h in self.loss_history]
        ax_total.semilogy(epochs, total_values, color='#2c3e50', linewidth=1.5)
        ax_total.set_xlabel('Epoch')
        ax_total.set_ylabel('Loss')
        ax_total.set_title('Total Loss')
        ax_total.grid(True, alpha=0.3)

        fig.suptitle('FRPINNs Training Loss Convergence', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'frpinn_loss_convergence.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss convergence plot saved to {save_path}")

    def plot_lr_schedule(self, save_dir=None):
        if not self.lr_history:
            return
        if save_dir is None:
            save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.lr_history, linewidth=1.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        save_path = os.path.join(save_dir, 'frpinn_lr_schedule.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_gradient_norms(self, save_dir=None):
        if save_dir is None:
            save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'frpinn_gradient_norms.png')
        self.gradient_monitor.plot_gradient_norms(save_path)

    def generate_training_report(self, save_dir=None):
        if save_dir is None:
            save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        report_path = os.path.join(save_dir, 'training_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FRPINNs Training Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Device: {self.device}\n")
            f.write(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"Total epochs: {len(self.loss_history)}\n\n")

            if self.loss_history:
                final = self.loss_history[-1]
                f.write("Final Loss Values:\n")
                for name in ['pde', 'gauge', 'bc', 'data', 'sym', 'total']:
                    f.write(f"  {name}: {final.get(name, 0.0):.6e}\n")
                f.write("\n")

                initial = self.loss_history[0]
                f.write("Loss Reduction:\n")
                for name in ['pde', 'gauge', 'bc', 'data', 'sym', 'total']:
                    init_val = initial.get(name, 0.0)
                    final_val = final.get(name, 0.0)
                    if init_val > 0:
                        ratio = init_val / (final_val + 1e-30)
                        f.write(f"  {name}: {init_val:.6e} -> {final_val:.6e} (x{ratio:.1f})\n")
                f.write("\n")

            grad_summary = self.gradient_monitor.get_gradient_summary()
            if grad_summary:
                f.write("Gradient Statistics:\n")
                for k, v in grad_summary.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            health = check_model_health(self.model)
            f.write(f"Model Health: {'Healthy' if health['healthy'] else 'Issues Found'}\n")
            if not health['healthy']:
                for issue in health['issues']:
                    f.write(f"  - {issue}\n")

        print(f"Training report saved to {report_path}")
        return report_path

    def plot_weight_norms(self, save_dir=None):
        if save_dir is None:
            save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        weight_history = self.gradient_monitor.weight_norms_history
        if not weight_history:
            return

        steps = [h['step'] for h in weight_history]
        norms = [h['total_norm'] for h in weight_history]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogy(steps, norms, linewidth=0.8, alpha=0.7, color='#e67e22')
        ax.set_xlabel('Step')
        ax.set_ylabel('Weight Norm')
        ax.set_title('Weight Norm During Training')
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(save_dir, 'frpinn_weight_norms.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Weight norms plot saved to {save_path}")

    def plot_eval_history(self, save_dir=None):
        if not self.eval_callback.eval_history:
            return
        if save_dir is None:
            save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        epochs = [h['epoch'] for h in self.eval_callback.eval_history]
        rel_l2 = [h['rel_l2'] for h in self.eval_callback.eval_history]
        rel_l1 = [h['rel_l1'] for h in self.eval_callback.eval_history]
        mae = [h['mae'] for h in self.eval_callback.eval_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].semilogy(epochs, rel_l2, 'b-o', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Relative L2 Error')
        axes[0].set_title('Relative L2 Error')
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(epochs, rel_l1, 'g-o', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Relative L1 Error')
        axes[1].set_title('Relative L1 Error')
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(epochs, mae, 'r-o', markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MAE')
        axes[2].set_title('Mean Absolute Error')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Evaluation Metrics During Training', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'frpinn_eval_metrics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Evaluation metrics plot saved to {save_path}")

    def resume_training(self, checkpoint_path, phase1_epochs=0, phase2_epochs=2000,
                        lr=0.001, warmup_epochs=1000, resample_interval=10,
                        grad_clip=1.0, scheduler_type='cosine_warmup',
                        early_stopping_patience=0, checkpoint_interval=1000):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

        if phase1_epochs > 0:
            self.train_phase1(
                n_epochs=phase1_epochs, lr=lr,
                warmup_epochs=warmup_epochs,
                resample_interval=resample_interval,
                grad_clip=grad_clip,
                scheduler_type=scheduler_type,
                early_stopping_patience=early_stopping_patience,
                checkpoint_interval=checkpoint_interval,
            )
        if phase2_epochs > 0:
            self.train_phase2(
                n_epochs=phase2_epochs,
                resample_interval=resample_interval,
                early_stopping_patience=early_stopping_patience,
                checkpoint_interval=checkpoint_interval,
            )

    def compute_spectral_analysis(self, freq_idx=0, n_points=5000):
        self.model.eval()
        sampler = self.sampler
        x = sampler.sample_interior(n_points, freq_idx, self.device)
        with torch.no_grad():
            pred = self.model(x)

        components = ['Kxx_Re', 'Kxx_Im', 'Kzz_Re', 'Kzz_Im',
                       'Kphi_Re', 'Kphi_Im']
        analysis = {}
        for i, comp in enumerate(components):
            field = pred[:, i].cpu().numpy()
            fft_vals = np.fft.fft(field)
            power_spectrum = np.abs(fft_vals) ** 2
            freqs = np.fft.fftfreq(len(field))
            analysis[comp] = {
                'mean': float(np.mean(field)),
                'std': float(np.std(field)),
                'min': float(np.min(field)),
                'max': float(np.max(field)),
                'spectral_peak_freq': float(freqs[np.argmax(power_spectrum[1:]) + 1]),
                'spectral_peak_power': float(np.max(power_spectrum[1:])),
                'total_power': float(np.sum(power_spectrum)),
            }
        return analysis

    def export_predictions(self, output_dir, freq_idx=0, n_rho=100, n_z=100):
        os.makedirs(output_dir, exist_ok=True)
        rho_grid, z_grid, pred_grid = self.predict_on_grid(
            self.rho_range, self.z_range, freq_idx, n_rho, n_z
        )

        np.savez(
            os.path.join(output_dir, f'predictions_freq{freq_idx}.npz'),
            rho_grid=rho_grid,
            z_grid=z_grid,
            predictions=pred_grid,
        )

        components = ['Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
                       'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
                       'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
                       'Kphi_Re', 'Kphi_Im']
        summary = {}
        for i, comp in enumerate(components):
            field = pred_grid[:, :, i]
            summary[comp] = {
                'min': float(np.min(field)),
                'max': float(np.max(field)),
                'mean': float(np.mean(field)),
                'std': float(np.std(field)),
            }

        with open(os.path.join(output_dir, f'prediction_summary_freq{freq_idx}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Predictions exported to {output_dir}")
        return summary

    def cleanup(self):
        self.logger.close()


def main():
    parser = argparse.ArgumentParser(description='FRPINNs Training - Chapter 3')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.config is not None:
        print(f"Loading config from: {args.config}")
        raw_config = load_config(args.config)
        config = config_to_frpinn_args(raw_config)
    else:
        default_config_path = os.path.join(project_root, 'config_frpinn.yaml')
        if os.path.exists(default_config_path):
            print(f"Loading default config from: {default_config_path}")
            raw_config = load_config(default_config_path)
            config = config_to_frpinn_args(raw_config)
        else:
            print("No config file found, using built-in defaults")
            config = create_default_config()

    csv_path = config.get('data', {}).get('csv_path', None)
    if csv_path is None:
        csv_path = os.path.join(project_root, 'data', 'lmgf_data.csv')

    if not os.path.exists(csv_path):
        print(f"Warning: CSV data file not found at {csv_path}")
        print("Creating placeholder CSV file...")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            import csv as csv_mod
            writer = csv_mod.writer(f)
            header = ['x', 'y', 'z', 'xp', 'yp', 'zp', 'rho', 'f_log',
                      'Kxx_Re', 'Kxx_Im', 'Kxz_Re', 'Kxz_Im',
                      'Kyz_Re', 'Kyz_Im', 'Kzx_Re', 'Kzx_Im',
                      'Kzy_Re', 'Kzy_Im', 'Kzz_Re', 'Kzz_Im',
                      'Kphi_Re', 'Kphi_Im']
            writer.writerow(header)
            writer.writerow([0.0] * 22)
        print(f"Placeholder CSV created at {csv_path}")
        print("Please replace with actual LMGF data from Strata.")

    trainer = FRPINNsTrainer(config=config, csv_path=csv_path, device=args.device)

    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume)

    training_cfg = config.get('training', {})

    print("=" * 60)
    print("FRPINNs Training - Chapter 3")
    print("=" * 60)
    print(f"Frequencies: {config['frequencies']}")
    print(f"Layers: {len(config['layer_info'])}")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("=" * 60)

    trainer.train(
        phase1_epochs=training_cfg.get('phase1_epochs', 8000),
        phase2_epochs=training_cfg.get('phase2_epochs', 2000),
        lr=training_cfg.get('lr', 0.001),
        warmup_epochs=training_cfg.get('warmup_epochs', 1000),
        resample_interval=training_cfg.get('resample_interval', 10),
        grad_clip=training_cfg.get('grad_clip', 1.0),
        scheduler_type=training_cfg.get('scheduler_type', 'cosine_warmup'),
        early_stopping_patience=training_cfg.get('early_stopping_patience', 0),
        checkpoint_interval=training_cfg.get('checkpoint_interval', 1000),
    )

    output_cfg = config.get('output', {})
    save_dir = os.path.join(project_root, output_cfg.get('checkpoint_dir', 'checkpoints'))
    trainer.save_model(os.path.join(save_dir, 'frpinn_model.pt'))
    trainer.save_loss_history(os.path.join(save_dir, 'frpinn_loss_history.json'))
    trainer.plot_loss_history(save_dir=save_dir)
    trainer.plot_lr_schedule(save_dir=save_dir)
    trainer.plot_gradient_norms(save_dir=save_dir)
    trainer.generate_training_report(save_dir=save_dir)
    trainer.cleanup()


if __name__ == '__main__':
    main()
