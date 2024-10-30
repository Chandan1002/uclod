import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import logging
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from datetime import datetime
import psutil
import GPUtil
from collections import defaultdict
import time
import shutil

class MetricsTracker:
    """Tracks and aggregates training metrics"""
    def __init__(self, window_size: int = 100):
        self.metrics = defaultdict(list)
        self.window_size = window_size
        self.step_times = []
        self.start_time = time.time()
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def step_time(self):
        """Record time for one step"""
        current_time = time.time()
        self.step_times.append(current_time - self.start_time)
        self.start_time = current_time
        if len(self.step_times) > self.window_size:
            self.step_times = self.step_times[-self.window_size:]
    
    def get_averages(self) -> Dict[str, float]:
        """Get averaged metrics over window"""
        return {k: np.mean(v) for k, v in self.metrics.items()}
    
    def get_step_time(self) -> float:
        """Get average step time"""
        return np.mean(self.step_times) if self.step_times else 0.0

class ResourceMonitor:
    """Monitors system resources during training"""
    def __init__(self, device: torch.device):
        self.device = device
        
    def get_stats(self) -> Dict[str, float]:
        """Get current system resource statistics"""
        stats = {}
        
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        stats.update({
            'cpu_percent': cpu_percent,
            'ram_used_gb': ram.used / (1024**3),
            'ram_percent': ram.percent
        })
        
        # GPU stats (if available)
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[self.device.index]
                stats.update({
                    'gpu_util': gpu.load * 100,
                    'gpu_mem_util': gpu.memoryUtil * 100,
                    'gpu_mem_used': torch.cuda.memory_allocated(self.device) / (1024**3),
                    'gpu_mem_cached': torch.cuda.memory_reserved(self.device) / (1024**3)
                })
            except Exception as e:
                logging.warning(f"Error getting GPU stats: {e}")
        
        return stats

class GradientMonitor:
    """Monitors gradient statistics during training"""
    def __init__(self, model: nn.Module):
        self.model = model
        
    def get_stats(self) -> Dict[str, float]:
        """Get gradient statistics for model parameters"""
        stats = {}
        total_norm = 0.0
        param_norm = 0.0
        
        # Calculate gradient norms
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Per-layer gradient stats
                stats[f'grad_norm/{name}'] = param_norm.item()
                stats[f'param_norm/{name}'] = param.data.norm(2).item()
                
                # Check for anomalies
                if torch.isnan(param.grad).any():
                    stats[f'grad_nan/{name}'] = 1.0
                if torch.isinf(param.grad).any():
                    stats[f'grad_inf/{name}'] = 1.0
        
        total_norm = total_norm ** 0.5
        stats['grad_norm_total'] = total_norm
        
        return stats

class MixedPrecisionMonitor:
    """Monitors mixed precision training statistics"""
    def __init__(self, scaler: GradScaler):
        self.scaler = scaler
        self.stats = defaultdict(list)
        
    def get_stats(self) -> Dict[str, float]:
        """Get mixed precision training statistics"""
        if self.scaler is None:
            return {}
            
        stats = {
            'grad_scale': self.scaler.get_scale(),
            'growth_tracker': self.scaler._growth_tracker,
        }
        
        # Track scale history
        self.stats['scale_history'].append(stats['grad_scale'])
        if len(self.stats['scale_history']) > 100:
            self.stats['scale_history'] = self.stats['scale_history'][-100:]
            
        stats['scale_mean'] = np.mean(self.stats['scale_history'])
        stats['scale_std'] = np.std(self.stats['scale_history'])
        
        return stats

def setup_logging(
    save_dir: str,
    log_config: Dict[str, Any]
) -> None:
    """Setup logging configuration"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = save_dir / f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log config
    logging.info("Training configuration:")
    logging.info(json.dumps(log_config, indent=2))

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    iteration: int,
    loss: float,
    save_dir: str,
    is_best: bool = False,
    max_keep: int = 5
) -> None:
    """Save model checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save latest checkpoint
    latest_path = save_dir / 'checkpoint_latest.pth'
    torch.save(state, latest_path)
    
    # Save numbered checkpoint
    numbered_path = save_dir / f'checkpoint_{iteration}.pth'
    torch.save(state, numbered_path)
    
    # Save best checkpoint if needed
    if is_best:
        best_path = save_dir / 'checkpoint_best.pth'
        shutil.copyfile(numbered_path, best_path)
    
    # Remove old checkpoints if needed
    checkpoints = sorted(save_dir.glob('checkpoint_[0-9]*.pth'))
    if len(checkpoints) > max_keep:
        for checkpoint in checkpoints[:-max_keep]:
            checkpoint.unlink()

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'iteration': checkpoint['iteration'],
        'loss': checkpoint['loss']
    }

def compute_model_size(model: nn.Module) -> Dict[str, int]:
    """Compute model size statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': (param_size + buffer_size) / (1024 * 1024)
    }

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create directory for experiment with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)