import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# PyTorch imports
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# Local imports
from visualization import DetectionVisualizer
from model import UnsupervisedObjectDetection, SUPPORTED_BACKBONES
from data import get_coco_dataloader
from optimization import HyperparamOptimizer
from utils import (
    setup_logging, 
    save_checkpoint, 
    load_checkpoint,
    MetricsTracker, 
    ResourceMonitor, 
    GradientMonitor,
    MixedPrecisionMonitor
)

# Default configurations
DEFAULT_CONFIGS = {
    'visualization': {
        'enabled': True,
        'save_heatmaps': True,
        'save_fpn_features': True,
        'save_grid_accuracy': True,
        'plot_frequency': 100,
        'max_images_per_batch': 4,
        'output_size': [800, 800],
        'colormap': "viridis"
    },
    'hpo': {  # Add this section
        'enabled': False,  # Disabled by default
        'backend': 'optuna',
        'num_trials': 100,
        'timeout': 172800,  # 48 hours
        'search_space': {
            'lr': {
                'min': 0.0001,
                'max': 0.1,
                'scale': 'log'
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'temperature': {
                'min': 0.1,
                'max': 1.0,
                'scale': 'linear'
            },
            'weight_decay': {
                'min': 0.00001,
                'max': 0.001,
                'scale': 'log'
            }
        },
        'early_stopping': {
            'patience': 5,
            'min_delta': 0.01
        }
    },
    # ... rest of DEFAULT_CONFIGS ...
}

def setup_monitoring(config: Dict, model: nn.Module, scaler: Optional[GradScaler], is_main_process: bool):
    """Setup monitoring with default values if config sections are missing"""
    if not is_main_process:
        return None, None, None, None, None
    
    # Setup logging directory
    log_dir = Path(config.get('logging', {}).get('log_dir', 'runs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize monitors
    metrics_tracker = MetricsTracker()
    resource_monitor = ResourceMonitor(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    gradient_monitor = GradientMonitor(model)
    mp_monitor = MixedPrecisionMonitor(scaler) if scaler else None
    
    # Initialize visualizer with default settings if visualization config is missing
    visualization_config = config.get('visualization', {'enabled': True})
    visualizer = DetectionVisualizer(log_dir) if visualization_config.get('enabled', True) else None
    
    return metrics_tracker, resource_monitor, gradient_monitor, mp_monitor, visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised Object Detection Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='local rank for distributed training')
    return parser.parse_args()

def setup_distributed(config, local_rank):
    if config['distributed']['enabled']:
        dist.init_process_group(
            backend=config['distributed']['dist_backend'],
            init_method=config['distributed']['init_method'],
            world_size=config['hardware']['num_gpus'],
            rank=local_rank
        )
        torch.cuda.set_device(local_rank)
    return local_rank == 0

def validate_config(config: Dict) -> None:
    """Validate configuration parameters"""
    # Validate required sections
    required_sections = ['model', 'data_dir', 'batch_size', 'num_workers']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config must contain '{section}' section")
    
    # Validate model parameters
    required_model_params = {
        'temperature': float,
        'backbone_name': str,
        'fpn_channels': int,
        'projection_dim': int
    }
    
    for param, param_type in required_model_params.items():
        if param not in config['model']:
            raise ValueError(f"Model config must contain '{param}'")
        if not isinstance(config['model'][param], param_type):
            raise ValueError(f"'{param}' must be of type {param_type}")
    
    # Validate backbone name
    if config['model']['backbone_name'] not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {config['model']['backbone_name']}. "
            f"Supported backbones: {list(SUPPORTED_BACKBONES.keys())}"
        )
    
    # Validate HPO settings if enabled
    if config.get('hpo', {}).get('enabled', False):
        if 'search_space' not in config['hpo']:
            raise ValueError("HPO config must contain 'search_space' when enabled")
        
        required_search_space = ['lr', 'batch_size', 'temperature', 'weight_decay']
        for param in required_search_space:
            if param not in config['hpo']['search_space']:
                raise ValueError(f"HPO search_space must contain '{param}'")

def update_config_with_defaults(config: Dict) -> Dict:
    """Update config with default values for missing sections"""
    updated_config = config.copy()
    
    for section, defaults in DEFAULT_CONFIGS.items():
        if section not in updated_config:
            updated_config[section] = defaults.copy()
        else:
            # Handle nested dictionaries
            for key, value in defaults.items():
                if key not in updated_config[section]:
                    updated_config[section][key] = value
                elif isinstance(value, dict) and isinstance(updated_config[section][key], dict):
                    # Recursively update nested dictionaries
                    updated_config[section][key] = {
                        **value,
                        **updated_config[section][key]
                    }
    
    return updated_config
def initialize_training(config, local_rank, is_main_process):
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Setup device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with the correct parameter names
    model = UnsupervisedObjectDetection(
        temperature=config['model']['temperature'],
        backbone_name=config['model']['backbone_name'],
        fpn_channels=config['model']['fpn_channels'],
        projection_dim=config['model']['projection_dim'],
        pretrained=config['model'].get('pretrained', True)
    ).to(device)
    
    if is_main_process:
        logging.info(f"Using backbone: {config['model']['backbone_name']}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"Using device: {device}")
    
    if config['distributed']['enabled']:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler']['T0'],
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['min_lr']
    )
    
    # Initialize mixed precision training
    scaler = None
    if config['mixed_precision']['enabled'] and torch.cuda.is_available():
        scaler = GradScaler(
            init_scale=2**config['mixed_precision']['initial_scale_power'],
            growth_factor=config['mixed_precision']['growth_factor'],
            backoff_factor=config['mixed_precision']['backoff_factor'],
            growth_interval=config['mixed_precision']['growth_interval']
        )
    
    return model, optimizer, scheduler, scaler, device
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Setup device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with the correct parameter names
    model = UnsupervisedObjectDetection(
        temperature=config['model']['temperature'],
        backbone_name=config['model']['backbone_name'],  # Changed from 'backbone' to 'backbone_name'
        fpn_channels=config['model']['fpn_channels'],
        projection_dim=config['model']['projection_dim'],
        pretrained=config['model'].get('pretrained', True)  # Added with default value
    ).to(device)
    
    if config['distributed']['enabled']:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler']['T0'],
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['min_lr']
    )
    
    # Initialize mixed precision training
    scaler = None
    if config['mixed_precision']['enabled'] and torch.cuda.is_available():
        scaler = GradScaler(
            init_scale=2**config['mixed_precision']['initial_scale_power'],
            growth_factor=config['mixed_precision']['growth_factor'],
            backoff_factor=config['mixed_precision']['backoff_factor'],
            growth_interval=config['mixed_precision']['growth_interval']
        )
    
    if is_main_process:
        logging.info(f"Initialized model with backbone: {config['model']['backbone_name']}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        logging.info(f"Device: {device}")
    
    return model, optimizer, scheduler, scaler, device

    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Setup device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UnsupervisedObjectDetection(
        temperature=config['model']['temperature'],
        backbone=config['model']['backbone'],
        fpn_channels=config['model']['fpn_channels'],
        projection_dim=config['model']['projection_dim']
    ).to(device)
    
    if config['distributed']['enabled']:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler']['T0'],
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['min_lr']
    )
    
    # Initialize mixed precision training
    scaler = None
    if config['mixed_precision']['enabled']:
        scaler = GradScaler(
            init_scale=2**config['mixed_precision']['initial_scale_power'],
            growth_factor=config['mixed_precision']['growth_factor'],
            backoff_factor=config['mixed_precision']['backoff_factor'],
            growth_interval=config['mixed_precision']['growth_interval']
        )
    
    return model, optimizer, scheduler, scaler, device

def setup_monitoring(config: Dict, model: nn.Module, scaler: Optional[GradScaler], is_main_process: bool):
    """Setup monitoring with default values if config sections are missing"""
    if not is_main_process:
        return None, None, None, None, None
    
    # Setup logging directory
    log_dir = Path(config.get('logging', {}).get('log_dir', 'runs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize monitors
    metrics_tracker = MetricsTracker()
    resource_monitor = ResourceMonitor(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    gradient_monitor = GradientMonitor(model)
    mp_monitor = MixedPrecisionMonitor(scaler) if scaler else None
    
    # Initialize visualizer with default settings if visualization config is missing
    visualization_config = config.get('visualization', {'enabled': True})
    visualizer = DetectionVisualizer(log_dir) if visualization_config.get('enabled', True) else None
    
    return metrics_tracker, resource_monitor, gradient_monitor, mp_monitor, visualizer

def train_epoch(epoch, model, train_loader, optimizer, scheduler, scaler,
                device, config, monitors, visualizer):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (images, _) in enumerate(train_loader):
        iteration = epoch * num_batches + batch_idx
        images = images.to(device)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(images)
            scaler.scale(loss).backward()
            
            if config['grad_clip']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(images)
            loss.backward()
            
            if config['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
        
        optimizer.zero_grad()
        scheduler.step(epoch + batch_idx / num_batches)
        
        # Update metrics and monitoring
        if monitors is not None:
            metrics_tracker, resource_monitor, gradient_monitor, mp_monitor, _ = monitors
            metrics = {
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
                **gradient_monitor.get_stats(),
                **resource_monitor.get_stats()
            }
            if mp_monitor:
                metrics.update(mp_monitor.get_stats())
            
            metrics_tracker.update(metrics)
            
            # Log and visualize
            if iteration % config['logging']['log_interval'] == 0:
                avg_metrics = metrics_tracker.get_averages()
                logging.info(f"Epoch: {epoch}, Iteration: {iteration}, "
                           f"Loss: {avg_metrics['loss']:.4f}, "
                           f"LR: {avg_metrics['lr']:.6f}")
                
                if visualizer and iteration % config['visualization']['plot_frequency'] == 0:
                    visualizer.visualize_batch(images, model.get_similarity_maps(),
                                            iteration)
        
        total_loss += loss.item()
    
    return total_loss / num_batches

def validate(model, val_loader, device, visualizer=None):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            images = images.to(device)
            loss = model(images)
            total_loss += loss.item()
            
            if visualizer and batch_idx == 0:
                visualizer.visualize_batch(images, model.get_similarity_maps(),
                                        f"val_{batch_idx}")
    
    return total_loss / num_batches


def validate_config(config: Dict) -> None:
    """Validate configuration parameters"""
    required_model_params = {
        'temperature': float,
        'backbone_name': str,
        'fpn_channels': int,
        'projection_dim': int
    }
    
    # Check if model section exists
    if 'model' not in config:
        raise ValueError("Config must contain 'model' section")
    
    # Validate model parameters
    for param, param_type in required_model_params.items():
        if param not in config['model']:
            raise ValueError(f"Model config must contain '{param}'")
        if not isinstance(config['model'][param], param_type):
            raise ValueError(f"'{param}' must be of type {param_type}")
    
    # Validate backbone name
    if config['model']['backbone_name'] not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {config['model']['backbone_name']}. "
                        f"Supported backbones: {list(SUPPORTED_BACKBONES.keys())}")


DEFAULT_CONFIGS = {
    'visualization': {
        'enabled': True,
        'save_heatmaps': True,
        'save_fpn_features': True,
        'save_grid_accuracy': True,
        'plot_frequency': 100,
        'max_images_per_batch': 4,
        'output_size': [800, 800],
        'colormap': "viridis"
    },
    'logging': {
        'save_dir': 'checkpoints',
        'log_dir': 'runs',
        'checkpoint_interval': 1000,
        'log_interval': 10,
        'eval_interval': 100,
        'save_optimizer': True,
        'save_best': True,
        'num_kept_checkpoints': 5
    },
    'mixed_precision': {
        'enabled': True,
        'initial_scale_power': 32,
        'growth_factor': 2,
        'backoff_factor': 0.5,
        'growth_interval': 2000
    },
    'debug': {
        'enabled': False,
        'detect_anomaly': False,
        'profile_cuda': False,
        'profile_cpu': False,
        'verbose': False
    }
}

def update_config_with_defaults(config: Dict) -> Dict:
    """Update config with default values for missing sections"""
    updated_config = config.copy()
    
    for section, defaults in DEFAULT_CONFIGS.items():
        if section not in updated_config:
            updated_config[section] = defaults
        else:
            # Update missing keys in existing sections
            for key, value in defaults.items():
                if key not in updated_config[section]:
                    updated_config[section][key] = value
    
    return updated_config

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = update_config_with_defaults(config)
    validate_config(config)

    # Setup logging
    setup_logging(config['logging']['save_dir'], config)

    # Run HPO if enabled
    if config['hpo']['enabled']:
        logging.info("Starting hyperparameter optimization...")
        optimizer = HyperparamOptimizer(config)
        best_params = optimizer.optimize()
        logging.info(f"Best parameters found: {best_params}")
        # Update config with best parameters
        config['model'].update(best_params)
        config['lr'] = best_params['lr']
        config['batch_size'] = best_params['batch_size']
        config['weight_decay'] = best_params['weight_decay']

    # Setup distributed training
    is_main_process = setup_distributed(config, args.local_rank)
    
    # Initialize training components
    model, optimizer, scheduler, scaler, device = initialize_training(
        config, args.local_rank, is_main_process
    )
    
    # Setup monitoring
    monitors = setup_monitoring(config, model, scaler, is_main_process)
    
    # Setup data loaders
    train_loader = get_coco_dataloader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        transform_config=config.get('augmentation', {}),
        distributed=config['distributed']['enabled'],
        training=True  # Specify this is training loader
    )
    
    # Create validation loader if enabled
    val_loader = None
    if config.get('validation', {}).get('enabled', False):
        val_loader = get_coco_dataloader(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            transform_config=config.get('augmentation', {}),
            distributed=False,  # No need for distributed validation
            training=False  # Specify this is validation loader
        )
    
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint_data = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint_data['epoch']
        if is_main_process:
            logging.info(f"Resumed from epoch {start_epoch}")
    
    # Hyperparameter optimization if enabled
    if config['hpo']['enabled'] and is_main_process:
        optimizer = HyperparamOptimizer(config)
        best_params = optimizer.optimize()
        logging.info(f"Best hyperparameters: {best_params}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config['epochs']):
        if config['distributed']['enabled']:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, scheduler, scaler,
            device, config, monitors, monitors[-1] if monitors else None
        )
        
        # Validation
        if is_main_process and epoch % config['logging']['eval_interval'] == 0:
            val_loss = validate(
                model, val_loader, device,
                monitors[-1] if monitors else None
            )
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoint
            if epoch % config['logging']['checkpoint_interval'] == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    config['logging']['save_dir'],
                    is_best=is_best
                )
    
    # Cleanup
    if config['distributed']['enabled']:
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        logging.info("Please check your config.yaml file and ensure all required sections are present.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.debug("Stack trace:", exc_info=True)
        sys.exit(1)