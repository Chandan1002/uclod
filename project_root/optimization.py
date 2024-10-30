import torch
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import optuna

class HyperparamOptimizer:
    """Handles hyperparameter optimization primarily using Optuna"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.best_params = None
        self.best_score = float('inf')
        self.study = None
        
        # Create results directory
        self.results_dir = Path(config.get('optimization', {}).get('results_dir', 'hpo_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def optimize(self, model_fn, train_fn) -> Dict[str, Any]:
        """Run optimization using Optuna"""
        study = optuna.create_study(
            direction="minimize",
            study_name="unsupervised_detection_optimization"
        )
        
        def objective(trial: optuna.Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                'lr': trial.suggest_loguniform('lr', 
                    self.config['optimization']['search_space']['lr']['min'],
                    self.config['optimization']['search_space']['lr']['max']),
                'batch_size': trial.suggest_categorical('batch_size',
                    self.config['optimization']['search_space']['batch_size']['values']),
                'temperature': trial.suggest_uniform('temperature',
                    self.config['optimization']['search_space']['temperature']['min'],
                    self.config['optimization']['search_space']['temperature']['max']),
                'weight_decay': trial.suggest_loguniform('weight_decay',
                    self.config['optimization']['search_space']['weight_decay']['min'],
                    self.config['optimization']['search_space']['weight_decay']['max']),
                'fpn_channels': trial.suggest_categorical('fpn_channels',
                    self.config['optimization']['search_space']['fpn_channels']['values']),
                'projection_dim': trial.suggest_categorical('projection_dim',
                    self.config['optimization']['search_space']['projection_dim']['values'])
            }
            
            # Create model with trial parameters
            model = model_fn(**params)
            
            # Train model and get validation loss
            val_loss = train_fn(model, params)
            
            return val_loss
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config['optimization']['num_trials'],
            timeout=self.config['optimization'].get('timeout', None)
        )
        
        self.study = study
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save results
        self.save_results()
        
        return self.best_params

    def save_results(self):
        """Save optimization results"""
        if not self.study:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),  # Convert numpy.float to Python float
            'n_trials': len(self.study.trials),
            'study_name': self.study.study_name,
        }
        
        # Save to JSON
        results_file = self.results_dir / f'optimization_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save visualization if matplotlib is available
        try:
            self.plot_results(timestamp)
        except Exception as e:
            logging.warning(f"Could not create optimization plots: {e}")
    
    def plot_results(self, timestamp: str):
        """Create and save optimization visualizations"""
        if not self.study:
            return
            
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'optimization_history_{timestamp}.png')
        plt.close()
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'param_importances_{timestamp}.png')
        plt.close()
        
        # Plot parallel coordinate
        plt.figure(figsize=(15, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'parallel_coordinate_{timestamp}.png')
        plt.close()

def create_optimizer(config: Dict[str, Any]) -> HyperparamOptimizer:
    """Create hyperparameter optimizer"""
    return HyperparamOptimizer(config)