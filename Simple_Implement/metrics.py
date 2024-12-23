import time
import logging
import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime
import csv
import os

logger = logging.getLogger(__name__)

def calculate_grid_metrics(similarity_map, gt_box, random_init=None):
    """Calculate grid alignment metrics (SGA, RIGA, GAP-R)
    
    Args:
        similarity_map: Tensor containing similarity scores
        gt_box: Ground truth bounding box coordinates [x, y, w, h]
        random_init: Optional pre-computed random initialization map
        
    Returns:
        dict: Dictionary containing SGA, RIGA and GAP-R metrics
    """
    # Calculate SGA
    feature_map_size = int(similarity_map.shape[1] ** 0.5)
    scale_factor = 800.0 / feature_map_size  # Assuming 800x800 input image
    
    # Get highest similarity location
    max_idx = torch.argmax(similarity_map.flatten())
    pred_y = (max_idx // feature_map_size) * scale_factor
    pred_x = (max_idx % feature_map_size) * scale_factor
    
    # Check if prediction falls within ground truth box
    is_inside = (
        pred_x >= gt_box[0] and
        pred_y >= gt_box[1] and
        pred_x <= gt_box[0] + gt_box[2] and
        pred_y <= gt_box[1] + gt_box[3]
    )
    
    sga = float(is_inside)
    
    # Calculate RIGA using random initialization if provided
    if random_init is None:
        random_init = torch.rand_like(similarity_map)
    rand_max_idx = torch.argmax(random_init.flatten())
    rand_y = (rand_max_idx // feature_map_size) * scale_factor
    rand_x = (rand_max_idx % feature_map_size) * scale_factor
    
    rand_is_inside = (
        rand_x >= gt_box[0] and
        rand_y >= gt_box[1] and
        rand_x <= gt_box[0] + gt_box[2] and
        rand_y <= gt_box[1] + gt_box[3]
    )
    
    riga = float(rand_is_inside)
    
    # Calculate GAP-R
    gap_r = sga / riga if riga > 0 else float('inf')
    
    return {
        'SGA': sga,
        'RIGA': riga,
        'GAP-R': gap_r
    }

def save_metrics(output_dir, metrics, prefix=''):
    """Save metrics to a JSON file
    
    Args:
        output_dir: Directory to save the metrics
        metrics: Dictionary containing metrics
        prefix: Optional prefix for the output filename
    """
    timestamp = datetime.datetime.now().isoformat()
    filename = f"{prefix}_metrics.json" if prefix else "metrics.json"
    metrics_path = os.path.join(output_dir, filename)
    
    output_metrics = {
        'accuracy': metrics.get('SGA', None),
        'riga': metrics.get('RIGA', None),
        'gap_r': metrics.get('GAP-R', None),
        'loss': metrics.get('final_loss', None),
        'timestamp': timestamp
    }
    
    # Add any additional metrics that might be present
    output_metrics.update({
        k: v for k, v in metrics.items() 
        if k not in ['SGA', 'RIGA', 'GAP-R', 'final_loss']
    })
    
    with open(metrics_path, 'w') as f:
        json.dump(output_metrics, f, indent=2)

def calculate_iou(pred_box, gt_box):
    """Calculate IoU between prediction and ground truth boxes"""
    # Calculate intersection coordinates
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    y2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = pred_box[2] * pred_box[3]
    gt_area = gt_box[2] * gt_box[3]
    union = pred_area + gt_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_metrics(similarity_map, crop_coords, k=5):
    """Calculate AP and AR at different IoU thresholds"""
    gt_box = crop_coords[0]  # [x, y, w, h]
    
    # Get top k predictions
    sim_map = similarity_map.flatten()
    values, indices = torch.topk(sim_map, k)
    
    # Convert predictions to boxes
    feature_map_size = int(similarity_map.shape[1] ** 0.5)
    scale_factor = 800.0 / feature_map_size  # Assuming 800x800 input image
    predictions = []
    
    for idx, conf in zip(indices, values):
        pred_y = (idx // feature_map_size) * scale_factor
        pred_x = (idx % feature_map_size) * scale_factor
        
        pred_box = [
            pred_x - gt_box[2]/2,  # x
            pred_y - gt_box[3]/2,  # y
            gt_box[2],             # width
            gt_box[3]              # height
        ]
        predictions.append({
            'box': pred_box,
            'confidence': conf.item()
        })
    
    # Calculate metrics at different IoU thresholds
    metrics = {}
    thresholds = [0.5, 0.75]
    
    for iou_threshold in thresholds:
        tp = 0
        precisions = []
        recalls = []
        
        for i, pred in enumerate(predictions, 1):
            iou = calculate_iou(pred['box'], gt_box)
            if iou >= iou_threshold:
                tp += 1
            
            precision = tp / i
            recall = tp / 1  # Only one ground truth
            precisions.append(precision)
            recalls.append(recall)
            
            # Debug print if needed
            # print(f"Pred {i} @ IoU {iou_threshold}: conf={pred['confidence']:.4f}, IoU={iou:.4f}")
        
        # Calculate AP and AR for this threshold
        ap = max(precisions) if precisions else 0
        ar = max(recalls) if recalls else 0
        
        metrics[f'AP{int(iou_threshold*100)}'] = ap
        metrics[f'AR{int(iou_threshold*100)}'] = ar
    
    # Calculate mAP
    metrics['mAP'] = sum(metrics[f'AP{int(t*100)}'] for t in thresholds) / len(thresholds)

    # Add grid metrics
    grid_metrics = calculate_grid_metrics(similarity_map, gt_box)
    metrics.update(grid_metrics) 
    
    return metrics


class CSVLoggerIter:
    def __init__(self, log_dir, rank=0):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = Path(log_dir) / 'logs'
        self.log_file = self.log_dir / f'iteration_metrics_{current_time}.csv'
        self.rank = rank
        self.fieldnames = ['iteration', 'timestamp', 'loss', 'map', 'lr', 'AP50', 'AP75', 'AR50', 'AR75', 'mAP', 'SGA', 'RIGA', 'GAP-R']

        if rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def update(self, metrics_dict, iteration):
        if self.rank != 0:
            return
        
        row_dict = {
            'iteration': iteration,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'loss': metrics_dict.get('loss', ''),
            'map': metrics_dict.get('map', ''),
            'lr': metrics_dict.get('lr', ''),
            'AP50': metrics_dict.get('AP50', ''),
            'AP75': metrics_dict.get('AP75', ''),
            'AR50': metrics_dict.get('AR50', ''),
            'AR75': metrics_dict.get('AR75', ''),
            'mAP': metrics_dict.get('mAP', '')
        }
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_dict)


class CSVLoggerEpoch:
    def __init__(self, log_dir, rank=0):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = Path(log_dir) / 'logs'
        self.log_file = self.log_dir / f'epoch_metrics_{current_time}.csv'
        self.rank = rank
        self.fieldnames = ['epoch', 'timestamp', 'loss', 'map', 'lr', 'AP50', 'AP75', 'AR50', 'AR75', 'mAP', 'SGA', 'RIGA', 'GAP-R']

        if rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def update_epoch(self, metrics_dict, epoch):
        if self.rank != 0:
            return
        
        row_dict = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'loss': metrics_dict.get('loss', ''),
            'map': metrics_dict.get('map', ''),
            'lr': metrics_dict.get('lr', ''),
            'AP50': metrics_dict.get('AP50', ''),
            'AP75': metrics_dict.get('AP75', ''),
            'AR50': metrics_dict.get('AR50', ''),
            'AR75': metrics_dict.get('AR75', ''),
            'mAP': metrics_dict.get('mAP', '')
        }
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_dict)


class TensorboardLogger:
    def __init__(self, log_dir, rank=0):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_dir = Path(log_dir) / 'tensorboard' / current_time
        self.rank = rank
        
        if rank == 0:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        else:
            self.writer = None

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.writer is not None and not torch.is_tensor(scalar_value):
            self.writer.add_scalar(tag, scalar_value, global_step)
    
    def close(self):
        if self.writer is not None:
            self.writer.close()


class MetricsLogger:
    def __init__(self, log_dir, distributed=False):
        self.log_dir = log_dir
        self.distributed = distributed
        self.running_metrics = defaultdict(float)
        self.n_steps = 0
        
        rank = dist.get_rank() if distributed else 0
        self.tb_logger = TensorboardLogger(log_dir, rank)
        
        # Separate CSV loggers for iteration and epoch
        self.csv_logger_iter = CSVLoggerIter(log_dir, rank)
        self.csv_logger_epoch = CSVLoggerEpoch(log_dir, rank)
    
    def update(self, metrics_dict, iteration, log_to_console=True):
        # Average metrics if distributed
        if self.distributed:
            gathered_metrics = {}
            for key, value in metrics_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.clone().detach()
                    dist.all_reduce(value)
                    value = value / dist.get_world_size()
                gathered_metrics[key] = value
            metrics_dict = gathered_metrics
        
        # Update running metrics
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.running_metrics[key] += value
        self.n_steps += 1
        
        # Log iteration-level metrics to CSV
        self.csv_logger_iter.update(metrics_dict, iteration)
        
        # Log to TensorBoard
        for key, value in metrics_dict.items():
            self.tb_logger.add_scalar(f'train/{key}', value, iteration)
        
        if log_to_console and (not self.distributed or (self.distributed and dist.get_rank() == 0)):
            log_str = f"Iteration {iteration} |"
            for key, value in metrics_dict.items():
                if isinstance(value, float):
                    log_str += f" {key}: {value:.4f} |"
                else:
                    log_str += f" {key}: {value} |"
            logger.info(log_str)
    
    def update_epoch(self, epoch_metrics_dict, epoch, log_to_console=True):
        # Average metrics if distributed
        if self.distributed:
            gathered_metrics = {}
            for key, value in epoch_metrics_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.clone().detach()
                    dist.all_reduce(value)
                    value = value / dist.get_world_size()
                gathered_metrics[key] = value
            epoch_metrics_dict = gathered_metrics

        # Log epoch-level metrics to separate CSV
        self.csv_logger_epoch.update_epoch(epoch_metrics_dict, epoch)
        
        if log_to_console and (not self.distributed or (self.distributed and dist.get_rank() == 0)):
            log_str = f"Epoch {epoch} |"
            for key, value in epoch_metrics_dict.items():
                if isinstance(value, float):
                    log_str += f" {key}: {value:.4f} |"
                else:
                    log_str += f" {key}: {value} |"
            logger.info(log_str)

    def update_grid_metrics(self, similarity_map, gt_box, iteration):
        """Update grid alignment metrics
        
        Args:
            similarity_map: Tensor containing similarity scores
            gt_box: Ground truth bounding box coordinates
            iteration: Current iteration number
        """
        grid_metrics = calculate_grid_metrics(similarity_map, gt_box)
        
        # Log to CSV and tensorboard
        self.update(grid_metrics, iteration)
        
        # Save detailed metrics at regular intervals or specific iterations
        if iteration % self.cfg.OUTPUT.SAVE_PERIOD == 0:
            save_metrics(self.log_dir, grid_metrics, prefix=f'iter_{iteration}')
            
    def log_gpu_stats(self, iteration):
        if torch.cuda.is_available() and (not self.distributed or (self.distributed and dist.get_rank() == 0)):
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
                self.tb_logger.add_scalar(f'system/gpu{i}_allocated_memory', allocated, iteration)
                self.tb_logger.add_scalar(f'system/gpu{i}_reserved_memory', reserved, iteration)
    
    def close(self):
        self.tb_logger.close()



class PerformanceMetrics:
    def __init__(self, model, metrics_logger):
        self.model = model
        self.metrics_logger = metrics_logger
        self.reset_metrics()
    
    def reset_metrics(self):
        self.total_loss = 0
        self.n_steps = 0
    
    def update(self, metrics_dict, iteration, lr=None, log_to_console=True):
        if lr is not None:
            metrics_dict['lr'] = lr
        self.metrics_logger.update(metrics_dict, iteration, log_to_console)
        
        if 'loss' in metrics_dict:
            self.total_loss += metrics_dict['loss']
        self.n_steps += 1
    
    def update_epoch(self, epoch_metrics_dict, epoch, log_to_console=True):
        self.metrics_logger.update_epoch(epoch_metrics_dict, epoch, log_to_console)
    
    def log_gpu_stats(self, iteration):
        self.metrics_logger.log_gpu_stats(iteration)
