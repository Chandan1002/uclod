import os
import sys
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime

def load_config(config_file):
    """Load YAML configuration file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_distributed_training(rank, world_size, config):
    """Setup for distributed training"""
    os.environ['MASTER_ADDR'] = config['DISTRIBUTED']['MASTER_ADDR']
    os.environ['MASTER_PORT'] = config['DISTRIBUTED']['MASTER_PORT']
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    # Initialize process group
    dist.init_process_group(
        backend=config['DISTRIBUTED']['BACKEND'],
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_worker(rank, world_size, config):
    """Worker function for distributed training"""
    try:
        # Setup distributed training for this worker
        setup_distributed_training(rank, world_size, config)
        
        # Import and run training
        from main import train
        train(config)
        
    except Exception as e:
        print(f"Error in worker {rank}: {str(e)}")
        raise e
    
    finally:
        cleanup_distributed()

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Get number of GPUs
    num_gpus = config['SYSTEM']['NUM_GPUS']
    
    # Verify GPU availability
    if num_gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} are available")
    
    # Set timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['OUTPUT']['DIR'] = os.path.join(config['OUTPUT']['DIR'], timestamp)
    os.makedirs(config['OUTPUT']['DIR'], exist_ok=True)
    
    # Save the configuration
    config_save_path = os.path.join(config['OUTPUT']['DIR'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    if num_gpus > 1:
        # Use multiprocessing for multi-GPU training
        mp.spawn(
            train_worker,
            args=(num_gpus, config),
            nprocs=num_gpus,
            join=True
        )
    else:
        # Single GPU training
        train_worker(0, 1, config)

if __name__ == "__main__":
    main()