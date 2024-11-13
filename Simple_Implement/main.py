import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from dataloader import build_unsupervised_train_loader
from model import UnsupervisedDetector

def setup_logger(output_dir, rank):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('train')

def save_checkpoint(model, optimizer, iteration, output_dir, filename):
    """Save model checkpoint"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return 0
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']

def train(cfg):
    """Main training function"""
    # Get rank for distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Setup logger
    logger = setup_logger(cfg['OUTPUT']['DIR'], rank)
    logger.info(f"Starting training on rank {rank} of {world_size}")
    
    # Set device
    if cfg['SYSTEM']['DEVICE'] == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cpu')
    
    # Build model
    model = UnsupervisedDetector(cfg)
    model.to(device)
    
    # Wrap model in DDP if using multiple GPUs
    if cfg['DISTRIBUTED']['ENABLED']:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True  # Important for partial model update
        )
    
    # Build optimizer with correct base learning rate
    base_lr = cfg['SOLVER']['BASE_LR'] * cfg['SYSTEM']['NUM_GPUS']
    optimizer = optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=cfg['SOLVER']['WEIGHT_DECAY']
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if cfg['MODEL']['WEIGHTS']:
        start_iter = load_checkpoint(model, optimizer, cfg['MODEL']['WEIGHTS'])
        logger.info(f"Resumed from checkpoint {cfg['MODEL']['WEIGHTS']}")
    
    # Build data loader
    data_loader = build_unsupervised_train_loader(cfg)
    logger.info("Data loader built successfully")
    
    # Training loop
    model.train()
    max_iter = cfg['SOLVER']['MAX_ITER']
    log_period = cfg['OUTPUT']['LOG_PERIOD']
    save_period = cfg['OUTPUT']['SAVE_PERIOD']
    
    logger.info(f"Starting training from iteration {start_iter}")
    
    for iteration, batch in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        
        # Log learning rate periodically
        if iteration % log_period == 0 and rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Iteration {iteration}/{max_iter}, lr: {lr:.6f}")
        
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass and compute loss
        loss_dict = model(batch)
        losses = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Log metrics
        if iteration % log_period == 0 and rank == 0:
            logger.info(f"Iteration {iteration}, total loss: {losses.item():.4f}")
        
        # Save checkpoint
        if iteration % save_period == 0 and rank == 0:
            save_checkpoint(
                model.module if isinstance(model, DistributedDataParallel) else model,
                optimizer,
                iteration,
                cfg['OUTPUT']['DIR'],
                f"model_{iteration:07d}.pth"
            )
        
        if iteration >= max_iter:
            break
    
    # Save final model
    if rank == 0:
        save_checkpoint(
            model.module if isinstance(model, DistributedDataParallel) else model,
            optimizer,
            max_iter,
            cfg['OUTPUT']['DIR'],
            "model_final.pth"
        )
        logger.info("Training completed successfully")