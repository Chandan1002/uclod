import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from dataloader import build_unsupervised_train_loader
from model import UnsupervisedDetector
from metrics import MetricsLogger, PerformanceMetrics
from metrics import MetricsLogger, calculate_metrics
from tqdm import tqdm

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

def log_gpu_stats(rank):
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        utilization = torch.cuda.utilization(i)

        print(f"GPU Rank {rank}:")
        print(f"  Memory Allocated: {allocated:.2f}MB")
        print(f"  Max Memory Allocated: {max_allocated:.2f}MB")
        print(f"  Memory Reserved: {reserved:.2f}MB")
        print(f"  Utilization: {utilization}%")

def save_checkpoint(model, optimizer, epoch, iteration, output_dir, filename):
    """Save model checkpoint"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return {'epoch': 0, 'iteration': 0}

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model']

    if isinstance(model, DistributedDataParallel):
        if not any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {'module.' + key: value for key, value in state_dict.items()}
    elif any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key in state_dict.items()}

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {str(e)}")
        print("Keys in checkpoint:", state_dict.keys())
        print("Keys in model:", model.state_dict().keys())
        raise e

    optimizer.load_state_dict(checkpoint['optimizer'])
    return {
        'epoch': checkpoint.get('epoch', 0),
        'iteration': checkpoint.get('iteration', 0)
    }

def calculate_max_iter(cfg, data_loader, logger):
    """Calculate MAX_ITER and update STEPS based on EPOCHS"""
    epochs = cfg['SOLVER'].get('EPOCHS', 20)

    total_images = len(data_loader.dataset)
    batch_size = cfg['SOLVER']['IMS_PER_BATCH'] * cfg['SYSTEM']['NUM_GPUS']
    iters_per_epoch = total_images // batch_size

    cfg['SOLVER']['MAX_ITER'] = iters_per_epoch * epochs
    cfg['SOLVER']['STEPS'] = (
        int(0.7 * cfg['SOLVER']['MAX_ITER']),
        int(0.9 * cfg['SOLVER']['MAX_ITER'])
    )

    if logger:
        logger.info(f"Training for {epochs} epochs ({cfg['SOLVER']['MAX_ITER']} iterations)")
        logger.info(f"Learning rate steps at iterations: {cfg['SOLVER']['STEPS']}")

    return cfg

def evaluate(model, data_loader, cfg):
    """
    Evaluate the model on the dataset
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        cfg: Configuration dictionary
    """
    model.eval()
    logger = logging.getLogger('train')
    logger.info("Starting evaluation...")

    save_viz = cfg.get('EVAL', {}).get('SAVE_VISUALIZATIONS', False)
    if save_viz:
        cfg['OUTPUT']['SAVE_VIZ'] = True

    try:
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader), 
                                   total=len(data_loader), 
                                   desc="Evaluating"):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                outputs = model(batch)
                metrics = calculate_metrics(outputs["similarity_map"], 
                                             batch["crop_coords"])
                if idx % 10 == 0:
                    logger.info(f"Batch {idx} metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise e

    logger.info("Evaluation completed")

def train(cfg):
    """Main training function"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    logger = setup_logger(cfg['OUTPUT']['DIR'], rank)
    logger.info(f"Starting training on rank {rank} of {world_size}")

    if cfg['SYSTEM']['DEVICE'] == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cpu')

    model = UnsupervisedDetector(cfg)
    model.to(device)

    metrics_logger = MetricsLogger(cfg['OUTPUT']['DIR'], cfg['DISTRIBUTED']['ENABLED'])
    performance_metrics = PerformanceMetrics(model, metrics_logger)

    if cfg['DISTRIBUTED']['ENABLED']:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    base_lr = cfg['SOLVER']['BASE_LR'] * cfg['SYSTEM']['NUM_GPUS']
    optimizer = optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=cfg['SOLVER']['WEIGHT_DECAY']
    )

    start_epoch = 0
    start_iter = 0
    if cfg['MODEL']['WEIGHTS']:
        checkpoint_data = load_checkpoint(model, optimizer, cfg['MODEL']['WEIGHTS'])
        start_epoch = checkpoint_data['epoch']
        start_iter = checkpoint_data['iteration']
        logger.info(f"Resumed from epoch {start_epoch}, iteration {start_iter}")

    data_loader = build_unsupervised_train_loader(cfg)
    logger.info("Data loader built successfully")

    cfg = calculate_max_iter(cfg, data_loader, logger)
    logger.info(f"Training for {cfg['SOLVER']['EPOCHS']} epochs "
                f"({cfg['SOLVER']['MAX_ITER']} iterations)")

    model.train()
    num_epochs = cfg['SOLVER']['EPOCHS']
    log_period = cfg['OUTPUT']['LOG_PERIOD']
    save_period = cfg['OUTPUT']['SAVE_PERIOD']

    logger.info(f"Starting training from iteration {start_iter}")

    try:
        for epoch in range(start_epoch, num_epochs):
            if cfg['DISTRIBUTED']['ENABLED']:
                data_loader.sampler.set_epoch(epoch)

            pbar = tqdm(total=len(data_loader), initial=0, 
                        desc=f'Epoch {epoch}/{num_epochs}', 
                        disable=(rank != 0))

            num_iterations = 0
            optimizer.zero_grad()
            accumulated_loss = 0

            epoch_metrics = {
                'loss': 0.0,
                'mAP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP': 0.0,
                'AR50': 0.0,
                'AR75': 0.0,
                'AR': 0.0,
            }
            num_batches = 0

            for iteration, batch in enumerate(data_loader):
                num_iterations += 1
                global_iter = epoch * len(data_loader) + iteration

                batch = {k: v.to(device) for k, v in batch.items()}

                output_dict = model(batch)
                loss = output_dict["loss"]
                loss = loss / cfg['SOLVER']['GRADIENT_ACCUMULATION_STEPS']

                if cfg['DISTRIBUTED']['ENABLED']:
                    dist.all_reduce(loss)
                    loss = loss / world_size

                loss.backward()
                accumulated_loss += loss.item() * cfg['SOLVER']['GRADIENT_ACCUMULATION_STEPS']

                if (iteration + 1) % cfg['SOLVER']['GRADIENT_ACCUMULATION_STEPS'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                metric_values = calculate_metrics(output_dict["similarity_map"], batch["crop_coords"])

                epoch_metrics['loss'] += loss.item()
                for k, v in metric_values.items():
                    epoch_metrics[k] += v
                num_batches += 1

                if iteration % log_period == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    ap = (metric_values['AP50'] + metric_values['AP75']) / 2
                    ar = (metric_values['AR50'] + metric_values['AR75']) / 2
                    metric_values.update({'AP': ap, 'AR': ar})
                    if rank == 0:
                        logger.info(f"Iteration {iteration} - Loss: {accumulated_loss:.8f}")
                        logger.info(f"Metrics - mAP: {metric_values['mAP']:.8f}, "
                                    f"AP50: {metric_values['AP50']:.8f}, "
                                    f"AP75: {metric_values['AP75']:.8f}, "
                                    f"AR50: {metric_values['AR50']:.8f}, "
                                    f"AR75: {metric_values['AR75']:.8f}")
                        print(f"Learning Rate: {lr:.6f}")
                        print("="*50)

                    metrics = {
                        'loss': loss.item(),
                        'lr': lr,
                        'epoch': epoch,
                        **metric_values
                    }
                    performance_metrics.update(metrics, global_iter, lr, log_to_console=(rank == 0))
                    performance_metrics.log_gpu_stats(global_iter)

                desc = f'Epoch {epoch}/{num_epochs} | loss: {accumulated_loss:.4f} | mAP: {metric_values["mAP"]:.4f}'
                pbar.set_description(desc)

                if iteration % save_period == 0 and rank == 0:
                    save_checkpoint(
                        model.module if isinstance(model, DistributedDataParallel) else model,
                        optimizer, epoch, iteration, cfg['OUTPUT']['DIR'],
                        f"model_epoch_{epoch:03d}_iter_{iteration:07d}.pth"
                    )

                pbar.update(1)
                accumulated_loss = 0

            if rank == 0:
                epoch_avg_metrics = {f'epoch_{k}': v / num_batches for k, v in epoch_metrics.items()}
                epoch_avg_metrics['epoch'] = epoch
                epoch_avg_metrics['lr'] = optimizer.param_groups[0]["lr"]

                performance_metrics.update(
                    epoch_avg_metrics, 
                    epoch,
                    epoch_avg_metrics['lr'],
                    log_to_console=True
                )

                logger.info(f"\nEpoch {epoch} Summary:")
                for k, v in epoch_avg_metrics.items():
                    logger.info(f"  {k}: {v:.4f}")

                save_checkpoint(
                    model.module if isinstance(model, DistributedDataParallel) else model,
                    optimizer, epoch + 1, 0, cfg['OUTPUT']['DIR'],
                    f"model_epoch_{epoch:03d}_final.pth"
                )
                logger.info(f"Completed epoch {epoch}")

            pbar.close()

        if rank == 0:
            save_checkpoint(
                model.module if isinstance(model, DistributedDataParallel) else model,
                optimizer, num_epochs, 0, cfg['OUTPUT']['DIR'],
                "model_final.pth"
            )
            logger.info("Training completed successfully")

        if rank == 0 and cfg.get('EVAL', {}).get('ENABLED', False):
            logger.info("Starting evaluation...")
            evaluate(model, data_loader, cfg)
        else:
            logger.info("Skipping evaluation (not enabled in config)")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e

    finally:
        metrics_logger.close()
        if cfg['DISTRIBUTED']['ENABLED']:
            dist.destroy_process_group()
