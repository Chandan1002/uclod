import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from dataloader import build_unsupervised_train_loader, save_image
from metrics import MetricsLogger, PerformanceMetrics, calculate_metrics
from model import UnsupervisedDetector


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# from train import save_metrics

# def save_metrics(output_dir, metrics):
#     """Save training metrics to a JSON file"""
#     metrics_path = os.path.join(output_dir, 'metrics.json')
#     with open(metrics_path, 'w') as f:
#         json.dump({
#             'accuracy': metrics['SGA'],  # Similarity Grid Accuracy
#             'riga': metrics['RIGA'],     # Random Initialization Grid Accuracy
#             'gap_r': metrics['GAP-R'],   # Grid Alignment Performance Ratio
#             'loss': metrics.get('final_loss', None),
#             'timestamp': datetime.now().isoformat()
#         }, f, indent=2)


def setup_logger(output_dir, rank):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("train")


def log_gpu_stats(rank):
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        utilization = torch.cuda.utilization(i)

        # print(f"GPU {i}:")
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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iteration": iteration,
    }
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return {"epoch": 0, "iteration": 0}

    checkpoint = torch.load(path, map_location="cpu")
    # Handle DDP state dict loading
    state_dict = checkpoint["model"]
    # If current model is DDP but checkpoint is not
    if isinstance(model, DistributedDataParallel):
        if not any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {"module." + key: value for key, value in state_dict.items()}
    # If current model is not DDP but checkpoint is
    elif any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key in state_dict.items()}

    # Load state dict with error handling
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {str(e)}")
        print("Keys in checkpoint:", state_dict.keys())
        print("Keys in model:", model.state_dict().keys())
        raise e

    # model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # return checkpoint['iteration']
    return {
        "epoch": checkpoint.get("epoch", 0),
        "iteration": checkpoint.get("iteration", 0),
    }


def calculate_max_iter(cfg, data_loader, logger):
    """Calculate MAX_ITER and update STEPS based on EPOCHS"""
    # Default to 200 epochs if not specified
    epochs = cfg["SOLVER"].get("EPOCHS", 20)

    total_images = len(data_loader.dataset)
    batch_size = cfg["SOLVER"]["IMS_PER_BATCH"] * cfg["SYSTEM"]["NUM_GPUS"]
    iters_per_epoch = total_images // batch_size

    # Calculate MAX_ITER from EPOCHS
    cfg["SOLVER"]["MAX_ITER"] = iters_per_epoch * epochs

    # Update STEPS to be at 70% and 90% of training
    cfg["SOLVER"]["STEPS"] = (
        int(0.7 * cfg["SOLVER"]["MAX_ITER"]),
        int(0.9 * cfg["SOLVER"]["MAX_ITER"]),
    )

    if logger:
        logger.info(
            f"Training for {epochs} epochs ({cfg['SOLVER']['MAX_ITER']} iterations)"
        )
        logger.info(f"Learning rate steps at iterations: {cfg['SOLVER']['STEPS']}")

    return cfg


def evaluate(model, data_loader, cfg):
    """Evaluate model with enhanced visualizations"""
    model.eval()
    logger = setup_logger(cfg["OUTPUT"]["DIR"], rank=0)
    metrics_logger = MetricsLogger(cfg["OUTPUT"]["DIR"], cfg["DISTRIBUTED"]["ENABLED"])

    # Add total metrics accumulator
    total_metrics = {}
    num_batches = 0

    if cfg["EVAL"]["SAVE_VISUALIZATIONS"]:
        from visualization import EvalVisualization

        visualizer = EvalVisualization(cfg)
        metrics_history = {}

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                batch = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = model(batch)
                crop_coords = batch["crop_coords"]
                metrics = calculate_metrics(
                    outputs["similarity_map"], batch["crop_coords"]
                )

                # Accumulate metrics
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
                num_batches += 1

                if cfg["EVAL"]["SAVE_VISUALIZATIONS"]:
                    for index in range(len(batch["image_original"])):
                        visualizer.save_detection_visualization(
                            image=batch["image_original"][index],
                            similarity_map=outputs["similarity_map"][index],
                            pred_boxes=outputs.get("boxes"),
                            gt_boxes=batch["crop_coords"][index],  # Changed this line
                            confidences=outputs.get("confidences", []),
                            metrics=metrics,
                            batch_idx=batch_idx,
                            sample_idx=index,
                        )
                        visualizer.save_image(
                            batch["image_original"][index]
                            .cpu()
                            .permute(1, 2, 0)
                            .numpy()
                            .copy(),
                            anchors=[crop_coords[index]],
                            batch_idx=batch_idx,
                            name="full",
                        )
                        visualizer.save_image(
                            batch["image_crop"][index].cpu().permute(1, 2, 0).numpy().copy(),
                            name="crop",
                            batch_idx=batch_idx,
                            anchors=None,
                            index=index
                        )
                        visualizer.generate_heatmap(
                            index=0,
                            x_j=batch["image_original"],
                            z_i=outputs["resnet_features"],
                            features=outputs["fpn_features"],
                            fpn_layer=0,
                            anchor=[batch["crop_coords"][0]],
                            name="full",
                            batch_idx=batch_idx
                        )

                    for k, v in metrics.items():
                        if k not in metrics_history:
                            metrics_history[k] = []
                        metrics_history[k].append(v)

                    if batch_idx % 10 == 0:
                        visualizer.save_metrics_history(metrics_history)

                if batch_idx % cfg["OUTPUT"]["LOG_PERIOD"] == 0:
                    logger.info(f"Batch {batch_idx} metrics: {metrics}")

        # Calculate and save final average metrics
        final_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        # metrics_logger.save_metrics(final_metrics)
        metrics_logger.update(final_metrics, 0)
        # metrics_logger.add_scalars('eval', final_metrics, 0)
        logger.info(f"Final Evaluation Metrics: {final_metrics}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise e
    finally:
        metrics_logger.close()

    return final_metrics


def train(cfg):
    """Main training function"""
    # Get rank for distributed training
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Setup logger
    logger = setup_logger(cfg["OUTPUT"]["DIR"], rank)
    logger.info(f"Starting training on rank {rank} of {world_size}")

    # Set device
    if cfg["SYSTEM"]["DEVICE"] == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Build model
    model = UnsupervisedDetector(cfg)
    model.to(device)

    # Tensorboard logging
    metrics_logger = MetricsLogger(cfg["OUTPUT"]["DIR"], cfg["DISTRIBUTED"]["ENABLED"])
    performance_metrics = PerformanceMetrics(model, metrics_logger)

    # Wrap model in DDP if using multiple GPUs
    if cfg["DISTRIBUTED"]["ENABLED"]:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True  # Important for partial model update
        )

    # Build optimizer with correct base learning rate
    base_lr = cfg["SOLVER"]["BASE_LR"] * cfg["SYSTEM"]["NUM_GPUS"]
    optimizer = optim.Adam(
        model.parameters(), lr=base_lr, weight_decay=cfg["SOLVER"]["WEIGHT_DECAY"]
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    start_iter = 0
    if cfg["MODEL"]["WEIGHTS"]:
        # start_iter = load_checkpoint(model, optimizer, cfg['MODEL']['WEIGHTS'])
        # logger.info(f"Resumed from checkpoint {cfg['MODEL']['WEIGHTS']}")
        checkpoint_data = load_checkpoint(model, optimizer, cfg["MODEL"]["WEIGHTS"])
        start_epoch = checkpoint_data["epoch"]
        start_iter = checkpoint_data["iteration"]
        logger.info(f"Resumed from epoch {start_epoch}, iteration {start_iter}")

    # Build data loader
    data_loader = build_unsupervised_train_loader(cfg)
    logger.info("Data loader built successfully")

    # Calculate and update MAX_ITER based on EPOCHS
    cfg = calculate_max_iter(cfg, data_loader, logger)
    logger.info(
        f"Training for {cfg['SOLVER']['EPOCHS']} epochs "
        f"({cfg['SOLVER']['MAX_ITER']} iterations)"
    )

    # Training loop
    model.train()
    # max_iter = cfg['SOLVER']['MAX_ITER']
    num_epochs = cfg["SOLVER"]["EPOCHS"]
    log_period = cfg["OUTPUT"]["LOG_PERIOD"]
    save_period = cfg["OUTPUT"]["SAVE_PERIOD"]

    logger.info(f"Starting training from iteration {start_iter}")

    try:
        for epoch in range(start_epoch, num_epochs):
            if cfg["DISTRIBUTED"]["ENABLED"]:
                data_loader.sampler.set_epoch(epoch)

            pbar = tqdm(
                total=len(data_loader),
                initial=0,
                desc=f"Epoch {epoch}/{num_epochs}",
                disable=(rank != 0),
            )

            num_iterations = 0
            optimizer.zero_grad()
            accumulated_loss = 0

            epoch_metrics = {
                "loss": 0.0,
                "mAP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "AP": 0.0,
                "AR50": 0.0,
                "AR75": 0.0,
                "AR": 0.0,
                "SGA": 0.0,  # Added
                "RIGA": 0.0,  # Added
                "GAP-R": 0.0,  # Added
            }
            num_batches = 0

            for iteration, batch in enumerate(data_loader):
                num_iterations += 1
                global_iter = epoch * len(data_loader) + iteration
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass and compute loss
                output_dict = model(batch)
                loss = output_dict["loss"]
                loss = loss / cfg["SOLVER"]["GRADIENT_ACCUMULATION_STEPS"]

                if cfg["DISTRIBUTED"]["ENABLED"]:
                    dist.all_reduce(loss)
                    loss = loss / world_size

                loss.backward()
                accumulated_loss += (
                    loss.item() * cfg["SOLVER"]["GRADIENT_ACCUMULATION_STEPS"]
                )

                if (iteration + 1) % cfg["SOLVER"]["GRADIENT_ACCUMULATION_STEPS"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                metric_values = calculate_metrics(
                    output_dict["similarity_map"], batch["crop_coords"]
                )

                # Accumulate metrics for epoch
                epoch_metrics["loss"] += loss.item()
                for k, v in metric_values.items():
                    epoch_metrics[k] += v
                num_batches += 1

                # Iteration logging
                if iteration % log_period == 0:
                    # log_gpu_stats(rank)
                    lr = optimizer.param_groups[0]["lr"]
                    # Add AP and AR calculations
                    ap = (metric_values["AP50"] + metric_values["AP75"]) / 2
                    ar = (metric_values["AR50"] + metric_values["AR75"]) / 2
                    metric_values.update({"AP": ap, "AR": ar})
                    if rank == 0:
                        logger.info(
                            f"Iteration {iteration} - Loss: {accumulated_loss:.16f}"
                        )
                        logger.info(
                            f"Metrics - mAP: {metric_values['mAP']:.16f}, "
                            f"AP50: {metric_values['AP50']:.16f}, "
                            f"AP75: {metric_values['AP75']:.16f}, "
                            f"AR50: {metric_values['AR50']:.16f}, "
                            f"AR75: {metric_values['AR75']:.16f}"
                        )
                        print(f"Learning Rate: {lr:.16f}")
                        print("=" * 50)

                    metrics = {
                        "loss": loss.item(),
                        "lr": lr,
                        "epoch": epoch,
                        **metric_values,
                    }
                    performance_metrics.update(
                        metrics, global_iter, lr, log_to_console=(rank == 0)
                    )
                    performance_metrics.log_gpu_stats(global_iter)

                # Progress bar update
                desc = f'Epoch {epoch}/{num_epochs} | loss: {accumulated_loss:.16f} | mAP: {metric_values["mAP"]:.16f}'
                pbar.set_description(desc)

                # Save periodic checkpoint
                if iteration % save_period == 0 and rank == 0:
                    save_checkpoint(
                        (
                            model.module
                            if isinstance(model, DistributedDataParallel)
                            else model
                        ),
                        optimizer,
                        epoch,
                        iteration,
                        cfg["OUTPUT"]["DIR"],
                        f"model_epoch_{epoch:03d}_iter_{iteration:07d}.pth",
                    )

                pbar.update(1)
                accumulated_loss = 0

            # End of epoch
            if rank == 0:
                # Calculate epoch averages
                epoch_avg_metrics = {
                    f"epoch_{k}": v / num_batches for k, v in epoch_metrics.items()
                }
                epoch_avg_metrics["epoch"] = epoch
                epoch_avg_metrics["lr"] = optimizer.param_groups[0]["lr"]

                # Log epoch metrics (use update_epoch instead of update)
                performance_metrics.update_epoch(
                    epoch_avg_metrics, epoch, log_to_console=True
                )

                logger.info(f"\nEpoch {epoch} Summary:")
                for k, v in epoch_avg_metrics.items():
                    logger.info(f"  {k}: {v:.16f}")

                # Save epoch checkpoint
                save_checkpoint(
                    (
                        model.module
                        if isinstance(model, DistributedDataParallel)
                        else model
                    ),
                    optimizer,
                    epoch + 1,
                    0,
                    cfg["OUTPUT"]["DIR"],
                    f"model_epoch_{epoch:03d}_final.pth",
                )
                logger.info(f"Completed epoch {epoch}")

            pbar.close()

        # Final checkpoint
        if rank == 0:
            save_checkpoint(
                model.module if isinstance(model, DistributedDataParallel) else model,
                optimizer,
                num_epochs,
                0,
                cfg["OUTPUT"]["DIR"],
                "model_final.pth",
            )
            logger.info("Training completed successfully")

        # Only run evaluation if enabled in config
        if rank == 0 and cfg.get("EVAL", {}).get("ENABLED", False):
            logger.info("Starting evaluation...")
            evaluate(model, data_loader, cfg)
        else:
            logger.info("Skipping evaluation (not enabled in config)")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e

    finally:
        # Cleanup
        metrics_logger.close()
        if cfg["DISTRIBUTED"]["ENABLED"]:
            dist.destroy_process_group()


# def generate_heatmap(index, x_j, z_i, features, fpn_layer, name=None, anchor=None):
#     image = save_image(
#         x_j[index].cpu().permute(1, 2, 0).numpy().copy(), save=False, anchors=anchor
#     )
#     # Reshaping and normalizing
#     positive = z_i[index]
#     # if isinstance(features, list):
#     #     negative = features[fpn_layer][index]
#     # else:
#     #     negative = features[index, :, :]
#     # NOTE: combined layers
#     negative = combine_layers(features, index)
#
#     normalized_vector1 = F.normalize(positive, p=2, dim=-1)
#     grid = np.zeros((x_j[index].shape[1], x_j[index].shape[2]), dtype=np.float32)
#     length = np.zeros((x_j[index].shape[1], x_j[index].shape[2]), dtype=np.float32)
#
#     if isinstance(features, list):
#         num_cells_x = features[fpn_layer].shape[-1]
#         num_cells_y = features[fpn_layer].shape[-2]
#     else:
#         num_cells_x = features.shape[-1]
#         num_cells_y = features.shape[-2]
#
#     height, width = x_j.shape[2:]
#     grid_size_x = torch.div(width, num_cells_x, rounding_mode="floor")
#     grid_size_y = torch.div(height, num_cells_y, rounding_mode="floor")
#
#     # Calculate centers dynamically
#     centers = [
#         (int((r + 0.5) * grid_size_y), int((c + 0.5) * grid_size_x))
#         for r in range(num_cells_y)
#         for c in range(num_cells_x)
#     ]
#
#     for i, grid_indx in enumerate(centers):
#         r = torch.div(i, num_cells_x, rounding_mode="floor")
#         c = i % num_cells_x
#         normalized_vector2 = F.normalize(negative[:, r, c], p=2, dim=-1)
#         dot_product = torch.dot(normalized_vector1, normalized_vector2)
#         cosine_similarity = dot_product.item()
#
#         y_slice = slice(
#             grid_indx[0] - int(grid_size_y / 2), grid_indx[0] + int(grid_size_y / 2)
#         )
#         x_slice = slice(
#             grid_indx[1] - int(grid_size_x / 2), grid_indx[1] + int(grid_size_x / 2)
#         )
#
#         grid[y_slice, x_slice] += cosine_similarity
#         length[y_slice, x_slice] += 1.0
#
#     length[length == 0] = 1
#     grid = grid / length
#
#     min_n, max_n = np.min(grid), np.max(grid)
#     print("Grid val 0", min_n, max_n)
#     # Normalize the grid to [0, 255]
#     # grid = (grid - (-1.0)) * (255.0) / (1.0 - (-1.0))
#     grid = (grid - (min_n)) * (255.0) / (max_n - (min_n))
#     grid = grid.astype(np.uint8)
#     min_n, max_n = np.min(grid), np.max(grid)
#     print("Grid val 255", min_n, max_n)
#     # NOTE: FLIPING VALUES
#     grid = (max_n + min_n) - grid
#     min_n, max_n = np.min(grid), np.max(grid)
#     print("Grid val flip", min_n, max_n)
#
#     # Apply Gaussian blur and generate heatmap
#     blur = cv2.GaussianBlur(grid, (13, 13), 11)
#     heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
#     super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
#
#     # Save heatmap image
#     if name is None:
#         output_heatmap = save_image(
#             super_imposed_img, name=f"heat_img{index}", normalize=False
#         )
#     else:
#         output_heatmap = save_image(super_imposed_img, name=f"{name}", normalize=False)
#
#     return output_heatmap, grid
#
#
# def combine_layers(features, index=0):
#     common_size = features[0].shape[-2:]  # Adjust this to your desired common size
#
#     # Resize tensors to the common spatial size using bilinear interpolation
#     resize = lambda x: F.interpolate(
#         x, size=common_size, mode="bilinear", align_corners=False
#     )
#     tensor2_resized = resize(features[1])[index]
#     tensor3_resized = resize(features[2])[index]
#     tensor4_resized = resize(features[3])[index]
#     tensor5_resized = resize(features[4])[index]
#
#     # Combine resized tensors using element-wise summation
#     combined_tensor = (
#         features[0][index]
#         + tensor2_resized
#         + tensor3_resized
#         + tensor4_resized
#         + tensor5_resized
#     )
#     return combined_tensor
