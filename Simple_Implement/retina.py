import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
import yaml
from pycocotools.cocoeval import COCOeval
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import retinanet_resnet50_fpn

from tqdm import tqdm

# Insert the parent directory of 'downstream' so that model.py becomes visible
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from metrics import MetricsLogger, PerformanceMetrics
from model import UnsupervisedDetector


def collate_fn(batch):
    return tuple(zip(*batch))


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        return 0, 1


def load_config(config_file):
    """Load YAML configuration file"""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_model_retina(cfg, retina):
    if cfg["LOAD"]["PATH"] != "":
        path = cfg["LOAD"]["PATH"]
        if cfg["LOAD"]["SUPERVISED"] != "":
            print("Loading supervised")
            pretrained_weights = torch.load(path + "/" + cfg["LOAD"]["SUPERVISED"])
            retina.load_state_dict(pretrained_weights, strict=False)

        elif cfg["LOAD"]["UNSUPERVISED"] != "":
            print("Loading unsupervised")
            backbone = UnsupervisedDetector(load_config(path + "/" + "config.yaml"))
            pretrained_weights = torch.load(path + "/" + cfg["LOAD"]["UNSUPERVISED"])
            backbone.load_state_dict(pretrained_weights, strict=False)
            backbone = backbone.pipeline2.fpn

            retina.backbone = backbone

    return retina


class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)
        # Create a mapping from COCO category IDs to continuous index range
        self.continuous_category_id_to_coco = {}
        self.coco_to_continuous_category_id = {}

        # Get all valid category IDs from COCO
        cat_ids = self.coco.getCatIds()

        # Load all categories and create mapping
        categories = self.coco.loadCats(cat_ids)

        # Sort categories by ID to ensure consistent mapping
        categories = sorted(categories, key=lambda x: x["id"])

        print("\nInitializing COCO category mapping:")
        for idx, cat in enumerate(categories):
            # Map to a zero-based index range (keeping 0 for background)
            continuous_id = idx + 1  # +1 because 0 is reserved for background
            coco_id = cat["id"]

            self.coco_to_continuous_category_id[coco_id] = continuous_id
            self.continuous_category_id_to_coco[continuous_id] = coco_id

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        # Extract bounding boxes and labels
        boxes = []
        labels = []

        # Use list comprehension for faster processing
        # boxes, labels = (
        #     zip(
        #         *[
        #             (
        #                 obj["bbox"],
        #                 self.coco_to_continuous_category_id[obj["category_id"]],
        #             )
        #             for obj in target
        #             if obj["category_id"] in self.coco_to_continuous_category_id
        #         ]
        #     )
        #     if target
        #     else ([], [])
        # )

        for obj in target:
            category_id = obj["category_id"]
            if category_id in self.coco_to_continuous_category_id:
                boxes.append(obj["bbox"])
                continuous_id = self.coco_to_continuous_category_id[category_id]
                labels.append(continuous_id)

        if len(boxes) == 0:  # Handle images with no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            boxes[:, 2:] += boxes[:, :2]

            # Filter out invalid boxes (width or height <= 0)
            valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_indices]
            labels = labels[valid_indices]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
        }
        return img, target


def move_to_device(obj, device):
    """
    Recursively moves all tensors in a nested structure to the specified device.

    Args:
        obj: The object to move to the device. Can be a dict, list, tuple, or tensor.
        device: The target device (e.g., 'cuda' or 'cpu').

    Returns:
        The object with all tensors moved to the specified device.
    """
    if isinstance(obj, dict):  # If obj is a dictionary
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):  # If obj is a list
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):  # If obj is a tuple
        return tuple(move_to_device(v, device) for v in obj)
    elif torch.is_tensor(obj):  # If obj is a tensor
        return obj.to(device)
    else:
        return obj  # If not a tensor, return as is


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


def train(
    model, train_loader, val_dataset, device, cfg, optimizer, lr_scheduler
):
    """Training function with metrics logging and per-epoch evaluation"""
    # Setup logging and metrics (only on rank 0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = setup_logger(cfg["OUTPUT"]["DIR"], rank=rank)
    logger.info("Retina Code: 100 percent data")

    if rank == 0:
        metrics_logger = MetricsLogger(cfg["OUTPUT"]["DIR"], distributed=True)
        performance_metrics = PerformanceMetrics(model, metrics_logger)

    num_epochs = cfg["SOLVER"]["EPOCHS"]
    log_period = cfg["OUTPUT"]["LOG_PERIOD"]

    try:
        for epoch in range(num_epochs):
            # Set epoch for samplers
            train_loader.sampler.set_epoch(epoch)

            # Training phase
            model.train()
            epoch_metrics = {
                "loss": 0.0,
                "loss_classification": 0.0,
                "loss_bbox_regression": 0.0,
            }
            num_batches = 0

            if rank == 0:
                pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")

            for batch_idx, (images, targets) in enumerate(train_loader):
                # Move data to device
                images = [image.to(device) for image in images]
                targets = move_to_device(targets, device)

                # Training step
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                # Update metrics
                global_iter = epoch * len(train_loader) + batch_idx

                # Accumulate losses for epoch metrics
                epoch_metrics["loss"] += losses.item()
                for k, v in loss_dict.items():
                    if k in epoch_metrics:
                        epoch_metrics[f"loss_{k}"] += v.item()
                num_batches += 1

                # Log metrics periodically
                if batch_idx % log_period == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    metrics = {
                        "loss": losses.item(),
                        "lr": lr,
                        "epoch": epoch,
                        **{k: v.item() for k, v in loss_dict.items()},
                    }

                    if rank == 0 and global_iter != 0:
                        performance_metrics.update(
                            metrics, global_iter, lr, log_to_console=True
                        )
                        performance_metrics.log_gpu_stats(global_iter)

                    logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - "
                        f"Loss: {losses.item():.4f}, LR: {lr:.6f}"
                    )

                # Update progress bar only for rank 0
                if rank == 0:
                    pbar.update(1)

            if rank == 0:
                pbar.close()

            # Step the learning rate scheduler
            lr_scheduler.step()

            # Synchronize before validation
            if dist.is_initialized():
                dist.barrier()

            # Only run evaluation on rank 0
            if rank == 0:
                logger.info(f"Running evaluation for epoch {epoch}")
                eval_metrics = evaluate(model, val_dataset, device, cfg)

                # Log evaluation metrics for this epoch
                eval_epoch_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
                eval_epoch_metrics["epoch"] = epoch
                performance_metrics.update(
                    eval_epoch_metrics, epoch, log_to_console=True
                )

                # Save checkpoint after each epoch
                torch.save(
                    model.module.state_dict(),
                    os.path.join(cfg["OUTPUT"]["DIR"], f"model_epoch_{epoch:03d}.pth"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        cfg["OUTPUT"]["DIR"], f"optimizer_epoch_{epoch:03d}.pth"
                    ),
                )

    except Exception as e:
        if rank == 0:
            logger.error(f"Error during training: {str(e)}")
        raise e
    finally:
        if rank == 0:
            metrics_logger.close()


def evaluate(model, dataset, device, cfg):
    """Evaluation function with metrics logging"""
    logger = setup_logger(cfg["OUTPUT"]["DIR"], rank=0)

    # Create a DataLoader for evaluation
    eval_loader = DataLoader(
        dataset,
        batch_size=cfg["SOLVER"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["SYSTEM"]["NUM_WORKERS"],
        collate_fn=collate_fn,  # Make sure to use the same collate_fn as training
    )

    model.eval()
    all_predictions = []
    try:
        with torch.no_grad():
            for images, targets in tqdm(eval_loader, desc="Evaluating"):
                images = [image.to(device) for image in images]
                targets = move_to_device(targets, device)
                outputs = model(images)
                for i, output in enumerate(outputs):
                    image_id = targets[i]["image_id"].item()
                    for box, label, score in zip(
                        output["boxes"].cpu().numpy(),
                        output["labels"].cpu().numpy(),
                        output["scores"].cpu().numpy(),
                    ):
                        prediction = {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1]),
                            ],
                            "score": float(score),
                        }
                        all_predictions.append(prediction)

        if len(set([p["image_id"] for p in all_predictions])) == 0:
            logger.info("No predictions were made")
            return {}

        print("Prediction image_ids:", set([p["image_id"] for p in all_predictions]))
        print("Ground truth image_ids:", set(dataset.coco.getImgIds()))

        # COCO evaluation
        coco_gt = dataset.coco
        coco_dt = coco_gt.loadRes(all_predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Log COCO metrics
        eval_metrics = {
            "AP": coco_eval.stats[0],  # AP at IoU=0.50:0.95
            "AP50": coco_eval.stats[1],  # AP at IoU=0.50
            "AP75": coco_eval.stats[2],  # AP at IoU=0.75
            "AP_small": coco_eval.stats[3],  # AP for small objects
            "AP_medium": coco_eval.stats[4],  # AP for medium objects
            "AP_large": coco_eval.stats[5],  # AP for large objects
        }

        logger.info("Evaluation Results:")
        for k, v in eval_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return eval_metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise e


def setup(rank, world_size, dist_url):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank
    )


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def main_worker(rank, world_size, cfg):
    setup(rank, world_size, "tcp://localhost:12355")

    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Setup datasets and dataloaders
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CustomCocoDetection(
        root=cfg["DATA"]["TRAIN_ROOT"],
        annFile=cfg["DATA"]["TRAIN_ANN"],
        transform=transform,
    )

    val_dataset = CustomCocoDetection(
        root=cfg["DATA"]["VAL_ROOT"],
        annFile=cfg["DATA"]["VAL_ANN"],
        transform=transform,
    )

    # Reduce training dataset to 10%
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    subset_size = int(0.01 * len(train_indices))
    train_subset = Subset(train_dataset, train_indices[:subset_size])

    # Create samplers for DDP
    train_sampler = DistributedSampler(train_subset, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["SOLVER"]["BATCH_SIZE"],
        sampler=train_sampler,
        num_workers=cfg["SYSTEM"]["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["SOLVER"]["BATCH_SIZE"],
        sampler=val_sampler,
        num_workers=cfg["SYSTEM"]["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create RetinaNet model
    model = retinanet_resnet50_fpn(
        num_classes=len(train_dataset.continuous_category_id_to_coco) + 1,
        pretrained_backbone=False
    )
    model = load_model_retina(cfg, model)
    model = model.to(device)

    # Wrap model with DDP
    model = DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=False
    )

    # Create optimizer and scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["SOLVER"]["BASE_LR"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["SOLVER"]["BASE_LR"],
        momentum=0.9,
        weight_decay=0.0001
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train and evaluate
    try:
        if rank == 0:
            print(f"Starting training on {world_size} GPUs")
        train(
            model,
            train_loader,
            val_dataset,
            device,
            cfg,
            optimizer,
            lr_scheduler,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    # Updated configuration
    cfg = {
        "OUTPUT": {
            "DIR": os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S")),
            "LOG_PERIOD": 10,
        },
        "MODEL": {
            "BACKBONE": {
                "NAME": "resnet50",
                "FREEZE_AT": 2,
            }
        },
        "SOLVER": {
            "EPOCHS": 100,
            "BASE_LR": 0.01,
            "BATCH_SIZE": 4,
        },
        "SYSTEM": {
            "NUM_GPUS": torch.cuda.device_count(),
            "NUM_WORKERS": 2,
        },
        "DATA": {
            "TRAIN_ROOT": "/mnt/drive_test/coco/train2017",
            "TRAIN_ANN": "/mnt/drive_test/coco/annotations/instances_train2017.json",
            "VAL_ROOT": "/mnt/drive_test/coco/val2017",
            "VAL_ANN": "/mnt/drive_test/coco/annotations/instances_val2017.json",
        },
        "LOAD": {
            "PATH": "",
            "UNSUPERVISED": "",
            "SUPERVISED": "",
        },
    }

    # Create output directory
    os.makedirs(cfg["OUTPUT"]["DIR"], exist_ok=True)

    # Launch DDP training
    world_size = cfg["SYSTEM"]["NUM_GPUS"]
    mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
