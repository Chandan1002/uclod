import logging
import os
import random
from datetime import datetime

import torch
import torchvision
import torchvision.transforms as T
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

from metrics import MetricsLogger, PerformanceMetrics

# class CustomCocoDetection(CocoDetection):
#     def __getitem__(self, idx):
#         img, target = super().__getitem__(idx)
#
#         # Extract bounding boxes and labels
#         boxes = [obj["bbox"] for obj in target]
#         labels = [obj["category_id"] for obj in target]
#
#         if len(boxes) == 0:  # Handle images with no objects
#             boxes = torch.zeros((0, 4), dtype=torch.float32)
#             labels = torch.zeros((0,), dtype=torch.int64)
#         else:
#             # Convert to tensors
#             boxes = torch.tensor(boxes, dtype=torch.float32)
#             labels = torch.tensor(labels, dtype=torch.int64)
#
#             # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
#             boxes[:, 2:] += boxes[:, :2]  # Convert width/height to max coords
#
#             # Filter out invalid boxes (width or height <= 0)
#             valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
#             boxes = boxes[valid_indices]
#             labels = labels[valid_indices]
#
#         target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
#         return img, target


# class CustomCocoDetection(CocoDetection):
#     def __getitem__(self, idx):
#         img, target = super().__getitem__(idx)
        
#         # Get the actual COCO image ID from the original annotation
#         image_id = self.ids[idx]  # This gets the actual COCO image ID
        
#         # Extract bounding boxes and labels
#         boxes = [obj["bbox"] for obj in target]
#         labels = [obj["category_id"] for obj in target]

#         if len(boxes) == 0:  # Handle images with no objects
#             boxes = torch.zeros((0, 4), dtype=torch.float32)
#             labels = torch.zeros((0,), dtype=torch.int64)
#         else:
#             # Convert to tensors
#             boxes = torch.tensor(boxes, dtype=torch.float32)
#             labels = torch.tensor(labels, dtype=torch.int64)

#             # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
#             boxes[:, 2:] += boxes[:, :2]  # Convert width/height to max coords

#             # Filter out invalid boxes (width or height <= 0)
#             valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
#             boxes = boxes[valid_indices]
#             labels = labels[valid_indices]

#         target = {
#             "boxes": boxes,
#             "labels": labels,
#             "image_id": torch.tensor([image_id])  # Use actual COCO image ID
#         }
#         return img, target
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
        categories = sorted(categories, key=lambda x: x['id'])
        
        print("\nInitializing COCO category mapping:")
        for idx, cat in enumerate(categories):
            # Map to a zero-based index range (keeping 0 for background)
            continuous_id = idx + 1  # +1 because 0 is reserved for background
            coco_id = cat['id']
            
            self.coco_to_continuous_category_id[coco_id] = continuous_id
            self.continuous_category_id_to_coco[continuous_id] = coco_id
            
            # print(f"COCO ID {coco_id} -> Continuous ID {continuous_id} ({cat['name']})")
            
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        # print(f"\nProcessing image {image_id}:")
        for obj in target:
            category_id = obj["category_id"]
            if category_id in self.coco_to_continuous_category_id:
                boxes.append(obj["bbox"])
                continuous_id = self.coco_to_continuous_category_id[category_id]
                labels.append(continuous_id)
            #     print(f"  COCO category {category_id} -> continuous ID {continuous_id}")
            # else:
            #     print(f"  WARNING: Skipping unknown category ID {category_id}")

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
            "image_id": torch.tensor([image_id])
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


def train(model, train_loader, val_set, device, cfg):
    """Training function with metrics logging and per-epoch evaluation"""
    # Setup logging and metrics
    logger = setup_logger(cfg["OUTPUT"]["DIR"], rank=0)
    metrics_logger = MetricsLogger(cfg["OUTPUT"]["DIR"], distributed=False)
    performance_metrics = PerformanceMetrics(model, metrics_logger)

    num_epochs = cfg["SOLVER"]["EPOCHS"]
    log_period = cfg["OUTPUT"]["LOG_PERIOD"]

    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_metrics = {
                "loss": 0.0,
                "loss_classifier": 0.0,
                "loss_box_reg": 0.0,
                "loss_objectness": 0.0,
                "loss_rpn_box_reg": 0.0,
            }
            num_batches = 0

            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")

            for batch_idx, (images, targets) in enumerate(train_loader):
                # Move data to device
                images = [image.to(device) for image in images]
                targets = move_to_device(targets, device)

                # # Training step
                # optimizer.zero_grad()
                # loss_dict = model(images, targets)
                # losses = sum(loss for loss in loss_dict.values())
                # losses.backward()
                # optimizer.step()
                # Training step
                optimizer.zero_grad()
                
                # Debug prints
                # for idx, target in enumerate(targets):
                #     print(f"\nBatch item {idx}:")
                #     print(f"Labels: {target['labels']}")
                #     print(f"Boxes shape: {target['boxes'].shape}")
                #     print(f"Labels shape: {target['labels'].shape}")
                #     print(f"Image ID: {target['image_id']}")
                #     
                #     # Validate label values
                #     if len(target['labels']) > 0:
                #         print(f"Min label: {target['labels'].min()}")
                #         print(f"Max label: {target['labels'].max()}")
                #         
                #     # Validate box coordinates
                #     if len(target['boxes']) > 0:
                #         print(f"Box coordinate ranges:")
                #         print(f"x_min: {target['boxes'][:, 0].min():.2f} to {target['boxes'][:, 0].max():.2f}")
                #         print(f"y_min: {target['boxes'][:, 1].min():.2f} to {target['boxes'][:, 1].max():.2f}")
                #         print(f"x_max: {target['boxes'][:, 2].min():.2f} to {target['boxes'][:, 2].max():.2f}")
                #         print(f"y_max: {target['boxes'][:, 3].min():.2f} to {target['boxes'][:, 3].max():.2f}")
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

                # Update metrics
                global_iter = epoch * len(train_loader) + batch_idx

                # Accumulate losses for epoch metrics
                epoch_metrics["loss"] += losses.item()
                for k, v in loss_dict.items():
                    epoch_metrics[k] += v.item()
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

                    performance_metrics.update(
                        metrics, global_iter, lr, log_to_console=True
                    )
                    performance_metrics.log_gpu_stats(global_iter)

                    logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - "
                        f"Loss: {losses.item():.4f}, LR: {lr:.6f}"
                    )

                pbar.update(1)

            pbar.close()

            # End of epoch logging
            lr_scheduler.step()

            # Calculate epoch averages
            epoch_avg_metrics = {
                f"epoch_{k}": v / num_batches for k, v in epoch_metrics.items()
            }
            epoch_avg_metrics["epoch"] = epoch
            epoch_avg_metrics["lr"] = optimizer.param_groups[0]["lr"]

            # Log epoch metrics
            performance_metrics.update_epoch(
                epoch_avg_metrics, epoch, log_to_console=True
            )

            logger.info(f"\nEpoch {epoch} Summary:")
            for k, v in epoch_avg_metrics.items():
                logger.info(f"  {k}: {(v/len(train_loader)):.4f}")

            # Evaluation phase
            logger.info(f"Running evaluation for epoch {epoch}")
            eval_metrics = evaluate(model, val_set, device, cfg)

            # Log evaluation metrics for this epoch
            eval_epoch_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
            eval_epoch_metrics["epoch"] = epoch
            performance_metrics.update(
                eval_epoch_metrics, epoch, log_to_console=True
            )

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(
                cfg["OUTPUT"]["DIR"], f"model_epoch_{epoch:03d}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e
    finally:
        metrics_logger.close()


def evaluate(model, dataset, device, cfg):
    """Evaluation function with metrics logging"""
    logger = setup_logger(cfg["OUTPUT"]["DIR"], rank=0)
    # metrics_logger = MetricsLogger(cfg["OUTPUT"]["DIR"], distributed=False)

    model.eval()
    all_predictions = []
    try:
        with torch.no_grad():
            for images, targets in tqdm(dataset, desc="Evaluating"):
                images = [img.to(device) for img in images]
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
            logger.info("Did not worked")

        print("Prediction image_ids:", set([p["image_id"] for p in all_predictions]))
        print("Ground truth image_ids:", set(val_dataset.coco.getImgIds()))

        if len(set([p["image_id"] for p in all_predictions])) == 0:
            logger.info("Did not worked")
            return {}


        # COCO evaluation
        coco_gt = val_dataset.coco
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

        # Log evaluation metrics
        # metrics_logger.update(eval_metrics, 0)
        logger.info("Evaluation Results:")
        for k, v in eval_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return eval_metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise e


if __name__ == "__main__":
    # Configuration dictionary similar to train.py
    cfg = {
        "OUTPUT": {
            "DIR": os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S")),
            "LOG_PERIOD": 10,
        },
        "SOLVER": {
            "EPOCHS": 100,
            "BASE_LR": 0.001,
        },
        "SYSTEM": {"DEVICE": "cuda" if torch.cuda.is_available() else "cpu"},
    }

    # Create output directory
    os.makedirs(cfg["OUTPUT"]["DIR"], exist_ok=True)

    # Step 1: Setup COCO dataset
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CustomCocoDetection(
        root="/mnt/drive_test/coco/train2017",
        annFile="/mnt/drive_test/coco/annotations/instances_train2017.json",
        transform=transform,
    )

    val_dataset = CustomCocoDetection(
        root="/mnt/drive_test/coco/val2017",
        annFile="/mnt/drive_test/coco/annotations/instances_val2017.json",
        transform=transform,
    )

    # The number of classes should be number of COCO categories + 1 (for background)
    num_classes = len(train_dataset.continuous_category_id_to_coco) + 1

    # Reduce training dataset to 10%
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    subset_size = int(0.1 * len(train_indices))
    train_subset = Subset(train_dataset, train_indices[:subset_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Step 2: Load pre-trained Faster R-CNN with FPN model
    model = fasterrcnn_resnet50_fpn(pretrained=False)

    # Replace the classification head to match the number of COCO classes
    # num_classes = 80  # 80 classes + background + other special classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    # Load the pre-trained weights
    # pretrained_weights = torch.load("/home/ecx/ml/Simple_implementation/output/20241231_142448/model_epoch_339_final.pth")
    # model.load_state_dict(pretrained_weights, strict=False)

    # Step 3: Define the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Step 4: Training and evaluation
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Train the model with validation loader for per-epoch evaluation
    train(model, train_loader, val_loader, device, cfg)
    # evaluate(model, val_loader, device, cfg)
