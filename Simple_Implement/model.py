import logging
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.utils import make_grid

# Setup logger
logger = logging.getLogger(__name__)

# def save_similarity_heatmap(similarity_map, orig_image, boxes, confidences, save_path):
#     """Save heatmap visualization with boxes"""
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot original image with boxes
#     ax1.imshow(orig_image)
#     if boxes is not None and len(boxes) > 0 and not isinstance(boxes, float):
#         for box, conf in zip(boxes, confidences):
#             x, y, w, h = box
#             rect = patches.Rectangle((float(x), float(y)), float(w), float(h),
#                                    linewidth=2, edgecolor='r', facecolor='none')
#             ax1.add_patch(rect)
#             ax1.text(float(x), float(y)-5, f'{float(conf):.2f}', color='red')
#     ax1.set_title('Detections')

#     # Plot heatmap
#     heatmap = ax2.imshow(similarity_map, cmap='viridis')
#     plt.colorbar(heatmap, ax=ax2)
#     ax2.set_title('Similarity Heatmap')

#     plt.savefig(save_path)
#     plt.close()


class UnsupervisedDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg["SYSTEM"]["DEVICE"])

        logger.info("Initializing UnsupervisedDetector")

        # Pipeline 1: ResNet for crop processing
        self.pipeline1 = ResNetEncoder(cfg)
        logger.info("Pipeline 1 (ResNet) initialized")

        # Pipeline 2: ResNet+FPN for full image
        self.pipeline2 = FPNEncoder(cfg)
        logger.info("Pipeline 2 (FPN) initialized")

        # Temperature parameter for NT-Xent loss
        self.temperature = cfg["MODEL"]["PROJECTION"]["TEMPERATURE"]
        logger.info(f"Temperature parameter set to {self.temperature}")

    def predict_boxes(self, features_map, confidence_threshold=0.0000000000000001):
        """
        Convert feature map to bounding box predictions
        Returns empty tensors if no detections above threshold
        """
        try:
            batch_size = features_map.size(0)
            feature_size = features_map.size(2)  # Assuming square feature map

            # Get top confidences and indices
            confidences, indices = features_map.view(batch_size, -1).max(dim=1)

            # Convert indices to grid coordinates
            grid_size = 800 / feature_size  # Assuming 800x800 input image
            y = (indices // feature_size) * grid_size
            x = (indices % feature_size) * grid_size

            # Fixed size boxes
            w = h = grid_size * 2

            # Stack coordinates [x, y, w, h]
            boxes = torch.stack(
                [x, y, w * torch.ones_like(x), h * torch.ones_like(x)], dim=1
            )

            # Filter by confidence
            mask = confidences > confidence_threshold
            if mask.any():
                boxes = boxes[mask]
                confidences = confidences[mask]
            else:
                # If no detections above threshold, return empty tensors
                boxes = torch.zeros((0, 4), device=features_map.device)
                confidences = torch.zeros(0, device=features_map.device)

            return boxes, confidences

        except Exception as e:
            logger.error(f"Error in predict_boxes: {str(e)}")
            # Return empty tensors on error
            return (
                torch.zeros((0, 4), device=features_map.device),
                torch.zeros(0, device=features_map.device),
            )

    def compute_similarity_map(self, features):
        """Compute similarity map from features"""
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity map
        b, c, h, w = features.shape
        features_flat = features.view(b, c, -1)
        similarity_map = torch.bmm(features_flat.transpose(1, 2), features_flat)
        similarity_map = similarity_map.view(b, h, w, h, w)

        # Get max similarity along spatial dimensions
        similarity_map, _ = similarity_map.max(dim=-1)
        similarity_map, _ = similarity_map.max(dim=-1)

        return similarity_map

    def forward(self, batch):
        """Forward pass through model - handles both training and evaluation modes"""
        if not self.training:
            logger.debug("Running in evaluation mode")
            # Evaluation mode
            try:
                x_j = batch["image_original"]
                x_i = batch["image_crop"]  # Random crop
                fpn_outputs = self.pipeline2.fpn(x_j)
                reduced_fpn = [self.pipeline2.proj_head(f) for _, f in fpn_outputs.items()]
                z_i = self.pipeline1(x_i)

                # Safely get the first FPN level
                first_level = sorted(fpn_outputs.keys())[0]
                features = self.pipeline2.proj_head(fpn_outputs[first_level])

                # Compute boxes and similarity map
                boxes, confidences = self.predict_boxes(features)
                similarity_map = self.compute_similarity_map(features)

                if self.cfg["OUTPUT"].get("SAVE_VIZ", False):
                    from visualization import EvalVisualization

                    visualizer = EvalVisualization(self.cfg)
                    for i in range(len(batch["image_original"])):
                        visualizer.save_detection_visualization(
                            image=batch["image_original"][i],
                            similarity_map=similarity_map[i],
                            pred_boxes=boxes[i : i + 1] if boxes is not None else None,
                            gt_boxes=batch.get("crop_coords", None),
                            confidences=(
                                confidences[i : i + 1]
                                if confidences is not None
                                else None
                            ),
                            metrics=None,  # Metrics calculated in evaluate()
                            batch_idx=0,  # These will be set in evaluate()
                            sample_idx=i,
                        )
                    # else:
                    #     logger.warning("No 'image_original' found in batch during evaluation, skipping visualization.")

                return {
                    "boxes": boxes,
                    "confidences": confidences,
                    "similarity_map": similarity_map,
                    "fpn_features": reduced_fpn,
                    "resnet_features": z_i,
                }

            except Exception as e:
                logger.error(f"Error in evaluation mode: {str(e)}")
                raise

        # Training mode
        logger.debug("Running in training mode")
        try:
            x_i = batch["image_crop"]  # Random crop
            x_j = batch["image_original"]  # Full image
            crop_coords = batch["crop_coords"]

            batch_size = x_i.size(0)
            logger.debug(f"Processing batch of size {batch_size}")
            logger.debug(f"Crop tensor shape: {x_i.shape}")
            logger.debug(f"Original image tensor shape: {x_j.shape}")

            # Pipeline 1: Process crop
            z_i = self.pipeline1(x_i)
            logger.debug(f"Pipeline 1 output shape: {z_i.shape}")

            # Pipeline 2: Process full image with FPN
            z_j, z_a, fpn_outputs = self.pipeline2(x_j, crop_coords)
            logger.debug(
                f"Pipeline 2 outputs - z_j shape: {z_j.shape}, z_a shape: {z_a.shape}"
            )

            # Calculate similarity map
            z_i_norm = F.normalize(z_i, dim=1)
            z_j_norm = F.normalize(z_j, dim=1)
            similarity_map = torch.sum(
                z_i_norm.unsqueeze(2) * z_j_norm.unsqueeze(1), dim=1
            )
            logger.debug(f"Similarity map shape: {similarity_map.shape}")

            # Compute anchor-based NT-Xent loss
            loss = self.compute_anchor_based_ntxent_loss(z_i, z_j, z_a, batch_size)
            logger.debug(f"Computed loss: {loss.item():.16f}")

            # Create output dictionary
            output_dict = {
                "loss": loss,
                "similarity_map": similarity_map,
                "fpn_features": fpn_outputs,
                "resnet_features": z_i,
            }

            return output_dict

        except Exception as e:
            logger.error(f"Error in training mode: {str(e)}")
            raise

    def compute_anchor_based_ntxent_loss(self, z_i, z_j, z_a, batch_size):
        """Compute anchor-based NT-Xent loss with proper shape handling"""
        logger.debug("Computing anchor-based NT-Xent loss")

        # Normalize all embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z_a = F.normalize(z_a, dim=2)

        # Compute positive similarity
        pos_sim = torch.sum(z_i * z_j, dim=1)
        logger.debug(f"Positive similarity shape: {pos_sim.shape}")

        # Compute negative similarities
        neg_sim_inter = torch.mm(z_i, z_j.t())
        neg_sim_anchors = torch.bmm(z_i.unsqueeze(1), z_a.transpose(1, 2)).squeeze(1)

        logger.debug(f"Inter-image negative similarity shape: {neg_sim_inter.shape}")
        logger.debug(f"Anchor negative similarity shape: {neg_sim_anchors.shape}")

        # Scale by temperature
        pos_sim = pos_sim / self.temperature
        neg_sim_inter = neg_sim_inter / self.temperature
        neg_sim_anchors = neg_sim_anchors / self.temperature

        # Mask out self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        neg_sim_inter.masked_fill_(mask, float("-inf"))

        # Combine all negative similarities
        neg_sim = torch.cat([neg_sim_inter, neg_sim_anchors], dim=1)
        logger.debug(f"Combined negative similarity shape: {neg_sim.shape}")

        # Final logits and labels
        logits = torch.cat([pos_sim.view(-1, 1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class ResNetEncoder(nn.Module):
    """Pipeline 1: ResNet encoder for crop processing"""

    def __init__(self, cfg):
        super().__init__()
        logger.info("Initializing ResNetEncoder")

        if cfg["MODEL"]["BACKBONE"]["NAME"] == "resnet50":
            backbone = models.resnet50(pretrained=True)
            logger.info("Using ResNet50 backbone")
        else:
            raise ValueError(
                f"Unsupported backbone: {cfg['MODEL']['BACKBONE']['NAME']}"
            )

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj_dim = cfg["MODEL"]["PROJECTION"]["DIM"]

        logger.info(f"Projection dimension: {self.proj_dim}")

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.proj_dim),
        )

    def forward(self, x):
        logger.debug(f"ResNetEncoder input shape: {x.shape}")
        features = self.backbone(x)
        logger.debug(f"Backbone features shape: {features.shape}")
        out = self.projection(features)
        logger.debug(f"Projection output shape: {out.shape}")
        return out


class FPNEncoder(nn.Module):
    """Pipeline 2: FPN encoder for full image processing"""

    def __init__(self, cfg):
        super().__init__()
        logger.info("Initializing FPNEncoder")

        self.fpn = resnet_fpn_backbone(
            backbone_name=cfg["MODEL"]["BACKBONE"]["NAME"],
            pretrained=True,
            trainable_layers=5,
        )
        logger.info("FPN backbone initialized")

        self.proj_dim = cfg["MODEL"]["PROJECTION"]["DIM"]
        self.num_anchors = 10
        logger.info(f"FPN projection dimension: {self.proj_dim}")
        logger.info(f"Number of anchors: {self.num_anchors}")

        # Projection head for FPN features
        self.proj_head = nn.Sequential(
            nn.Conv2d(256, 512, 1), nn.ReLU(), nn.Conv2d(512, self.proj_dim, 1)
        )

    def forward(self, x, crop_coords):
        """Forward pass through FPN and feature sampling"""
        logger.debug(f"FPNEncoder input shape: {x.shape}")

        batch_size = x.size(0)
        device = x.device

        # Get FPN features
        fpn_outputs = self.fpn(x)
        logger.debug(f"FPN output keys: {fpn_outputs.keys()}")

        pos_features_list = []
        neg_features_list = []
        reduced_fpn = []

        # Process each FPN level
        for level_name, features in fpn_outputs.items():
            logger.debug(
                f"Processing FPN level {level_name}, feature shape: {features.shape}"
            )

            # Project features
            features = self.proj_head(features)
            reduced_fpn.append(features)
            h, w = features.shape[-2:]

            # Create grid coordinates
            grid_y = torch.arange(h, device=device).float()
            grid_x = torch.arange(w, device=device).float()
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid_points = torch.stack([grid_x, grid_y], dim=-1)

            # Scale grid to match image coordinates
            scale = x.shape[-1] / h
            grid_points = grid_points * scale

            for b in range(batch_size):
                # Get crop center
                y, x_coord, h_crop, w_crop = crop_coords[b]
                center_x = x_coord + w_crop / 2
                center_y = y + h_crop / 2
                center = torch.tensor([center_x, center_y], device=device)

                # Calculate distances to all grid points
                distances = torch.sum((grid_points - center.view(1, 1, 2)) ** 2, dim=-1)

                # Get closest point for positive sample
                min_dist_idx = torch.argmin(distances.view(-1))
                pos_y, pos_x = min_dist_idx // w, min_dist_idx % w
                pos_feature = features[b, :, pos_y, pos_x]
                pos_features_list.append(pos_feature)

                # Get negative samples
                dist_threshold = distances[pos_y, pos_x] * 2
                valid_mask = distances > dist_threshold
                valid_indices = torch.nonzero(valid_mask)

                if len(valid_indices) >= self.num_anchors:
                    perm = torch.randperm(len(valid_indices))[: self.num_anchors]
                    neg_indices = valid_indices[perm]
                else:
                    # If not enough valid points, sample with replacement
                    indices = torch.randint(0, len(valid_indices), (self.num_anchors,))
                    neg_indices = valid_indices[indices]

                # Gather negative features
                neg_features = []
                for idx in neg_indices:
                    ny, nx = idx
                    neg_feature = features[b, :, ny, nx]
                    neg_features.append(neg_feature)

                neg_features = torch.stack(neg_features)
                neg_features_list.append(neg_features)

        logger.debug(f"Collected {len(pos_features_list)} positive features")
        logger.debug(f"Collected {len(neg_features_list)} sets of negative features")

        # Stack and process features
        try:
            pos_features = torch.stack(pos_features_list)
            pos_features = pos_features.view(batch_size, -1, self.proj_dim)
            z_j = torch.mean(pos_features, dim=1)

            neg_features = torch.stack(neg_features_list)
            neg_features = neg_features.view(
                batch_size, -1, self.num_anchors, self.proj_dim
            )
            z_a = torch.mean(neg_features, dim=1)

            logger.debug(f"Final z_j shape: {z_j.shape}")
            logger.debug(f"Final z_a shape: {z_a.shape}")

            return z_j, z_a, reduced_fpn

        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            logger.error(f"pos_features_list length: {len(pos_features_list)}")
            logger.error(f"neg_features_list length: {len(neg_features_list)}")
            raise
