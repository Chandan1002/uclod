import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Setup logger
logger = logging.getLogger(__name__)

class UnsupervisedDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg['SYSTEM']['DEVICE'])
        
        logger.info("Initializing UnsupervisedDetector")
        
        # Pipeline 1: ResNet for crop processing
        self.pipeline1 = ResNetEncoder(cfg)
        logger.info("Pipeline 1 (ResNet) initialized")
        
        # Pipeline 2: ResNet+FPN for full image
        self.pipeline2 = FPNEncoder(cfg)
        logger.info("Pipeline 2 (FPN) initialized")
        
        # Temperature parameter for NT-Xent loss
        self.temperature = cfg['MODEL']['PROJECTION']['TEMPERATURE']
        logger.info(f"Temperature parameter set to {self.temperature}")

    def forward(self, batch):
        """Forward pass through both pipelines"""
        logger.debug("Starting forward pass")
        
        x_i = batch['image_crop']  # Random crop
        x_j = batch['image_original']  # Full image
        crop_coords = batch['crop_coords']
        
        batch_size = x_i.size(0)
        logger.debug(f"Processing batch of size {batch_size}")
        logger.debug(f"Crop tensor shape: {x_i.shape}")
        logger.debug(f"Original image tensor shape: {x_j.shape}")
        
        # Pipeline 1: Process crop
        z_i = self.pipeline1(x_i)
        logger.debug(f"Pipeline 1 output shape: {z_i.shape}")
        
        # Pipeline 2: Process full image with FPN
        z_j, z_a = self.pipeline2(x_j, crop_coords)
        logger.debug(f"Pipeline 2 outputs - z_j shape: {z_j.shape}, z_a shape: {z_a.shape}")
        
        # Compute anchor-based NT-Xent loss
        loss = self.compute_anchor_based_ntxent_loss(z_i, z_j, z_a, batch_size)
        logger.debug(f"Computed loss: {loss.item():.4f}")
        
        return {"loss": loss}

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
        neg_sim_inter.masked_fill_(mask, float('-inf'))
        
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
        
        if cfg['MODEL']['BACKBONE']['NAME'] == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            logger.info("Using ResNet50 backbone")
        else:
            raise ValueError(f"Unsupported backbone: {cfg['MODEL']['BACKBONE']['NAME']}")
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj_dim = cfg['MODEL']['PROJECTION']['DIM']
        
        logger.info(f"Projection dimension: {self.proj_dim}")
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.proj_dim)
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
            backbone_name=cfg['MODEL']['BACKBONE']['NAME'],
            pretrained=True,
            trainable_layers=5
        )
        logger.info("FPN backbone initialized")
        
        self.proj_dim = cfg['MODEL']['PROJECTION']['DIM']
        self.num_anchors = 10
        logger.info(f"FPN projection dimension: {self.proj_dim}")
        logger.info(f"Number of anchors: {self.num_anchors}")
        
        # Projection head for FPN features
        self.proj_head = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, self.proj_dim, 1)
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
        
        # Process each FPN level
        for level_name, features in fpn_outputs.items():
            logger.debug(f"Processing FPN level {level_name}, feature shape: {features.shape}")
            
            # Project features
            features = self.proj_head(features)
            h, w = features.shape[-2:]
            
            # Create grid coordinates
            grid_y = torch.arange(h, device=device).float()
            grid_x = torch.arange(w, device=device).float()
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
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
                    perm = torch.randperm(len(valid_indices))[:self.num_anchors]
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
            neg_features = neg_features.view(batch_size, -1, self.num_anchors, self.proj_dim)
            z_a = torch.mean(neg_features, dim=1)
            
            logger.debug(f"Final z_j shape: {z_j.shape}")
            logger.debug(f"Final z_a shape: {z_a.shape}")
            
            return z_j, z_a
            
        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            logger.error(f"pos_features_list length: {len(pos_features_list)}")
            logger.error(f"neg_features_list length: {len(neg_features_list)}")
            raise