from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import amp
import logging

SUPPORTED_BACKBONES = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2
}

BACKBONE_CHANNELS = {
    'resnet18': (128, 256, 512),      # C3, C4, C5 channels
    'resnet34': (128, 256, 512),
    'resnet50': (512, 1024, 2048),
    'resnet101': (512, 1024, 2048),
    'resnet152': (512, 1024, 2048),
    'resnext50_32x4d': (512, 1024, 2048),
    'resnext101_32x8d': (512, 1024, 2048),
    'wide_resnet50_2': (512, 1024, 2048),
    'wide_resnet101_2': (512, 1024, 2048)
}

class FPNBlock(nn.Module):
    """Feature Pyramid Network block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, upsampled_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1x1(x)
        if upsampled_features is not None:
            x = x + F.interpolate(upsampled_features, size=x.shape[-2:], mode='nearest')
        x = self.conv3x3(x)
        return x

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network implementation"""
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        last_inner = self.lateral_convs[-1](features[-1])
        results = [self.fpn_convs[-1](last_inner)]
        
        for idx in range(len(features)-2, -1, -1):
            inner_lateral = self.lateral_convs[idx](features[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.fpn_convs[idx](last_inner))
        
        return results

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UnsupervisedObjectDetection(nn.Module):
    """Main model implementing the paper's architecture with configurable backbones"""
    def __init__(self, 
                 temperature: float = 0.5,
                 backbone_name: str = "resnet50",
                 fpn_channels: int = 256,
                 projection_dim: int = 128,
                 pretrained: bool = True):
        super().__init__()
        
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone {backbone_name} not supported. "
                           f"Supported backbones: {list(SUPPORTED_BACKBONES.keys())}")
        
        self.backbone_name = backbone_name
        self.temperature = temperature
        
        # Get backbone builder and channel dimensions
        backbone_fn = SUPPORTED_BACKBONES[backbone_name]
        in_channels_list = BACKBONE_CHANNELS[backbone_name]
        
        # Pipeline 1: Backbone for crops
        backbone1 = backbone_fn(pretrained=pretrained)
        self.backbone1 = nn.Sequential(*list(backbone1.children())[:-2])  # Remove avg pool and fc
        
        # Pipeline 2: Backbone with FPN
        backbone2 = backbone_fn(pretrained=pretrained)
        self.backbone2 = nn.Sequential(*list(backbone2.children())[:-2])
        
        # FPN
        self.fpn = FeaturePyramidNetwork(in_channels_list, fpn_channels)
        
        # Projection heads (adjust input dimension based on backbone)
        proj_in_dim = in_channels_list[-1]  # Use the last channel dimension
        self.proj_head1 = ProjectionHead(proj_in_dim, 512, projection_dim)
        self.proj_head2 = ProjectionHead(fpn_channels, 512, projection_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following paper's specification"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    @amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, images: torch.Tensor, crops: torch.Tensor, 
                centers: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass implementing both pipelines"""
        batch_size = images.shape[0]
        
        # Pipeline 1: Process crops
        crop_features = self.backbone1(crops)
        crop_features = F.adaptive_avg_pool2d(crop_features, (1, 1))
        crop_features = crop_features.view(batch_size, -1)
        z_i = self.proj_head1(crop_features)
        
        # Pipeline 2: Process full images through FPN
        backbone_features = []
        x = images
        for i, layer in enumerate(self.backbone2):
            x = layer(x)
            if i in [5, 6, 7]:  # Store C3, C4, C5 features
                backbone_features.append(x)
        
        fpn_features = self.fpn(backbone_features)
        
        # Get positive and negative pairs based on locations
        z_j, z_a = self._get_location_based_features(
            fpn_features, centers, batch_size
        )
        
        # Compute loss
        loss = self._compute_loss(z_i, z_j, z_a)
        
        # Compute additional metrics
        with torch.no_grad():
            metrics = {
                'pos_sim': F.cosine_similarity(z_i, z_j).mean(),
                'neg_sim': F.cosine_similarity(z_i, z_a.view(-1, z_a.size(-1))).mean(),
                'z_i_norm': z_i.norm(dim=1).mean(),
                'z_j_norm': z_j.norm(dim=1).mean(),
                'z_a_norm': z_a.norm(dim=2).mean()
            }
        
        return loss, metrics