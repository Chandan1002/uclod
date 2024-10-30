import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import numpy as np
import io
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
import matplotlib.colors as mcolors

class DetectionVisualizer:
    """Visualizes detection results and similarity heatmaps"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.cmap = plt.get_cmap('viridis')
    
    def visualize_batch(self, 
                       images: torch.Tensor, 
                       similarity_maps: torch.Tensor,
                       iteration: int):
        """Visualizes a batch of images with their similarity heatmaps"""
        B, C, H, W = images.shape
        fig, axes = plt.subplots(B, 2, figsize=(12, 6*B))
        
        # Handle single image case
        if B == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(B, 8)):  # Limit to 8 images maximum
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Similarity heatmap
            heatmap = similarity_maps[i].cpu().numpy()
            axes[i, 1].imshow(heatmap, cmap='hot')
            axes[i, 1].set_title('Similarity Heatmap')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        # Save to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        
        self.writer.add_image(f'detection/visualization', 
                             image, 
                             iteration)
        plt.close()

    def create_heatmap(self, tensor: torch.Tensor) -> torch.Tensor:
        """Creates a colored heatmap from a tensor using matplotlib"""
        heatmap = tensor.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap
        colored_heatmap = self.cmap(heatmap)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        return torch.from_numpy(colored_heatmap).permute(2, 0, 1)

    def plot_fpn_features(self, 
                         fpn_features: Dict[str, torch.Tensor],
                         iteration: int):
        """Visualizes FPN features at different levels"""
        for level, features in enumerate(fpn_features):
            # Average across channels
            feature_map = features.mean(dim=1)
            
            # Create heatmap
            heatmap = self.create_heatmap(feature_map)
            
            self.writer.add_image(f'fpn/level_{level}',
                                heatmap, 
                                iteration)

    def plot_grid_accuracy(self, 
                          pred_grid: torch.Tensor,
                          gt_grid: torch.Tensor,
                          iteration: int):
        """Visualizes grid accuracy comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.heatmap(pred_grid.cpu(), ax=ax1, cmap='viridis')
        ax1.set_title('Predicted Grid')
        
        sns.heatmap(gt_grid.cpu(), ax=ax2, cmap='viridis')
        ax2.set_title('Ground Truth Grid')
        
        # Save to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        
        self.writer.add_image('grid/accuracy_comparison',
                             image,
                             iteration)
        plt.close()

    def close(self):
        """Closes the tensorboard writer"""
        self.writer.close()