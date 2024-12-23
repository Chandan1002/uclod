import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import make_grid

class EvalVisualization:
    def __init__(self, cfg):
        self.cfg = cfg
        self.viz_dir = os.path.join(cfg['OUTPUT']['DIR'], 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def save_detection_visualization(self, image, similarity_map, pred_boxes, gt_boxes, 
                               confidences, metrics, batch_idx, sample_idx):
        """Create visualization with proper normalization and tensor handling"""
        fig = plt.figure(figsize=(15, 10))
        
        # Original image with boxes
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        
        # Normalize image for visualization
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        ax1.imshow(img_np)
        
        # Draw predicted boxes in red
        if pred_boxes is not None and len(pred_boxes) > 0:
            for box, conf in zip(pred_boxes, confidences):
                x, y, w, h = box.cpu().numpy()
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(x, y-5, f'{conf:.2f}', color='red')
        
        # Draw ground truth box - handle both tensor types
        if gt_boxes is not None:
            if gt_boxes.dim() == 0:  # Handle 0-d tensor
                logger.warning(f"Skipping GT box visualization for batch {batch_idx}, sample {sample_idx} - invalid tensor dimension")
            else:
                box_data = gt_boxes.cpu().numpy()
                if box_data.size == 4:  # Make sure we have x,y,w,h 
                    y, x, h, w = box_data
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                        edgecolor='g', facecolor='none')
                    ax1.add_patch(rect)
        
        ax1.set_title('Detections (Red: Predicted, Green: GT)')
        
        # Similarity heatmap
        ax2 = plt.subplot2grid((2, 2), (0, 1))

        # 1. Apply temperature scaling and softmax
        temperature = 0.07  # From the paper
        sim_np = similarity_map.cpu()
        sim_np = torch.nn.functional.softmax(sim_np.flatten() / temperature, dim=0)
        sim_np = sim_np.reshape(similarity_map.shape).numpy()
        
        # 2. Create overlay visualization
        ax2.imshow(img_np)  # Show original image first
        heatmap = ax2.imshow(sim_np, cmap='viridis', alpha=0.7)  # Overlay heatmap
        plt.colorbar(heatmap, ax=ax2)
        ax2.set_title('Similarity Heatmap (Overlaid)')
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, f'eval_batch{batch_idx}_sample{sample_idx}.png')
        plt.savefig(save_path)
        plt.close()

        # # Normalize similarity map using softmax
        # similarity_softmax = torch.nn.functional.softmax(similarity_map.flatten() / temperature).reshape(similarity_map.shape)

        # sim_np = similarity_map.cpu().numpy()
        # # heatmap = ax2.imshow(sim_np, cmap='viridis')
        # # plt.colorbar(heatmap, ax=ax2)
        # # ax2.set_title('Similarity Heatmap')
        # # Create a blended visualization
        # alpha = 0.7  # Transparency factor
        # heatmap = ax2.imshow(img_np)  # Show original image
        # overlay = ax2.imshow(sim_np, cmap='viridis', alpha=alpha)  # Overlay heatmap
        # plt.colorbar(overlay, ax=ax2)
        # ax2.set_title('Similarity Heatmap (Overlaid)')
        
        # plt.tight_layout()
        # save_path = os.path.join(self.viz_dir, f'eval_batch{batch_idx}_sample{sample_idx}.png')
        # plt.savefig(save_path)
        # plt.close()
    

    def save_batch_grid(self, images, similarity_maps, pred_boxes, batch_idx):
        """Create grid visualization for entire batch"""
        n = len(images)
        fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
        
        for i in range(n):
            # Image with boxes
            img = images[i].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(img)
            
            if pred_boxes is not None and i < len(pred_boxes):
                boxes = pred_boxes[i]
                if not isinstance(boxes, float) and len(boxes) > 0:
                    for box in boxes:
                        x, y, w, h = box.cpu().numpy()
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                              edgecolor='r', facecolor='none')
                        axes[0, i].add_patch(rect)
            
            axes[0, i].set_title(f'Sample {i}')
            
            # Similarity map
            sim_map = similarity_maps[i].cpu().numpy()
            im = axes[1, i].imshow(sim_map, cmap='viridis')
            plt.colorbar(im, ax=axes[1, i])
            
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, f'batch_grid_{batch_idx}.png')
        plt.savefig(save_path)
        plt.close()

    def save_confidence_distribution(self, confidences, batch_idx):
        """Visualize distribution of detection confidences"""
        plt.figure(figsize=(10, 5))
        plt.hist(confidences.cpu().numpy(), bins=50)
        plt.title('Distribution of Detection Confidences')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        save_path = os.path.join(self.viz_dir, f'conf_dist_batch{batch_idx}.png')
        plt.savefig(save_path)
        plt.close()

    def save_metrics_history(self, metrics_history, save_name='metrics_history.png'):
        """Plot metrics over evaluation progress"""
        plt.figure(figsize=(10, 5))
        
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
            
        plt.title('Metrics History')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        
        save_path = os.path.join(self.viz_dir, save_name)
        plt.savefig(save_path)
        plt.close()
