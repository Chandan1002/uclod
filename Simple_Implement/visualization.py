import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from dataloader import build_unsupervised_train_loader, save_image


class EvalVisualization:
    def __init__(self, cfg):
        self.cfg = cfg
        self.viz_dir = os.path.join(cfg["OUTPUT"]["DIR"], "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

    def save_detection_visualization(
        self,
        image,
        similarity_map,
        pred_boxes,
        gt_boxes,
        confidences,
        metrics,
        batch_idx,
        sample_idx,
    ):
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
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor="r", facecolor="none"
                )
                ax1.add_patch(rect)
                ax1.text(x, y - 5, f"{conf:.2f}", color="red")

        # Draw ground truth box - handle both tensor types
        if gt_boxes is not None:
            if gt_boxes.dim() == 0:  # Handle 0-d tensor
                logger.warning(
                    f"Skipping GT box visualization for batch {batch_idx}, sample {sample_idx} - invalid tensor dimension"
                )
            else:
                box_data = gt_boxes.cpu().numpy()
                if box_data.size == 4:  # Make sure we have x,y,w,h
                    y, x, h, w = box_data
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=2, edgecolor="g", facecolor="none"
                    )
                    ax1.add_patch(rect)

        ax1.set_title("Detections (Red: Predicted, Green: GT)")

        # Similarity heatmap
        ax2 = plt.subplot2grid((2, 2), (0, 1))

        # 1. Apply temperature scaling and softmax
        temperature = 0.07  # From the paper
        sim_np = similarity_map.cpu()
        sim_np = torch.nn.functional.softmax(sim_np.flatten() / temperature, dim=0)
        sim_np = sim_np.reshape(similarity_map.shape).numpy()

        # 2. Create overlay visualization
        ax2.imshow(img_np)  # Show original image first
        heatmap = ax2.imshow(sim_np, cmap="viridis", alpha=0.7)  # Overlay heatmap
        plt.colorbar(heatmap, ax=ax2)
        ax2.set_title("Similarity Heatmap (Overlaid)")

        plt.tight_layout()
        save_path = os.path.join(
            self.viz_dir, f"eval_batch{batch_idx}_sample{sample_idx}.png"
        )
        plt.savefig(save_path)
        plt.close()

    def save_batch_grid(self, images, similarity_maps, pred_boxes, batch_idx):
        """Create grid visualization for entire batch"""
        n = len(images)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

        for i in range(n):
            # Image with boxes
            img = images[i].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(img)

            if pred_boxes is not None and i < len(pred_boxes):
                boxes = pred_boxes[i]
                if not isinstance(boxes, float) and len(boxes) > 0:
                    for box in boxes:
                        x, y, w, h = box.cpu().numpy()
                        rect = patches.Rectangle(
                            (x, y), w, h, linewidth=2, edgecolor="r", facecolor="none"
                        )
                        axes[0, i].add_patch(rect)

            axes[0, i].set_title(f"Sample {i}")

            # Similarity map
            sim_map = similarity_maps[i].cpu().numpy()
            im = axes[1, i].imshow(sim_map, cmap="viridis")
            plt.colorbar(im, ax=axes[1, i])

        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, f"batch_grid_{batch_idx}.png")
        plt.savefig(save_path)
        plt.close()

    def save_image(self, image, anchors, batch_idx, name="", index=0):
        save_path = os.path.join(self.viz_dir, f"{name}_batch_{batch_idx}_sample_{index}.png")
        save_image(
            image,
            name=save_path,
            anchors=anchors,
            save=True,
        )

    def save_confidence_distribution(self, confidences, batch_idx):
        """Visualize distribution of detection confidences"""
        plt.figure(figsize=(10, 5))
        plt.hist(confidences.cpu().numpy(), bins=50)
        plt.title("Distribution of Detection Confidences")
        plt.xlabel("Confidence")
        plt.ylabel("Count")

        save_path = os.path.join(self.viz_dir, f"conf_dist_batch{batch_idx}.png")
        plt.savefig(save_path)
        plt.close()

    def save_metrics_history(self, metrics_history, save_name="metrics_history.png"):
        """Plot metrics over evaluation progress"""
        plt.figure(figsize=(10, 5))

        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)

        plt.title("Metrics History")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()

        save_path = os.path.join(self.viz_dir, save_name)
        plt.savefig(save_path)
        plt.close()

    def generate_heatmap(
        self, index, x_j, z_i, features, fpn_layer, name=None, anchor=None, batch_idx=0
    ):
        save_path = os.path.join(self.viz_dir, f"{name}_heatmap_batch_{batch_idx}_sample_{index}.png")
        image = save_image(
            x_j[index].cpu().permute(1, 2, 0).numpy().copy(), save=False, anchors=anchor
        )
        # Reshaping and normalizing
        positive = z_i[index]
        # if isinstance(features, list):
        #     negative = features[fpn_layer][index]
        # else:
        #     negative = features[index, :, :]
        # NOTE: combined layers
        negative = self.combine_layers(features, index)

        normalized_vector1 = F.normalize(positive, p=2, dim=-1)
        grid = np.zeros((x_j[index].shape[1], x_j[index].shape[2]), dtype=np.float32)
        length = np.zeros((x_j[index].shape[1], x_j[index].shape[2]), dtype=np.float32)

        if isinstance(features, list):
            num_cells_x = features[fpn_layer].shape[-1]
            num_cells_y = features[fpn_layer].shape[-2]
        else:
            num_cells_x = features.shape[-1]
            num_cells_y = features.shape[-2]

        height, width = x_j.shape[2:]
        grid_size_x = torch.div(width, num_cells_x, rounding_mode="floor")
        grid_size_y = torch.div(height, num_cells_y, rounding_mode="floor")

        # Calculate centers dynamically
        centers = [
            (int((r + 0.5) * grid_size_y), int((c + 0.5) * grid_size_x))
            for r in range(num_cells_y)
            for c in range(num_cells_x)
        ]

        for i, grid_indx in enumerate(centers):
            r = torch.div(i, num_cells_x, rounding_mode="floor")
            c = i % num_cells_x
            normalized_vector2 = F.normalize(negative[:, r, c], p=2, dim=-1)
            dot_product = torch.dot(normalized_vector1, normalized_vector2)
            cosine_similarity = dot_product.item()

            y_slice = slice(
                grid_indx[0] - int(grid_size_y / 2), grid_indx[0] + int(grid_size_y / 2)
            )
            x_slice = slice(
                grid_indx[1] - int(grid_size_x / 2), grid_indx[1] + int(grid_size_x / 2)
            )

            grid[y_slice, x_slice] += cosine_similarity
            length[y_slice, x_slice] += 1.0

        length[length == 0] = 1
        grid = grid / length

        min_n, max_n = np.min(grid), np.max(grid)
        # Normalize the grid to [0, 255]
        # grid = (grid - (-1.0)) * (255.0) / (1.0 - (-1.0))
        grid = (grid - (min_n)) * (255.0) / (max_n - (min_n))
        grid = grid.astype(np.uint8)
        min_n, max_n = np.min(grid), np.max(grid)
        # NOTE: FLIPING VALUES
        grid = (max_n + min_n) - grid
        min_n, max_n = np.min(grid), np.max(grid)

        # Apply Gaussian blur and generate heatmap
        blur = cv2.GaussianBlur(grid, (13, 13), 11)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)

        # Save heatmap image
        if name is None:
            output_heatmap = save_image(
                super_imposed_img, name=f"heat_img{index}", normalize=False
            )
        else:
            print('saving heatmap')
            output_heatmap = save_image(
                super_imposed_img, name=save_path, normalize=False
            )

        return output_heatmap, grid

    def combine_layers(self, features, index=0):
        common_size = features[0].shape[-2:]  # Adjust this to your desired common size

        # Resize tensors to the common spatial size using bilinear interpolation
        resize = lambda x: F.interpolate(
            x, size=common_size, mode="bilinear", align_corners=False
        )
        tensor2_resized = resize(features[1])[index]
        tensor3_resized = resize(features[2])[index]
        tensor4_resized = resize(features[3])[index]
        tensor5_resized = resize(features[4])[index]

        # Combine resized tensors using element-wise summation
        combined_tensor = (
            features[0][index]
            + tensor2_resized
            + tensor3_resized
            + tensor4_resized
            + tensor5_resized
        )
        return combined_tensor
