import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import torchvision.transforms as transforms

# Import the dataloader from the new dataloader.py file
from dataloader import get_coco_dataloader
from torchvision import models

# Function to load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize distributed process group
def setup_distributed(rank, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',  # Use NCCL backend for GPU training
            init_method='env://',  # Read environment variables for initialization
            rank=rank,
            world_size=world_size
        )

# Cleanup distributed process group
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# Define ResNet-50 Backbone
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.resnet(x)

# Define RetinaNet Backbone using ResNet-50
class RetinaNetBackbone(nn.Module):
    def __init__(self):
        super(RetinaNetBackbone, self).__init__()
        self.retinanet = models.resnet50(pretrained=True)
        self.retinanet.fc = nn.Identity()  # Remove final classification layer

    def forward(self, x):
        return self.retinanet(x)

# Define projection head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

# NT-Xent Loss with Anchor Negatives
class NTXentLossWithAnchor(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLossWithAnchor, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, zi, zj, anchor_negative):
        # Compute cosine similarity for positive pairs
        sim_pos = self.cosine_similarity(zi, zj) / self.temperature  # Shape: (batch_size,)

        # Flatten anchor_negative to make it a 2D matrix
        anchor_negative = anchor_negative.view(-1, anchor_negative.size(-1))  # Shape: (num_negatives, feature_dim)

        # Compute cosine similarity for negative pairs using matrix multiplication
        sim_neg = torch.mm(zi, anchor_negative.T) / self.temperature  # Shape: (batch_size, num_negatives)

        # Concatenate positive and negative similarities
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # Shape: (batch_size, 1 + num_negatives)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(zi.device)  # Positive at index 0

        # Compute the cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss



# Define the main Unsupervised Object Detection model
class UnsupervisedObjectDetectionModel(nn.Module):
    def __init__(self):
        super(UnsupervisedObjectDetectionModel, self).__init__()
        self.resnet_backbone = ResNetBackbone()
        self.retinanet_backbone = RetinaNetBackbone()
        self.projection_head = ProjectionHead(input_dim=2048)  # ResNet-50 has 2048 features
        self.loss_fn = NTXentLossWithAnchor()

    def forward(self, xi, xj, negatives):
        # Pipeline 1: Process cropped image xi through ResNet-50
        ri = self.resnet_backbone(xi)
        zi = self.projection_head(ri)

        # Pipeline 2: Process full image xj through ResNet-50 as RetinaNet Backbone
        rj = self.retinanet_backbone(xj)
        zj = self.projection_head(rj)

        # Process negatives: extract features from the negatives
        negative_features = []
        for negative in negatives:
            r_neg = self.resnet_backbone(negative)
            z_neg = self.projection_head(r_neg)
            negative_features.append(z_neg)
        anchor_negative = torch.stack(negative_features)

        # Compute loss using positive pair (zi, zj) and the extracted negative features
        loss = self.loss_fn(zi, zj, anchor_negative)
        return loss

# Training function with Distributed Data Parallel and Gradient Accumulation
def train_model(model, dataloader, optimizer, device, epochs, final_batch_size, mini_batch_size, rank, world_size):
    model.train()
    accumulation_steps = final_batch_size // mini_batch_size
    print(f"Mini-batch size (before gradient accumulation): {mini_batch_size}")
    print(f"Final batch size (after gradient accumulation): {final_batch_size}")
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()  # Initialize gradients to zero at the start of each epoch
        for i, (images, targets) in enumerate(dataloader):
            # Stack images into a batch and move them to the device
            images = torch.stack(images).to(device)

            # Use images directly (augmentations are already applied in the DataLoader)
            xi = images
            xj = images

            # Generate negatives (random other images or cropped locations)
            negatives = torch.stack([images for _ in range(10)], dim=0)

            # Compute the loss
            loss = model(xi, xj, negatives)
            loss = loss / accumulation_steps  # Scale the loss by accumulation steps

            # Backpropagation
            loss.backward()

            # Accumulate gradients and update the optimizer every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # Update the model parameters
                optimizer.zero_grad()  # Reset gradients to zero after updating

            running_loss += loss.item() * accumulation_steps

            if (i + 1) % (accumulation_steps * 10) == 0:  # Print loss every 10 accumulated steps
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/(i+1):.4f}")

        # Update the sampler for distributed training (if any)
        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)


# Main function for distributed training
def main():
    config = load_config('config.yaml')
    print(config['batch_size'])

    # Get the distributed configuration from environment variables or set defaults for single GPU
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set up the distributed process group if needed
    setup_distributed(rank, world_size)

    # Set the device based on the local rank (for multi-GPU per node)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Load COCO dataset
    train_loader, train_sampler = get_coco_dataloader(config['data_dir'], config['batch_size'], config['num_workers'], rank, world_size)

    # Initialize model
    model = UnsupervisedObjectDetectionModel().to(device)

    # Wrap the model with Distributed Data Parallel (DDP) only for multi-GPU setup
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Start training with gradient accumulation
    train_model(
        model, train_loader, optimizer, device, 
        epochs=config['epochs'], 
        final_batch_size=config['final_batch_size'], 
        mini_batch_size=config['batch_size'], 
        rank=rank, world_size=world_size
    )

    # Clean up the distributed process group if needed
    cleanup_distributed()

if __name__ == "__main__":
    main()  # <-- Ensure the main function is called

