import detectron2
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, register_pascal_voc
from detectron2.utils.logger import setup_logger
from torchvision import models
import torch
import torch.nn as nn
import os
import yaml

# Import the dataloader from the new dataloader.py file
from dataloader import get_coco_dataloader, get_pascal_voc_dataloader

# Set up the logger
setup_logger()

print("Printing from detectron folder")

# Function to load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Custom trainer with gradient accumulation
class GradientAccumulationTrainer(DefaultTrainer):
    def __init__(self, cfg, accumulation_steps):
        super().__init__(cfg)
        self.accumulation_steps = accumulation_steps
        self.grad_step = 0
        self._data_loader_iter = iter(self.data_loader)
        self.grad_scaler = torch.cuda.amp.GradScaler()  # Initialize gradient scaler for mixed precision

    def run_step(self):
        assert self.model.training, "Model is not in training mode!"

        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        
        with torch.amp.autocast('cuda'):  # Enable mixed precision
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            losses /= self.accumulation_steps

        self.grad_scaler.scale(losses).backward()

        # Accumulate gradients and update the optimizer every accumulation_steps
        self.grad_step += 1
        if self.grad_step % self.accumulation_steps == 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

# Define ResNet-50 Backbone
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.resnet(x)

# Define RetinaNet Backbone
class RetinaNetBackbone(nn.Module):
    def __init__(self):
        super(RetinaNetBackbone, self).__init__()
        # Using Detectron2 model zoo to load RetinaNet backbone
        from detectron2.model_zoo import get_config
        from detectron2.modeling import build_model

        cfg = get_cfg()
        cfg.merge_from_file("/media/ck/Project1TB/clod/detectron/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_50_FPN_1x/137849486/model_final.pth"
        self.retinanet = build_model(cfg)
        self.retinanet.eval()

    def forward(self, x):
        return self.retinanet.backbone(x)

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

# Define the main Contrastive Learning Model
class ContrastiveLearningModel(nn.Module):
    def __init__(self):
        super(ContrastiveLearningModel, self).__init__()
        self.resnet_backbone = ResNetBackbone()
        self.retinanet_backbone = RetinaNetBackbone()
        self.projection_head = ProjectionHead(input_dim=2048)  # ResNet-50 has 2048 features
        self.loss_fn = NTXentLossWithAnchor()

    def forward(self, xi, xj, negatives):
        # Pipeline 1: Process cropped image xi through ResNet-50
        ri = self.resnet_backbone(xi)
        zi = self.projection_head(ri)

        # Pipeline 2: Process full image xj through RetinaNet Backbone
        rj = self.retinanet_backbone(xj)
        zj = self.projection_head(rj)

        # Process negatives: extract features from the negatives
        negative_features = []
        for negative in negatives:
            r_neg = self.resnet_backbone(negative)
            z_neg = self.projection_head(r_neg)
            negative_features.append(z_neg)
        anchor_negative = torch.stack(negative_features)

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

# Main function to set up the Detectron2 configuration and start training
def main():
    config = load_config('config.yaml')
    print(config['batch_size'])

    # Load dataset based on config setting
    if config['dataset'] == 'coco':
        train_loader = get_coco_dataloader(config['data_dir'], config['batch_size'], config['num_workers'])
    elif config['dataset'] == 'pascal_voc':
        train_loader = get_pascal_voc_dataloader(config['data_dir'], config['batch_size'], config['num_workers'])
    else:
        raise ValueError("Unsupported dataset. Please choose either 'coco' or 'pascal_voc'.")
    
    # Set up the Detectron2 configuration
    cfg = get_cfg()
    cfg.merge_from_file(config['detectron2_cfg_file'])
    cfg.DATASETS.TRAIN = ("coco_train",) if config['dataset'] == 'coco' else ("voc_train",)
    cfg.SOLVER.IMS_PER_BATCH = config['batch_size']
    cfg.SOLVER.BASE_LR = config['lr']
    cfg.SOLVER.MAX_ITER = config['max_iter']
    cfg.MODEL.WEIGHTS = config['pretrained_weights']  # Path to pretrained weights (if any)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize model
    model = ContrastiveLearningModel()

    # Set up the custom trainer with gradient accumulation and start training
    accumulation_steps = config.get('accumulation_steps', 1)
    trainer = GradientAccumulationTrainer(cfg, accumulation_steps=accumulation_steps)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()  # <-- Ensure the main function is called
