import random

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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

# Step 1: Setup COCO dataset
transform = T.Compose(
    [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

train_dataset = CocoDetection(
    root="/mnt/drive_test/coco/train2017",
    annFile="/mnt/drive_test/coco/annotations/instances_train2017.json",
    transform=transform,
)

val_dataset = CocoDetection(
    root="/mnt/drive_test/coco/val2017",
    annFile="/mnt/drive_test/coco/annotations/instances_val2017.json",
    transform=transform,
)

# Reduce training dataset to 10%
train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)
subset_size = int(0.1 * len(train_indices))
train_subset = Subset(train_dataset, train_indices[:subset_size])

train_loader = DataLoader(
    train_subset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)),
)

# Step 2: Load pre-trained Faster R-CNN with FPN model
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Replace the classification head to match the number of COCO classes
num_classes = 91  # 80 classes + background + other special classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
)

# Load the pre-trained weights
pretrained_weights = torch.load("./output/20241225_212615/model_final.pth")
model.load_state_dict(pretrained_weights, strict=False)

# Step 3: Define the optimizer and learning rate scheduler
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Step 4: Training loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = move_to_device(obj=targets, device=device)

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    lr_scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


# Step 5: Evaluation
def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            # Perform evaluation (e.g., calculate mAP)


evaluate(model, val_loader, device)

