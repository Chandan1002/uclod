import os
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# Define COCO DataLoader with DistributedSampler
def get_coco_dataloader(data_dir, batch_size, num_workers, rank, world_size):
    # Define augmentations (apply to images before they are converted to tensors)
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.Resize((224, 224)),  # Resize the images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    coco_train = datasets.CocoDetection(
        root=os.path.join(data_dir, 'train2017'),
        annFile=os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
        transform=augmentations  # Apply augmentations directly during data loading
    )

    if world_size > 1:
        train_sampler = DistributedSampler(coco_train, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None  # No need for DistributedSampler in single-GPU mode

    train_loader = DataLoader(
        coco_train,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))  # Collate function to handle variable-sized annotations
    )

    return train_loader, train_sampler

