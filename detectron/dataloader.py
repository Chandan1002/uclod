import torch
from torch.utils.data import DataLoader
import detectron2.data.datasets as d2_datasets
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

# Function to get COCO DataLoader
def get_coco_dataloader(data_dir, batch_size, num_workers):
    # Register COCO dataset if not already registered
    train_json = f"{data_dir}/annotations/instances_train2017.json"
    train_images = f"{data_dir}/train2017"
    dataset_name = "coco_train"
    if dataset_name not in DatasetCatalog.list():
        d2_datasets.register_coco_instances(dataset_name, {}, train_json, train_images)
    
    # MetadataCatalog should be called here to register proper data
    MetadataCatalog.get(dataset_name)

    # Build DataLoader for COCO
    data_loader = build_detection_train_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return data_loader, None

# Function to get Pascal VOC DataLoader
def get_pascal_voc_dataloader(data_dir, batch_size, num_workers):
    # Register Pascal VOC dataset if not already registered
    dataset_name = "voc_train"
    if dataset_name not in DatasetCatalog.list():
        d2_datasets.register_pascal_voc(dataset_name, data_dir, year="2007", split="trainval")
    
    # MetadataCatalog should be called here to register proper data
    MetadataCatalog.get(dataset_name)

    # Build DataLoader for Pascal VOC
    data_loader = build_detection_train_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return data_loader, None
