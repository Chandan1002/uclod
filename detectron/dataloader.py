import torch
from torch.utils.data import DataLoader
import detectron2.data.datasets as d2_datasets
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

# Function to get COCO DataLoader
def get_coco_dataloader(data_dir, batch_size, num_workers):
    # Register COCO dataset if not already registered
    train_json = f"{data_dir}/annotations/instances_train2017.json"import torch
from torch.utils.data import DataLoader
import detectron2.data.datasets as d2_datasets
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog, detection_utils
from detectron2.data import transforms as T
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
import copy

# Custom mapper to ensure crop is inside the full image
def custom_mapper(dataset_dict):
    # Make a copy so that the original dataset_dict is not modified
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")

    # Define transformations including a crop that stays within image bounds
    transform_list = [
        T.RandomCrop("relative_range", (0.5, 0.5)),  # Crop size as a fraction of the original image
        T.Resize((800, 800)),  # Resize to a fixed size
        T.RandomFlip(),  # Example flip
    ]

    # Apply transformations
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # Transform annotations as well
    annos = [
        detection_utils.transform_instance_annotations(annotation, transforms, image.shape[:2])
        for annotation in dataset_dict.pop("annotations")
    ]
    dataset_dict["annotations"] = annos

    return dataset_dict

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
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)
    return data_loader

# Function to get Pascal VOC DataLoader
def get_pascal_voc_dataloader(data_dir, batch_size, num_workers):
    # Register Pascal VOC dataset if not already registered
    dataset_name = "voc_train"
    if dataset_name not in DatasetCatalog.list():
        d2_datasets.register_pascal_voc(dataset_name, data_dir, year="2007", split="trainval")
    
    # MetadataCatalog should be called here to register proper data
    MetadataCatalog.get(dataset_name)

    # Build DataLoader for Pascal VOC
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)
    return data_loader

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
