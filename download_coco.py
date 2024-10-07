import os
import requests
from zipfile import ZipFile

def download_file(url, dest_path):
    """Download a file from a URL and save it locally."""
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
    print(f"Downloaded {dest_path}")

def extract_zip(file_path, extract_to):
    """Extract a zip file to a directory."""
    print(f"Extracting {file_path} to {extract_to} ...")
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_path}")

def download_and_extract_coco(dest_dir):
    """Download and extract the COCO 2017 dataset (train images and annotations)."""
    
    # COCO URLs for images and annotations
    coco_urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }

    # Paths for downloaded files
    train_images_zip = os.path.join(dest_dir, 'train2017.zip')
    annotations_zip = os.path.join(dest_dir, 'annotations_trainval2017.zip')

    # Download the train images and annotations
    download_file(coco_urls['train_images'], train_images_zip)
    download_file(coco_urls['annotations'], annotations_zip)

    # Extract the downloaded files
    extract_zip(train_images_zip, dest_dir)
    extract_zip(annotations_zip, dest_dir)

    print("COCO dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    # Specify the directory where the COCO dataset should be downloaded and extracted
    coco_dir = '/media/ck/data/coco'  # Update this path to the directory where you want the data

    # Create the directory if it doesn't exist
    os.makedirs(coco_dir, exist_ok=True)

    # Download and extract the COCO dataset
    download_and_extract_coco(coco_dir)

