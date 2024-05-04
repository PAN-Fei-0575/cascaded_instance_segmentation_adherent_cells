import json
import os
import shutil
import cv2
import imageio
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
import argparse
from math import radians, fabs, sin, cos


def rename_image(filename):
    base_name, extension = os.path.splitext(filename)
    if extension.lower() == '.jpg':
        new_extension = '.png'
    else:
        new_extension = extension
    new_filename = f"{base_name}R{new_extension}"
    return new_filename

def load_image(image_path):
    try:
        image = imageio.imread(image_path)
        return image
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Unable to read image '{image_path}': {e}")
        return None

def save_image(image_path, image_data):
    if not os.path.exists(image_path):
        imageio.imwrite(image_path, image_data)
        print(f"Image saved at '{image_path}'.")
    else:
        print(f"Image '{image_path}' already exists. Skipping...")

def rotate_image(data, degree, is_mask=False):
    """
    Function to rotate an image or mask data.

    Args:
        data (numpy.ndarray): Image or mask data.
        degree (float): Rotation angle in degrees.
        is_mask (bool): Whether the data is a mask.

    Returns:
        numpy.ndarray: Rotated image or mask data.
    """
    # Get height and width of input data
    height, width = data.shape[:2]  
    # Calculate new height and width after rotation
    new_height = round(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  
    new_width = round(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)  
    # Adjust rotation matrix for new width and height
    rotation_matrix[0, 2] += (new_width - width) // 2  
    rotation_matrix[1, 2] += (new_height - height) // 2  

    if is_mask:  # Check if data is a mask
        rotated_data = cv2.warpAffine(data, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0))  # Rotate mask data
    else:
        rotated_data = cv2.warpAffine(data, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255, 255))  # Rotate image data

    return rotated_data  # Return rotated data

def process_dataset(src_ann_file, src_img_dir, tar_ann_file, tar_img_dir):
    os.makedirs(os.path.join(tar_img_dir, "images"), exist_ok=True)

    src_coco_gt = COCO(src_ann_file)
    src_img_ids_all = sorted(src_coco_gt.getImgIds())
    src_ann_ids_all = sorted(src_coco_gt.getAnnIds(src_img_ids_all))

    with open(src_ann_file, 'r') as json_file:
        data = json.load(json_file)

    for item in range(len(src_img_ids_all)):
        src_img_id = src_img_ids_all[item]
        src_img_info = src_coco_gt.loadImgs(src_img_id)[0]
        src_img_name = os.path.basename(src_img_info['file_name'])
        src_img_path = os.path.join(src_img_dir, src_img_name)
        tar_img_path = os.path.join(tar_img_dir, 'images', src_img_name)
        shutil.copy(src_img_path, tar_img_path)
        data['images'][item]['file_name'] = src_img_name

        src_img = load_image(src_img_path)
        rotated_img = rotate_image(src_img, 45)
        height, width = rotated_img.shape[:2]
        rotated_img_name = rename_image(src_img_name)
        rotated_img_save_path = os.path.join(tar_img_dir, "images", rotated_img_name)
        save_image(rotated_img_save_path, rotated_img)

        rotated_img_info = {
            "id": src_img_ids_all[-1] + src_img_id + 1,
            "file_name": rotated_img_name,
            "height": height,
            "width": width
        }
        data['images'].append(rotated_img_info)

        src_ann_ids = sorted(src_coco_gt.getAnnIds(imgIds=src_img_id))
        src_anns = src_coco_gt.loadAnns(src_ann_ids)

        for j in range(len(src_ann_ids)):
            src_ann_id = src_ann_ids[j]
            src_ann = src_anns[j]
            src_mask_data = src_coco_gt.annToMask(src_ann)
            rotated_mask = rotate_image(src_mask_data, 45, True)
            rotated_mask_fortran_binary = np.asfortranarray(rotated_mask.astype('uint8'))
            rotated_mask_fortran_binary_encoded = mask.encode(rotated_mask_fortran_binary)
            rotated_mask_fortran_binary_encoded['counts'] = rotated_mask_fortran_binary_encoded['counts'].decode('utf-8')
            area = mask.area(rotated_mask_fortran_binary_encoded)
            bounding_box = mask.toBbox(rotated_mask_fortran_binary_encoded)

            rotated_ann = {
                "id": src_ann_ids_all[-1] + src_ann_id + 1,
                "image_id": src_img_ids_all[-1] + src_img_id + 1,
                "category_id": src_ann['category_id'],
                "bbox": bounding_box.tolist(),
                "segmentation": rotated_mask_fortran_binary_encoded,
                "area": area.tolist(),
                "iscrowd": 0
            }
            data['annotations'].append(rotated_ann)

    with open(os.path.join(tar_ann_file), 'w') as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser(description='Process dataset by rotating images and masks.')
    parser.add_argument('--dataset', type=str, default='hepg2', help='Name of the cell image dataset.')
    args = parser.parse_args()

    datasets = {
        'hepg2': {
            'dir': r"data/hepg2/coco_style",
            'sets': ["train", "val", "test"],
            'ann': 'annotations_new_non_overlap.json'
        },
        'livecell': {
            'dir': r"data/livecell/coco_style",
            'sets': ["train", "val", "test"],
            'ann': 'annotations.json'
        }
    }

    dataset_info = datasets.get(args.dataset)
    if dataset_info is None:
        print(f"Error: Dataset '{args.dataset}' is not supported.")
        return

    for set_name in dataset_info['sets']:
        src_ann_file = os.path.join(dataset_info['dir'], set_name, dataset_info['ann'])
        src_image_dir = os.path.join(dataset_info['dir'], set_name, 'images')
        tar_ann_file = os.path.join(dataset_info['dir'] + "_rotated", set_name, 'annotations.json')
        tar_image_dir = os.path.join(dataset_info['dir'] + "_rotated", set_name)

        process_dataset(src_ann_file, src_image_dir, tar_ann_file, tar_image_dir)

if __name__ == "__main__":
    main()
