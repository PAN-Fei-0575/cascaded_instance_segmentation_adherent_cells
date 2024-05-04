import json
import os
import random
import shutil
from argparse import ArgumentParser

import cv2
import imageio.v2
import numpy as np
import skimage.measure
from pycocotools import mask as pycoco_mask
from pycocotools.coco import COCO


def is_grayscale(img):
    if len(img.shape) == 2:
        return True  # Grayscale
    elif len(img.shape) == 3 and img.shape[2] == 3:
        return False  # RGB
    else:
        raise ValueError("Unsupported img format")

def load_img(img_path):
    try:
        img = imageio.v2.imread(img_path)
        return img
    except FileNotFoundError:
        print(f"Error: File '{img_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Unable to read img '{img_path}': {e}")
        return None


def save_img(img_path, img_data):
    if not os.path.exists(img_path):
        imageio.v2.imsave(img_path, img_data)
        print(f"Image saved at '{img_path}'.")
    else:
        print(f"Image '{img_path}' already exists. Skipping...")


def calculate_cropping_bbox(height=864, width=1152):
    red_img = np.zeros((height, width, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255
    rotated_red_img = rotate_img(red_img, 45)
    restore_red_img = rotate_img(rotated_red_img, -45)
    gray_img = np.dot(restore_red_img[..., :3], [0.2989, 0.5870, 0.1140])
    thresholded = gray_img < 200  # Using a threshold of 200 (adjust as needed)
    non_white_indices = np.argwhere(thresholded)
    min_row, min_col = np.min(non_white_indices, axis=0)
    max_row, max_col = np.max(non_white_indices, axis=0)
    return min_row, min_col, max_row, max_col


def calculate_iou(ground_truth, prediction):
    intersection = np.sum(np.logical_and(ground_truth > 0, prediction > 0))
    union = np.sum(np.logical_or(ground_truth > 0, prediction > 0))
    return np.sum(intersection) / np.sum(union)


def rotate_img(img, degree, is_mask=False):
    height, width = img.shape[:2]
    new_height = round(width * np.abs(np.sin(np.radians(degree))) + height * np.abs(np.cos(np.radians(degree))))
    new_width = round(height * np.abs(np.sin(np.radians(degree))) + width * np.abs(np.cos(np.radians(degree))))
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    rotation_matrix[0, 2] += (new_width - width) // 2
    rotation_matrix[1, 2] += (new_height - height) // 2
    if is_mask:
        rotated_data = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0))
    else:
        rotated_data = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255, 255))
    return rotated_data


def create_file_dict(root_folder):
    file_dict = {}
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            file_list = os.listdir(subfolder_path)
            file_dict[subfolder] = file_list
    return file_dict


def split_files(src_folder, dest_folder_A, dest_folder_B, dest_folder_C, split_ratios=(0.6, 0.2, 0.2)):
    file_list = os.listdir(src_folder)
    random.shuffle(file_list)

    total_files = len(file_list)
    split_indices = [
        round(total_files * split_ratios[0]),
        round(total_files * (split_ratios[0] + split_ratios[1]))
    ]

    files_to_copy_A = file_list[:split_indices[0]]
    files_to_copy_B = file_list[split_indices[0]:split_indices[1]]
    files_to_copy_C = file_list[split_indices[1]:]

    for file in files_to_copy_A:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(dest_folder_A, src_folder.split(os.sep)[-1] + '_' + file)
        shutil.copy(src_path, dest_path)

    for file in files_to_copy_B:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(dest_folder_B, src_folder.split(os.sep)[-1] + '_' + file)
        shutil.copy(src_path, dest_path)

    for file in files_to_copy_C:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(dest_folder_C, src_folder.split(os.sep)[-1] + '_' + file)
        shutil.copy(src_path, dest_path)


def generate_txt(root_folder):
    class_map = {'class1': '0', 'class2': '1'}

    # Create the 'meta' folder if it doesn't exist
    meta_folder = os.path.join(root_folder, 'meta')
    os.makedirs(meta_folder, exist_ok=True)

    for set_folder in ['train', 'val', 'test']:
        merged_txt = os.path.join(meta_folder, f'{set_folder}.txt')

        with open(merged_txt, 'w') as merged_file:
            for class_folder in ['class1', 'class2']:
                img_dir = os.path.join(root_folder, set_folder, class_folder)
                class_label = class_map[class_folder]

                for img_file in os.listdir(img_dir):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(class_folder, img_file)
                        merged_file.write(f"{img_path} {class_label}\n")


def generate_classification_dataset(content, config_dir):
    # Splitting the content by lines
    lines = content.split('\n')

    # Function to insert the function parameter into a path
    def insert_into_path(path):
        parts = path.split('/')
        index = parts.index('hepg2') + 1  # Find the index after 'hepg2'
        parts.insert(index, config_dir)  # Insert the function name
        return '/'.join(parts)

    # Modifying paths in the content
    for i, line in enumerate(lines):
        if 'data_prefix' in line or 'ann_file' in line:
            start = line.find("r'") + 2
            end = line.find("'", start)
            path = line[start:end]
            new_path = insert_into_path(path)
            lines[i] = line[:start] + new_path + line[end:]

    # Joining the lines back into content
    return '\n'.join(lines)


# Original template
template = """# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


dataset_type = 'HepG2'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train = dict(
        type='ClassBalancedDataset',
        oversample_thr=0.5,
        dataset=dict(
            type=dataset_type,
            data_prefix=r'data/hepg2/val/train/',
            ann_file=r'data/hepg2/val/meta/train.txt',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_prefix=r'data/hepg2/val/val/',
        ann_file=r'data/hepg2/val/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=r'data/hepg2/val/test/',
        ann_file=r'data/hepg2/val/meta/test.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')"""

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hepg2', help='Name of the cell image dataset.')
    parser.add_argument('--set', default='val', help='Choose the set to be processed')
    parser.add_argument('--config_dir', default='mask_rcnn_r50_caffe_fpn_1x_hepg2',
                        help='Choose the set to be processed')
    parser.add_argument('--img_height',  type=int,  default=864)
    parser.add_argument('--img_width',  type=int,  default=1152)
    args = parser.parse_args()

    mmdet_root_dir = r'G:\dvsisn_hepg2\mmdetection-2.28.2'
    mmdet_dataset_dir = os.path.join(mmdet_root_dir, 'data', args.dataset, 'coco_style', args.set, 'images')
    coco_gt_path = os.path.join(mmdet_root_dir, 'data', args.dataset, 'coco_style_rotated', args.set, 'annotations.json')
    inference_result_coco_dt_path = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set,
                                                 'results.segm.json')
    mask_preds_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set, 'mask_prediction')
        
    coco_gt = COCO(coco_gt_path)
    img_ids = sorted(coco_gt.getImgIds())
    min_row, min_col, max_row, max_col = calculate_cropping_bbox(args.img_height, args.img_width)

    with open(inference_result_coco_dt_path) as json_file:
        inference_results = json.load(json_file)
        
    num_preds = len(inference_results)
    img_id_to_name = {}
    img_id_to_name_digit = {}
        
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_id_to_name[img_id] = img_name
        img_name_digit = os.path.splitext(img_name)[0]
        img_id_to_name_digit[img_id] = img_name_digit

    img_name_digit_to_id = {value: key for key, value in img_id_to_name_digit.items()}


    if args.set == 'val':
        masks_kept_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set, 'masks_kept')
        masked_imgs_kept_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set,
                                              'masked_img_kept')
        masks_dropped_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set, 'masks_dropped')
        masked_img_dropped_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set,
                                                'masked_img_dropped')

        unrotated_mask_count = 0
        rotated_mask_count = 0

        for idx in range(num_preds):
            img_id = inference_results[idx]['image_id']
            img_name_digit = img_id_to_name_digit.get(img_id, "")
            mask_pred_folder = os.path.join(mask_preds_dir, img_name_digit)
            os.makedirs(mask_pred_folder, exist_ok=True)
            mask_pred = pycoco_mask.decode(inference_results[idx]['segmentation'])
            labeled_img, mask_count = skimage.measure.label(mask_pred, return_num=True)

            if mask_count == 1:
                mask_pred_copy = (mask_pred * 255).astype(np.uint8)
                mask_pred_save_path = os.path.join(mask_preds_dir, img_name_digit, f"{idx}.png")
                save_img(mask_pred_save_path, mask_pred_copy)

            if img_name_digit[-1] == 'R':
                rotated_mask_count += 1
            else:
                unrotated_mask_count += 1

        for img_id in img_ids:
            img_name = img_id_to_name.get(img_id, "")
            img_name_digit = img_id_to_name_digit.get(img_id, "")
            if img_name_digit[-1] == 'R':
                continue
            unrotated_mask_pred_dir = os.path.join(mask_preds_dir, img_name_digit)
            unrotated_pred_mask_names = sorted(os.listdir(unrotated_mask_pred_dir))
            sorted_unrotated_pred_mask_names = sorted(unrotated_pred_mask_names,
                                                      key=lambda x: int(os.path.splitext(x)[0]))
            unrotated_pred_mask_count = len(sorted_unrotated_pred_mask_names)
            rotated_mask_pred_dir = os.path.join(mask_preds_dir, img_name_digit + 'R')
            rotated_pred_mask_names = os.listdir(rotated_mask_pred_dir)
            sorted_rotated_pred_mask_names = sorted(rotated_pred_mask_names, key=lambda x: int(os.path.splitext(x)[0]))
            rotated_pred_mask_count = len(sorted_rotated_pred_mask_names)
            ann_ids = sorted(coco_gt.getAnnIds(imgIds=img_id))
            anns = coco_gt.loadAnns(ann_ids)
            ann_count = len(anns)

            iou_matrix = np.zeros(shape=(ann_count, unrotated_pred_mask_count + rotated_pred_mask_count))

            for i in range(ann_count):
                gt_mask = coco_gt.annToMask(anns[i])
                for j in range(unrotated_pred_mask_count):
                    unrotated_pred_mask_path = os.path.join(unrotated_mask_pred_dir,
                                                            sorted_unrotated_pred_mask_names[j])
                    unrotated_pred_mask = (load_img(unrotated_pred_mask_path) / 255).astype(np.uint8)
                    iou_matrix[i, j] = calculate_iou(unrotated_pred_mask, gt_mask)

                for k in range(rotated_pred_mask_count):
                    rotated_pred_mask_path = os.path.join(rotated_mask_pred_dir, sorted_rotated_pred_mask_names[k])
                    rotated_pred_mask = load_img(rotated_pred_mask_path)
                    restored_mask = rotate_img(rotated_pred_mask, -45, is_mask=True)
                    cropped_mask = restored_mask[min_row:max_row + 1, min_col:max_col + 1]
                    cropped_mask = (cropped_mask / 255).astype(np.uint8)
                    iou_matrix[i, unrotated_pred_mask_count + k] = calculate_iou(cropped_mask, gt_mask)

            best_match_indices = np.argmax(iou_matrix, axis=1)
            masked_imgs_kept_save_dir = os.path.join(masked_imgs_kept_dir, img_name_digit)
            masks_kept_save_dir = os.path.join(masks_kept_dir, img_name_digit)
            os.makedirs(masked_imgs_kept_save_dir, exist_ok=True)
            os.makedirs(masks_kept_save_dir, exist_ok=True)

            for match_idx in best_match_indices:
                if match_idx < unrotated_pred_mask_count:
                    mask_src_path = os.path.join(unrotated_mask_pred_dir,
                                                    sorted_unrotated_pred_mask_names[match_idx])
                    mask_save_path = os.path.join(masks_kept_save_dir, sorted_unrotated_pred_mask_names[match_idx])
                    shutil.copy(mask_src_path, mask_save_path)
                    raw_img_path = os.path.join(mmdet_dataset_dir, img_name)
                    masked_img_save_path = os.path.join(masked_imgs_kept_save_dir,
                                                          sorted_unrotated_pred_mask_names[match_idx])
                    raw_img = load_img(raw_img_path)
                    mask = load_img(mask_save_path)
                    if is_grayscale(raw_img):
                        raw_img[mask == 0] = 0
                    elif not is_grayscale(raw_img):
                        raw_img[mask == 0] = [0, 0, 0]
                    save_img(masked_img_save_path, raw_img)
                else:
                    mask_src_path = os.path.join(rotated_mask_pred_dir,
                                                    sorted_rotated_pred_mask_names[
                                                        match_idx - unrotated_pred_mask_count])
                    rotated_pred_mask = load_img(mask_src_path)
                    reversed_rotated_mask = rotate_img(rotated_pred_mask, -45, True)
                    cropped_reversed_rotated_mask = reversed_rotated_mask[min_row:max_row + 1, min_col:max_col + 1]
                    non_255_mask = cropped_reversed_rotated_mask < 255
                    filtered_mask = cropped_reversed_rotated_mask.copy()
                    filtered_mask[non_255_mask] = 0
                    mask_save_path = os.path.join(masks_kept_save_dir,
                                                  sorted_rotated_pred_mask_names[match_idx - unrotated_pred_mask_count])
                    save_img(mask_save_path, filtered_mask)
                    raw_img_path = os.path.join(mmdet_dataset_dir, img_name)
                    masked_img_save_path = os.path.join(masked_imgs_kept_save_dir, sorted_rotated_pred_mask_names[
                        match_idx - unrotated_pred_mask_count])
                    raw_img = load_img(raw_img_path)
                    mask = load_img(mask_save_path)
                    if is_grayscale(raw_img):
                        raw_img[mask == 0] = 0
                    elif not is_grayscale(raw_img):
                        raw_img[mask == 0] = [0, 0, 0]
                    save_img(masked_img_save_path, raw_img)

        mask_pred_file_dict = create_file_dict(mask_preds_dir)
        mask_pred_kept_file_dict = create_file_dict(masks_kept_dir)
        mask_removed_file_dict = {}

        for key in mask_pred_kept_file_dict:
            mask_removed_filename_list = list(set(mask_pred_file_dict[key]) - set(mask_pred_kept_file_dict[key]))
            mask_removed_file_dict[key] = mask_removed_filename_list
            mask_removed_filename_list = list(set(mask_pred_file_dict[key + 'R']) - set(mask_pred_kept_file_dict[key]))
            mask_removed_file_dict[key + 'R'] = mask_removed_filename_list

        for key in mask_removed_file_dict:
            img_id = img_name_digit_to_id.get(key)
            _ann_ids = sorted(coco_gt.getAnnIds(imgIds=img_id))
            _anns = coco_gt.loadAnns(_ann_ids)
            _ann_count = len(_anns)
            if key[-1] != 'R':
                os.makedirs(os.path.join(masks_dropped_dir, key), exist_ok=True)
                os.makedirs(os.path.join(masked_img_dropped_dir, key), exist_ok=True)
                raw_img_path = os.path.join(mmdet_dataset_dir, key + img_name[-4:])
                for item in mask_removed_file_dict[key]:
                    temp_mask_path = os.path.join(mask_preds_dir, key, item)
                    dst_mask_path = os.path.join(masks_dropped_dir, key, item)
                    masked_img_dropped_path = os.path.join(masked_img_dropped_dir, key, item)
                    shutil.copy(temp_mask_path, dst_mask_path)
                    mask = load_img(temp_mask_path)
                    for i in range(_ann_count):
                        _gt_mask = coco_gt.annToMask(_anns[i])
                        if 0.55 > calculate_iou(_gt_mask, mask) > 0.2:
                            raw_img = load_img(raw_img_path)
                            if is_grayscale(raw_img):
                                raw_img[mask == 0] = 0
                            elif not is_grayscale(raw_img):
                                raw_img[mask == 0] = [0, 0, 0]
                            save_img(masked_img_dropped_path, raw_img)
                            break
            else:
                raw_img_path = os.path.join(mmdet_dataset_dir, key[:-1] + img_name[-4:])
                for item in mask_removed_file_dict[key]:
                    temp_mask_path = os.path.join(mask_preds_dir, key, item)
                    dst_mask_path = os.path.join(masks_dropped_dir, key[:-1], item)
                    masked_img_dropped_path = os.path.join(masked_img_dropped_dir, key[:-1], item)
                    mask = load_img(temp_mask_path)
                    for i in range(_ann_count):
                        _gt_mask = coco_gt.annToMask(_anns[i])
                        if 0.55 > calculate_iou(_gt_mask, mask) > 0.2:
                            reversed_rotated_mask = rotate_img(mask, -45, True)
                            cropped_reversed_rotated_mask = reversed_rotated_mask[min_row:max_row + 1, min_col:max_col + 1]
                            non_255_mask = cropped_reversed_rotated_mask < 255
                            filtered_mask = cropped_reversed_rotated_mask.copy()
                            filtered_mask[non_255_mask] = 0
                            save_img(dst_mask_path, filtered_mask)
                            raw_img = load_img(raw_img_path)
                            mask = load_img(dst_mask_path)
                            if is_grayscale(raw_img):
                                raw_img[mask == 0] = 0
                            elif not is_grayscale(raw_img):
                                raw_img[mask == 0] = [0, 0, 0]
                            save_img(masked_img_dropped_path, raw_img)
                            break

        mmcls_root = r'G:\dvsisn_hepg2\mmpretrain-0.25.0'
        mmcls_dataset_path = os.path.join(mmcls_root, 'data', args.dataset, args.config_dir, args.set)

        mmcls_train_dir = os.path.join(mmcls_dataset_path, 'train')
        mmcls_val_dir = os.path.join(mmcls_dataset_path, 'val')
        mmcls_test_dir = os.path.join(mmcls_dataset_path, 'test')

        mmcls_dataset_folders = [
            os.path.join(mmcls_train_dir, 'class1'),
            os.path.join(mmcls_train_dir, 'class2'),
            os.path.join(mmcls_val_dir, 'class1'),
            os.path.join(mmcls_val_dir, 'class2'),
            os.path.join(mmcls_test_dir, 'class1'),
            os.path.join(mmcls_test_dir, 'class2')
        ]

        for folder in mmcls_dataset_folders:
            os.makedirs(folder, exist_ok=True)

        for subfolder in os.listdir(masked_imgs_kept_dir):
            subfolder_path = os.path.join(masked_imgs_kept_dir, subfolder)
            split_files(subfolder_path, mmcls_dataset_folders[0], mmcls_dataset_folders[2], mmcls_dataset_folders[4])

        for subfolder in os.listdir(masked_img_dropped_dir):
            subfolder_path = os.path.join(masked_img_dropped_dir, subfolder)
            split_files(subfolder_path, mmcls_dataset_folders[1], mmcls_dataset_folders[3], mmcls_dataset_folders[5])

        generate_txt(mmcls_dataset_path)

        # Inserting the function parameter into the content
        args.config_dir = 'mask_rcnn_r50_caffe_fpn_1x_hepg2'  # Change this to 'faster_rcnn' or any other function name
        modified_template = generate_classification_dataset(template, args.config_dir)

        # Writing content to a file
        with open(os.path.join(mmcls_root, r"configs\_base_\datasets\hepg2.py"), "w") as f:
            f.write(modified_template)

    elif args.set == 'test':
        reversed_mask_preds_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set,
                                               'reversed_mask_prediction')
        reversed_masked_imgs_dir = os.path.join(mmdet_root_dir, 'work_dirs', args.config_dir, args.set,
                                                  'reversed_masked_imgs')

        for idx in range(num_preds):
            img_id = inference_results[idx]['image_id']
            img_name = img_id_to_name.get(img_id, "")
            img_name_digit = img_id_to_name_digit.get(img_id, "")
            # mask_pred_folder = os.path.join(mask_preds_dir, img_name_digit)
            # os.makedirs(mask_pred_folder, exist_ok=True)
            mask_pred = pycoco_mask.decode(inference_results[idx]['segmentation'])
            labeled_img, mask_count = skimage.measure.label(mask_pred, return_num=True)

            if mask_count == 1:
                if img_name_digit[-1] !='R':
                    mask_pred_copy = (mask_pred * 255).astype(np.uint8)
                    mask_pred_save_path = os.path.join(mask_preds_dir, img_name_digit, f"{idx}.png")
                    os.makedirs(os.path.join(mask_preds_dir, img_name_digit), exist_ok=True)
                    save_img(mask_pred_save_path, mask_pred_copy)
                    reversed_mask_pred_save_path = os.path.join(reversed_mask_preds_dir, img_name_digit, f"{idx}.png")
                    os.makedirs(os.path.join(reversed_mask_preds_dir, img_name_digit), exist_ok=True)
                    save_img(reversed_mask_pred_save_path, mask_pred_copy)
                    raw_img_path = os.path.join(mmdet_dataset_dir, img_name)
                    raw_img = load_img(raw_img_path)
                    if is_grayscale(raw_img):
                        raw_img[mask_pred_copy == 0] = 0
                    elif not is_grayscale(raw_img):
                        raw_img[mask_pred_copy == 0] = [0, 0, 0]
                    masked_img_save_path = os.path.join(reversed_masked_imgs_dir, img_name_digit, f"{idx}.png")
                    os.makedirs(os.path.join(reversed_masked_imgs_dir, img_name_digit), exist_ok=True)
                    save_img(masked_img_save_path, raw_img)
                else:
                    reversed_rotated_mask = rotate_img(mask_pred, -45, True)
                    cropped_reversed_rotated_mask = reversed_rotated_mask[min_row:max_row + 1, min_col:max_col + 1]
                    non_255_mask = cropped_reversed_rotated_mask < 1
                    filtered_mask = cropped_reversed_rotated_mask.copy()
                    filtered_mask[non_255_mask] = 0
                    mask_pred_copy = (filtered_mask * 255).astype(np.uint8)
                    mask_pred_save_path = os.path.join(mask_preds_dir, img_name_digit[:-1], f"{idx}.png")
                    os.makedirs(os.path.join(mask_preds_dir, img_name_digit[:-1]), exist_ok=True)
                    save_img(mask_pred_save_path, mask_pred_copy)
                    reversed_mask_pred_save_path = os.path.join(reversed_mask_preds_dir, img_name_digit[:-1], f"{idx}.png")
                    os.makedirs(os.path.join(reversed_mask_preds_dir, img_name_digit[:-1]), exist_ok=True)
                    save_img(reversed_mask_pred_save_path, mask_pred_copy)
                    raw_img_path = os.path.join(mmdet_dataset_dir, f"{img_name_digit[:-1]}.png")
                    raw_img = load_img(raw_img_path)
                    if is_grayscale(raw_img):
                        raw_img[filtered_mask == 0] = 0
                    elif not is_grayscale(raw_img):
                        raw_img[filtered_mask == 0] = [0, 0, 0]
                    masked_img_save_path = os.path.join(reversed_masked_imgs_dir, img_name_digit[:-1], f"{idx}.png")
                    os.makedirs(os.path.join(reversed_masked_imgs_dir, img_name_digit[:-1]), exist_ok=True)
                    save_img(masked_img_save_path, raw_img)


if __name__ == "__main__":
    main()
