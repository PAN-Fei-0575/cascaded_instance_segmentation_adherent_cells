import os
import shutil
from argparse import ArgumentParser

import cv2
import imageio.v2
import numpy as np


def load_image(image_path):
    try:
        image = imageio.v2.imread(image_path)
        return image
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Unable to read image '{image_path}': {e}")
        return None


def save_image(image_path, image_data):
    if not os.path.exists(image_path):
        imageio.v2.imsave(image_path, image_data)
        print(f"Image saved at '{image_path}'.")
    else:
        print(f"Image '{image_path}' already exists. Skipping...")


def calculate_mask_iou(det_mask, pred_mask):
    union = (pred_mask + det_mask) != 0
    intersection = (pred_mask * det_mask) != 0
    return np.sum(intersection) / np.sum(union)


def mask_iou_per(det_mask_path, pred_mask_path, base_path):
    det_mask = load_image(os.path.join(base_path, det_mask_path)) / 255
    pred_mask = load_image(os.path.join(base_path, pred_mask_path)) / 255

    intersection = (pred_mask * det_mask) != 0
    det_iou = np.sum(intersection) / np.sum(det_mask)
    pred_iou = np.sum(intersection) / np.sum(pred_mask)

    if det_iou > pred_iou:
        result_iou = det_iou
        chosen_mask_path = det_mask_path
    else:
        result_iou = pred_iou
        chosen_mask_path = pred_mask_path
    return result_iou, chosen_mask_path


def count_connected_components(mask_path):
    mask = load_image(mask_path)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(num):
        if stats[i, :][4] < 6:
            num = num - 1
    return num


def main():
    parser = ArgumentParser()
    parser.add_argument('--set', default='test', help='The set')
    parser.add_argument('--mmdet_config', default='mask_rcnn_r50_caffe_fpn_1x_hepg2', help='The MMDet config file name')
    parser.add_argument('--mmcls_config', default='resnet18_8xb32_in1k_hepg2', help='The MMCls config file name')
    args = parser.parse_args()

    mmcls_base_dir = r'G:\dvsisn_hepg2\mmpretrain-0.25.0'
    mmdet_base_dir = r'G:\dvsisn_hepg2\mmdetection-2.28.2'
    reversed_mask_pred_dir = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, args.set,
                                          'reversed_mask_prediction')
    vis_cls_final_dir = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config,  'vis_cls_final')
    collect_final_dir = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config, 'vis_cls_final', 'collect')

    for directory in [vis_cls_final_dir, collect_final_dir]:
        os.makedirs(directory, exist_ok=True)

    image_folders = [folder for folder in os.listdir(reversed_mask_pred_dir) if folder != 'collect']

    for image_folder_name in image_folders:
        img_folder_dir = os.path.join(reversed_mask_pred_dir, image_folder_name)
        img_folder_final_dir = os.path.join(vis_cls_final_dir, image_folder_name)
        os.makedirs(img_folder_final_dir, exist_ok=True)
        masks = os.listdir(img_folder_dir)
        masks_to_remove = []

        for mask in masks.copy():
            if mask in masks_to_remove:
                continue
            mask1_path = os.path.join(reversed_mask_pred_dir, image_folder_name, mask)
            if count_connected_components(mask1_path) > 2:
                masks.remove(mask)
                masks_to_remove.append(mask)
                continue
            for compare_mask in masks.copy():
                if compare_mask in masks_to_remove:
                    continue
                mask2_path = os.path.join(reversed_mask_pred_dir, image_folder_name, compare_mask)
                if mask1_path != mask2_path:
                    iou, path = mask_iou_per(mask, compare_mask, img_folder_dir)
                    if iou > 0.7:
                        if path != mask:
                            masks.remove(path)
                            masks_to_remove.append(path)

        if not masks:
            print('The list of ' + image_folder_name + 'is null.')
        else:
            for mask in masks:
                src_mask_path = os.path.join(reversed_mask_pred_dir, image_folder_name, mask)
                dst_mask_path = os.path.join(vis_cls_final_dir, image_folder_name, mask)
                shutil.copy(src_mask_path, dst_mask_path)


if __name__ == '__main__':
    main()
