import os
import shutil
from argparse import ArgumentParser

import cv2
import torch
import imageio.v2
import numpy as np
from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


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
    return result_iou, chosen_mask_path, det_iou, pred_iou


def count_connected_components(mask_path):
    mask = load_image(mask_path)
    num1, labels, stats1, centroids = cv2.connectedComponentsWithStats(mask, 8)
    inverted_mask = 255 - mask
    num2, labels, stats2, centroids = cv2.connectedComponentsWithStats(inverted_mask, 8)
    for i in range(num1):
        if stats1[i, :][4] < 20:
            num1 = num1 - 1
    for i in range(num2):
        if stats2[i, :][4] < 50:
            num2 = num2 - 1
    num = num1 + num2
    return num

def inference_score(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label`, and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # Build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)

    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # Scatter to specified GPU
        data = scatter(data, [device])[0]

    # Forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)

    return scores


def iterate_images(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            yield image_path


def main():
    parser = ArgumentParser()
    parser.add_argument('--set', default='test', help='The set')
    parser.add_argument('--mmdet_config', default='mask_rcnn_r50_caffe_fpn_1x_hepg2', help='The MMDet config file name')
    parser.add_argument('--mmcls_config', default='resnet18_8xb32_in1k_hepg2', help='The MMCls config file name')
    
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    mmcls_base_dir = r'G:\dvsisn_hepg2\mmpretrain-0.25.0'
    mmdet_base_dir = r'G:\dvsisn_hepg2\mmdetection-2.28.2'
    reversed_mask_preds_dir = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, args.set,
                                          'reversed_mask_prediction')
    reversed_masked_images_dir = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, 'test',
                                              'reversed_masked_images')
    model_config_path = os.path.join(mmcls_base_dir, 'configs', 'resnet', f'{args.mmcls_config}.py')
    model_checkpoint_path = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, 'latest.pth')
    model = init_model(model_config_path, model_checkpoint_path, device=args.device)
    vis_cls_final_dir = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config,  'vis_cls_final')
    collect_final_dir = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config, 'vis_cls_final', 'collect')

    for directory in [vis_cls_final_dir, collect_final_dir]:
        os.makedirs(directory, exist_ok=True)

    image_folders = [folder for folder in os.listdir(reversed_mask_preds_dir) if folder != 'collect']

    for image_folder_name in image_folders:
        img_folder_dir = os.path.join(reversed_mask_preds_dir, image_folder_name)
        img_folder_final_dir = os.path.join(vis_cls_final_dir, image_folder_name)
        os.makedirs(img_folder_final_dir, exist_ok=True)
        masks = os.listdir(img_folder_dir)
        masks_to_remove = []

        for mask in masks.copy():
            if mask in masks_to_remove:
                continue
            mask1_path = os.path.join(reversed_mask_preds_dir, image_folder_name, mask)
            masked_image1_path = os.path.join(reversed_masked_images_dir, image_folder_name, mask)
 
            if count_connected_components(mask1_path) > 6:
                masks.remove(mask)
                masks_to_remove.append(mask)
                continue           
            
            for compare_mask in masks.copy():
                if compare_mask in masks_to_remove:
                    continue
                
                mask2_path = os.path.join(reversed_mask_preds_dir, image_folder_name, compare_mask)
                masked_image2_path = os.path.join(reversed_masked_images_dir, image_folder_name, compare_mask)
                if mask1_path != mask2_path:
                    iou, path, iou1, iou2 = mask_iou_per(mask, compare_mask, img_folder_dir)
                    if iou > 0.5:
                        cls_score1 = inference_score(model, masked_image1_path);
                        cls_score2 = inference_score(model, masked_image2_path);
                        score1 = count_connected_components(mask1_path) + iou1*3 + 1-cls_score1[0][0]
                        # print(mask1_path, count_connected_components(mask1_path))
                        score2 = count_connected_components(mask2_path) + iou2*3 + 1-cls_score2[0][0]
                        # print(mask2_path, count_connected_components(mask2_path))
                        if score2 > score1:
                            masks.remove(compare_mask)
                            masks_to_remove.append(compare_mask)

        if not masks:
            print('The list of ' + image_folder_name + 'is null.')
        else:
            for mask in masks:
                src_mask_path = os.path.join(reversed_mask_preds_dir, image_folder_name, mask)
                dst_mask_path = os.path.join(vis_cls_final_dir, image_folder_name, mask)
                shutil.copy(src_mask_path, dst_mask_path)


if __name__ == '__main__':
    main()
