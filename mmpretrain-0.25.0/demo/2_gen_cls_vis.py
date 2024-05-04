import os
import shutil
from argparse import ArgumentParser

import imageio.v2
import torch
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
    parser.add_argument('--mmdet_config', default='mask_rcnn_r50_caffe_fpn_1x_hepg2', help='The MMDet config file name')
    parser.add_argument('--mmcls_config', default='resnet18_8xb32_in1k_hepg2', help='The MMCls config file name')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    mmdet_base_dir = r'G:\dvsisn_hepg2\mmdetection-2.28.2'
    mmcls_base_dir = r'G:\dvsisn_hepg2\mmpretrain-0.25.0'

    reversed_mask_preds_dir = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, 'test',
                                           'reversed_mask_prediction')
    reversed_masked_images_dir = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, 'test',
                                              'reversed_masked_imgs')
    model_config_path = os.path.join(mmcls_base_dir, 'configs', 'resnet', f'{args.mmcls_config}.py')
    model_checkpoint_path = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, 'latest.pth')
    model = init_model(model_config_path, model_checkpoint_path, device=args.device)
    vis_folder = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config, 'vis_cls')
    os.makedirs(vis_folder, exist_ok=True)

    for image_path in iterate_images(reversed_masked_images_dir):
        img_name = image_path.split(os.path.sep)[-2]
        mask_name = image_path.split(os.path.sep)[-1]
        vis_imgname_folder = os.path.join(vis_folder, img_name)
        reversed_mask_pred_path = os.path.join(reversed_mask_preds_dir, img_name, mask_name)
        mask_save_path = os.path.join(vis_imgname_folder, mask_name)
        score = inference_score(model, image_path)
        if score[0][0] > 0.4:
            os.makedirs(vis_imgname_folder, exist_ok=True)
            shutil.copy(reversed_mask_pred_path, mask_save_path)


if __name__ == '__main__':
    main()
