import json
import os

import sys
from argparse import ArgumentParser

import imageio.v2
import numpy as np
import mmcv
from pycocotools import mask as pycoco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode as pycoco_encode
from mmdet.core.visualization.image import imshow_det_bboxes



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


def create_image_dict(folder_path):
    image_dict = {}
    for subfolder in os.listdir(folder_path):
        if subfolder != 'collect':
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                images = [img.split('.')[0] for img in os.listdir(subfolder_path)]
                image_dict[subfolder] = images
    return image_dict


def generate_coco_detection_file(coco_gt_path, coco_pred_path, vis_cls_final_dir, coco_dt_path):
    coco_gt = COCO(coco_gt_path)
    image_ids = sorted(coco_gt.getImgIds())
    image_id_name = {}
    for image_id in image_ids:
        image_info = coco_gt.loadImgs(image_id)[0]
        image_name = image_info['file_name'].split(os.sep)[-1]
        image_name_digit = image_name.split('.')[0].split('/')[-1]
        image_id_name[image_name_digit] = image_id

    img_name_mask_name_dict = create_image_dict(vis_cls_final_dir)

    f = open(coco_pred_path, "r")
    data = json.loads(f.read())
    pred_result = []

    for subfolder, mask_list in img_name_mask_name_dict.items():
        img_id = image_id_name[subfolder]
        for mask in mask_list:
            mask_pred_path = os.path.join(vis_cls_final_dir, subfolder, mask + '.png')
            mask_pred = load_image(mask_pred_path)
            mask_pred = (mask_pred / 255).astype(np.uint8)
            rle_mask_pred = pycoco_encode(np.asfortranarray(mask_pred))
            rle_mask_pred['counts'] = rle_mask_pred['counts'].decode('utf-8')

            temp_data = data[int(mask)]
            cat_id = temp_data['category_id']
            conf_score = temp_data['score']
            bounding_box = pycoco_mask.toBbox(rle_mask_pred)

            temp_dict = {"image_id": int(img_id), "category_id": int(cat_id),
                         "segmentation": rle_mask_pred,
                         "bbox": bounding_box.tolist(),
                         "score": conf_score}
            pred_result.append(temp_dict)

    with open(os.path.join(coco_dt_path, 'segm_coco.json'), 'w') as f:
        json.dump(pred_result, f)

    f.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--set', default='test', help='The set')
    parser.add_argument('--mmdet_config', default='mask_rcnn_r50_caffe_fpn_1x_hepg2', help='The MMDet config file name')
    parser.add_argument('--mmcls_config', default='resnet18_8xb32_in1k_hepg2', help='The MMCls config file name')
    args = parser.parse_args()

    # Define paths and variables
    mmcls_base_dir = r'G:\dvsisn_hepg2\mmpretrain-0.25.0'
    mmdet_base_dir = r'G:\dvsisn_hepg2\mmdetection-2.28.2'
    reversed_mask_pred_dir = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, args.set,
                                          'reversed_mask_prediction')
    vis_cls_final_dir = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config,  'vis_cls_final')
    test_set_coco_dt_path = os.path.join(mmcls_base_dir, 'work_dirs', args.mmcls_config, args.mmdet_config, 'vis_cls_final', 'collect')

    test_set_coco_gt_path = os.path.join(mmdet_base_dir, 'data', 'hepg2', 'coco_style', args.set, 'annotations_new_non_overlap.json')
    test_set_coco_pred_path = os.path.join(mmdet_base_dir, 'work_dirs', args.mmdet_config, args.set,
                                           'results.segm.json')

    # Generate COCO detection file
    test_set_coco_gt = COCO(test_set_coco_gt_path)
    generate_coco_detection_file(test_set_coco_gt_path, test_set_coco_pred_path, vis_cls_final_dir,
                                 test_set_coco_dt_path)

    # Evaluate detections
    cocoDt = test_set_coco_gt.loadRes(os.path.join(test_set_coco_dt_path, 'segm_coco.json'))
    imgIds = sorted(test_set_coco_gt.getImgIds())
    catIds = test_set_coco_gt.getCatIds()
    
    for imgId in imgIds:
        annIds = cocoDt.getAnnIds(imgId, catIds)
        anns = cocoDt.loadAnns(annIds)

        conf_score = []
        bboxes = []
        labels = []
        segms = []
        class_names = []

        img_name = cocoDt.imgs[imgId]['file_name']
        img_path = os.path.join(mmdet_base_dir, 'data', 'hepg2', 'coco_style', args.set, 'images', img_name)
        out_file = os.path.join(test_set_coco_dt_path, img_name.split('/')[-1])
        if not os.path.exists(out_file):
            img = mmcv.imread(img_path)
            for ann in anns:
                conf_score.append(ann['score'])
                bboxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                segms.append(cocoDt.annToMask(ann))
                if ann['category_id'] == 1:
                    class_names.append('cell')
                elif ann['category_id'] == 2:
                    class_names.append('bad_cell')

            conf_score = np.array(conf_score)
            conf_score = conf_score[:, np.newaxis]
            bboxes = np.array(bboxes)
            labels = np.array(labels)
            segms = np.array(segms)

            # Calculate x2 and y2
            x2 = bboxes[:, 0] + bboxes[:, 2]
            y2 = bboxes[:, 1] + bboxes[:, 3]

            # Stack x1, y1, x2, y2 horizontally
            bboxes = np.hstack((bboxes[:, :2], x2[:, np.newaxis], y2[:, np.newaxis]))
            bboxes = np.concatenate((bboxes, conf_score), axis=1)

            imshow_det_bboxes(img,
                              bboxes=bboxes,
                              labels=labels,
                              segms=segms,
                              class_names=class_names,
                              score_thr=0,
                              bbox_color='green',
                              text_color='green',
                              mask_color=None,
                              thickness=2,
                              font_size=8,
                              win_name='',
                              show=False,
                              wait_time=0,
                              out_file=out_file)

    # Bbox evaluation
    annType = 'bbox'
    cocoEval = COCOeval(test_set_coco_gt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds

    # Save the current sys.stdout
    original_stdout = sys.stdout

    # Define the file name you want to save to
    file_name = os.path.join(test_set_coco_dt_path, 'stats.txt')

    # Open a new file in write mode
    with open(file_name, "w") as file:
        # Redirect sys.stdout to the file
        sys.stdout = file

        for k in range(1, len(catIds)):
            print("bbox evaluation for catId,", catIds[k])
            cocoEval.params.catIds = catIds[k]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        print("bbox for all catIds")
        cocoEval.params.catIds = catIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # Segmentation evaluation
        annType = 'segm'
        cocoEval = COCOeval(test_set_coco_gt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds

        for k in range(1, len(catIds)):
            print("segmentation evaluation for catId,", catIds[k])
            cocoEval.params.catIds = catIds[k]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        print("segmentation for all catIds")
        cocoEval.params.catIds = catIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    file.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()
