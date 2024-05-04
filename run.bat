cd D:\mmdetection-2.28.2\
python demo/0_create_rotated_dataset.py --dataset hepg2
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_hepg2.py
python tools/val.py configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_hepg2.py work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/latest.pth --show-dir work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/val/ --eval bbox segm --eval-options jsonfile_prefix=work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/val/results
python tools/test.py configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_hepg2.py work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/latest.pth --show-dir work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/test/ --eval bbox segm --eval-options jsonfile_prefix=work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/test/results
python demo/1_create_cls_dataset.py --dataset hepg2 --set val --config_dir mask_rcnn_r50_caffe_fpn_1x_hepg2 --img_height 864 --img_width 1152
python demo/1_create_cls_dataset.py --dataset hepg2 --set test --config_dir mask_rcnn_r50_caffe_fpn_1x_hepg2 --img_height 864 --img_width 1152
cd D:\mmpretrain-0.25.0\
python tools/train.py configs/resnet/resnet18_8xb32_in1k_hepg2.py --work-dir work_dirs/resnet18_8xb32_in1k_hepg2/mask_rcnn_r50_caffe_fpn_1x_hepg2
python demo/2_gen_cls_vis.py --mmdet_config mask_rcnn_r50_caffe_fpn_1x_hepg2 --mmcls_config resnet18_8xb32_in1k_hepg2 --device cuda:0
python demo/3_mask_nms.py --set test --mmdet_config mask_rcnn_r50_caffe_fpn_1x_hepg2 --mmcls_config resnet18_8xb32_in1k_hepg2
python demo/4_perf_eval.py --set test --mmdet_config mask_rcnn_r50_caffe_fpn_1x_hepg2 --mmcls_config resnet18_8xb32_in1k_hepg2