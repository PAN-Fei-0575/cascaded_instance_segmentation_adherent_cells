_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/hepg2.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

lr_config = dict(policy='step', step=[3, 6, 9])
runner = dict(type='EpochBasedRunner', max_epochs=12)