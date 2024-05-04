_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/hepg2.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', step=[3, 6, 9])
runner = dict(type='EpochBasedRunner', max_epochs=12)