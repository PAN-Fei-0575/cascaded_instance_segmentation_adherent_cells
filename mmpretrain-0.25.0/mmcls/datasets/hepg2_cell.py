# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:14:26 2022

@author: weixing
"""

# -*- coding:utf8 -*-

import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset
 
@DATASETS.register_module()
class HepG2(BaseDataset):
    CLASSES = ["class1", "class2"]
    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos