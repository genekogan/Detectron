#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import json
from os import listdir
from os.path import isfile, isdir, join

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
from utils.vis import convert_from_cls_format

import pycocotools.mask as mask_util

from utils.colormap import colormap
import utils.env as envu
import utils.keypoints as keypoint_utils


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)




def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--save-thresh',
        dest='save_thresh',
        help='minimum threshold for score to keep found objects',
        default=0.6,
        type=float
    )
    parser.add_argument(
        '--image-dir',
        dest='image_dir',
        help='outer directory of all images',
        type=str
    )
    parser.add_argument(
        '--json-dir',
        dest='json_dir',
        help='outer directory to put json files',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='outer directory to put output PDFs',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_result_json(boxes, segms, keypoints, thresh=0.7, dataset=None):

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints, _ = keypoint_utils.get_keypoints()

    if segms is not None:
        masks = mask_util.decode(segms)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    sorted_inds = np.argsort(-boxes[:,4])

    results = {'mask_rle':segms, 'objects':[]}
    for i in sorted_inds:
        score = boxes[i, -1]
        
        if score < thresh:
            continue

        bbox = boxes[i, :4]
        class_idx = classes[i]
        class_text = dataset.classes[class_idx]
        mask_idx = i
        mask = masks[:, :, mask_idx]
        #kps = keypoints[i]
        _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = [ c.reshape((-1, 2)).tolist() for c in contour ]        
        obj = {'box':bbox.tolist(), 'class':class_text, 'mask_idx':mask_idx, 'contours':contours, 'score':float(score)}
        results['objects'].append(obj)

    return results




def main2(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    final_json = {'images':[]}

    for i, im_name in enumerate(im_list):

        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        results = get_result_json(cls_boxes, cls_segms, cls_keyps, thresh=args.save_thresh, dataset=dummy_coco_dataset)
        results['path'] = im_name
        results['width'] = im.shape[0]
        results['height'] = im.shape[1]

        final_json['images'].append(results)
        
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.save_thresh,
            kp_thresh=2
        )

    with open('%s/results.json'%args.output_dir, 'w') as outfile:
        json.dump(final_json, outfile)



def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    image_dir = args.image_dir
    output_dir = args.output_dir
    json_dir = args.json_dir
    image_ext = args.image_ext

    inner_dirs = [d for d in listdir(image_dir) if isdir(join(image_dir, d)) ]
    for inner_dir in inner_dirs:

        if not os.path.isdir(join(output_dir, inner_dir)):
            os.mkdir(join(output_dir, inner_dir))

        if not os.path.isdir(join(json_dir, inner_dir)):
            os.mkdir(join(json_dir, inner_dir))

        files = [f for f in listdir(join(image_dir, inner_dir)) if isfile(join(join(image_dir, inner_dir), f)) and f.endswith(image_ext) ]

        for f in files:

            image_file = join(join(image_dir, inner_dir), f)
            out_file = join(join(output_dir, inner_dir), f.replace(".%s"%image_ext, ".pdf"))
            json_file = join(join(json_dir, inner_dir), f.replace(".%s"%image_ext, ".json"))
            
            print("Processing %s -> %s"%(image_file, out_file))
            logger.info('Processing {} -> {}'.format(image_file, out_file))

            im = cv2.imread(image_file)
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, im, None, timers=timers
                )
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

            results = get_result_json(cls_boxes, cls_segms, cls_keyps, thresh=args.save_thresh, dataset=dummy_coco_dataset)
            results['path'] = image_file
            results['width'] = im.shape[0]
            results['height'] = im.shape[1]
            
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                image_file,
                join(output_dir, inner_dir),
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=args.save_thresh,
                kp_thresh=2
            )

            with open(json_file, 'w') as outfile:
                json.dump(results, outfile)




        

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
