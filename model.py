from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from runway import RunwayModel
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()
#cv2.ocl.setUseOpenCL(False)

detectron = RunwayModel()


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


@detectron.setup
def setup():
    global dummy_coco_dataset
    cfg_file = 'configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
    weights = 'https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
    merge_cfg_from_file(cfg_file)
    cfg.NUM_GPUS = 1
    weights = cache_url(weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'
    model = infer_engine.initialize_model_from_cfg(weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    return model


@detectron.command('detect', inputs={'image': 'image'}, outputs={'output': 'image'})
def detect(model, inp):
    im = np.array(inp['image'])
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, im, None)#, timers=timers)
    #results = get_result_json(cls_boxes, cls_segms, cls_keyps, thresh=args.save_thresh, dataset=dummy_coco_dataset)
    im_new = vis_utils.vis_one_image_opencv(im[:, :, ::-1], cls_boxes, segms=cls_segms, keypoints=cls_keyps, thresh=0.9, kp_thresh=2, show_box=True, dataset=dummy_coco_dataset, show_class=True)
    out = np.array(im_new)
    output = np.clip(out, 0, 255).astype(np.uint8)
    return dict(output=output)


if __name__ == '__main__':
    detectron.run()
