from script.dataset import load_vehicle_dataset
from script.feature_source import FeatureExtracter
from script.training import training_model
from script.slider import Slider
from script.model_localization import non_maximum_supperssion, visualize_bbox
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
ap.add_argument("-t", "--thresh", type=float, default=0.9, help="threshold for classification")
ap.add_argument("-st", "--stride", type=int, default=20, help="stride")
ap.add_argument("-v", "--visualize", type=bool, default=False, help="visualize")
ap.add_argument("-io", "--iou", type=float, default=0.2, help="iou threshold")
args = vars(ap.parse_args())

sourcer_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb,
  'spatial_size': (64, 64),            # (16, 16), (32, 32), (64, 64)
  'orientations': 9,        # 6 - 12
  'pixels_per_cell': 8,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'transform_sqrt': True,
  'block_norm': 'L2',
  'hog_visualize': False
}

X, y = load_vehicle_dataset()
feature_extracter = FeatureExtracter(**sourcer_params)
model = training_model(X, y, feature_extracter, './save_model', evaluate=False)

image = cv2.imread(args["image"])
windowSize = [(80, 80)]
feature_window_size = (64, 64)
slider = Slider(model, None, stride=args['stride'], scale=args['scale'], visualize=args['visualize'])
predict_bbox = []
for window_size in windowSize:
  slider.update_window_size(window_size)
  predict_bbox += slider.predict(image, threshold=args['thresh'])
predict_bbox = non_maximum_supperssion(predict_bbox, iou_threshold=args['iou'])
visualize_bbox(image, predict_bbox)