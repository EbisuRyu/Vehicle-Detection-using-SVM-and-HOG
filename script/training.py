# import the necessary packages
from script.model_localization import pyramid, sliding_window, iou_bbox, non_maximum_supperssion, visualize_bbox
from script.model_classification import SVMObjectClassifier
from sklearn.model_selection import train_test_split
from script.dataset import load_vehicle_dataset
from feature_source import FeatureExtracter
import numpy as np
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
ap.add_argument("-t", "--thresh", type=float, default=0.9, help="threshold for classification")
args = vars(ap.parse_args())

sourcer_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb 
  'orientations': 11,        # 6 - 12
  'pixels_per_cell': 16,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'transform_sqrt': True,
  'block_norm': 'L2'
}
feature_extracter = FeatureExtracter(**sourcer_params)
# load the image and define the window width and height
image = cv2.imread(args["image"])
windowSize = [(50, 50), (80, 80), (100, 100)]
X, y = load_vehicle_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
model = SVMObjectClassifier(C=0.5)
model.get_feature_extracter(feature_extracter)
model.train(X_train, y_train)
model.evaluate(X_test, y_test)

predict_bboxs = []
# loop over the image pyramid
for resized, scale in pyramid(image, scale=2):
    for (winW, winH) in windowSize:
        for (x, y, window) in sliding_window(resized, stepSize=20, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            x_min, y_min, x_max, y_max = x, y, x + winW, y + winH
            probs, labels = model.predict([window])
            conf_score, class_name = probs[0], labels[0]
            print(conf_score, class_name)
            label = f'{class_name}: {conf_score:.2f}'
            clone = resized.copy()
            if conf_score > args["thresh"] and class_name == 'vehicle':
                x_min = int(x_min * scale)
                y_min = int(y_min * scale)
                x_max = int(x_max * scale)
                y_max = int(y_max * scale)
                predict_bboxs.append([x_min, y_min, x_max, y_max, class_name, conf_score])
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            #cv2.rectangle(image, (x_min, y_min - 20), (x_min + w, y_min), (0, 255, 0), -1)
            #cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)
            
predict_bboxs = non_maximum_supperssion(predict_bboxs, 0.2)
visualize_bbox(image, predict_bboxs)