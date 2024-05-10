import numpy as np
import cv2
import time 
from script.model_localization import pyramid, sliding_window, iou_bbox, non_maximum_supperssion, visualize_bbox
class Slider:
    
    def __init__(self, classifier, windows, feature_window_size, stride, scale, minSize=(30, 30), strip_position=None, visualize=False):
        self.classifier = classifier
        self.stride = stride
        self.windows = windows
        self.feature_window_size = feature_window_size
        self.scale = scale
        self.minSize = minSize
        self.strip_position = strip_position
        self.visualize = visualize
        self.curent_strip = None
        
    def generate_bounding_boxes(self, image):
        bboxs = []
        window_images = []
        for resized, scale in pyramid(image, scale=self.scale, minSize=self.minSize):
            for (winW, winH) in self.windows:
                y_start = int(self.strip_position[0] / scale)
                y_end = int(self.strip_position[1] / scale)
                self.curent_strip = image[y_start:y_end, :, :]
                for (x, y, window) in sliding_window(resized, stepSize=self.stride, windowSize=(winW, winH), strip_position=(y_start, y_end)):
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue
                    x_min, y_min, x_max, y_max = x, y, x + winW, y + winH
                    bboxs.append([x_min, y_min, x_max, y_max, scale])
                    window = cv2.resize(window, self.feature_window_size)
                    window_images.append(window)
                    if self.visualize:
                        clone = resized.copy()
                        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                        cv2.imshow("Window", clone)
                        cv2.waitKey(1)
                        time.sleep(0.025)
        cv2.destroyAllWindows()
        return np.array(bboxs), np.array(window_images)
    
    def predict(self, image, threshold=0.7):
        predict_bboxs = []
        bboxs, window_images = self.generate_bounding_boxes(image)
        probs, labels = self.classifier.predict(window_images)
        vehicle_ids = np.where(labels == 'vehicle')
        scale_predict_bboxs = bboxs[vehicle_ids]
        vehicle_probs = probs[vehicle_ids]
        for (x_min, y_min, x_max, y_max, scale), probability in zip(scale_predict_bboxs, vehicle_probs):
            if probability > threshold:
                x_min = int(x_min * scale)
                y_min = int(y_min * scale)
                x_max = int(x_max * scale)
                y_max = int(y_max * scale)
                predict_bboxs.append([x_min, y_min, x_max, y_max, 'vehicle', probability])
        return predict_bboxs