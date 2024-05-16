import numpy as np
import cv2
import time 
from script.model_localization import pyramid, sliding_window, iou_bbox, non_maximum_supperssion, visualize_bbox

class Slider:
    
    def __init__(self, classifier, window_size, stride, scale, minSize=(30, 30), strip_position=None, visualize=False):
        self.classifier = classifier
        self.stride = stride
        self.window_size = window_size
        self.scale = scale
        self.minSize = minSize
        self.strip_position = strip_position
        self.visualize = visualize
        self.curent_strip = None
        
    def update_strip_position(self, strip_position):
        self.strip_position = strip_position
    
    def update_window_size(self, window_size):
        self.window_size = window_size
    
    def window_visualize(self, image, x_min, y_min, x_max, y_max):
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Window", image)
        cv2.waitKey(1)
        time.sleep(0.025)
        
    def generate_bounding_boxes(self, image):
        bboxs = []
        window_images = []
        for resized, scale in pyramid(image, scale=self.scale, minSize=self.minSize):
            if self.strip_position:
                y_start, y_end = int(self.strip_position[0] / scale), int(self.strip_position[1] / scale)
                self.curent_strip = image[y_start:y_end, :, :]
                strip_postion = (y_start, y_end)
            else:
                strip_postion = None
            (winW, winH) = self.window_size 
            for (x, y, window) in sliding_window(resized, stepSize=self.stride, windowSize=(winW, winH), strip_position=strip_postion):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                x_min, y_min, x_max, y_max = x, y, x + winW, y + winH
                bboxs.append([x_min, y_min, x_max, y_max, scale])
                window_images.append(window)
                if self.visualize:
                    clone = resized.copy()
                    self.window_visualize(clone, x_min, y_min, x_max, y_max)
                        
        if self.visualize:
            cv2.destroyAllWindows()
        return np.array(bboxs), np.array(window_images)
    
    def predict(self, image, threshold=0.7):
        predict_bboxs = []
        bboxs, window_images = self.generate_bounding_boxes(image)
        probs, labels = self.classifier.predict(window_images)
        '''scale_predict_bboxs = bboxs[np.where(labels == 'vehicle')]
        vehicle_probs = probs[np.where(labels == 'vehicle')]'''
        
        for (x_min, y_min, x_max, y_max, scale), probability, label in zip(bboxs, probs, labels):
            if probability > threshold:
                x_min, y_min = int(x_min * scale), int(y_min * scale)
                x_max, y_max = int(x_max * scale), int(y_max * scale)
                predict_bboxs.append([x_min, y_min, x_max, y_max, label, probability])
        return predict_bboxs