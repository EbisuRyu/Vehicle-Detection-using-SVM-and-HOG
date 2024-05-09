# import the necessary packages
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

def pyramid(image, scale=1.5, minSize=(30, 30)):

    current_scale = 1.0
    yield image, current_scale
    while True:
        current_scale *= scale
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image, current_scale
  
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def visualize_bbox(image, bboxes):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, class_name, conf_score = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f'{class_name}: {conf_score:.2f}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x_min, y_min - 20), (x_min + w, y_min), (0, 255, 0), -1)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def np_max(a, b):
    res = [max(a, b[i]) for i in range(b.size)]
    return np.array(res)

def np_min(a, b):
    res = [min(a, b[i]) for i in range(b.size)]
    return np.array(res)
    
def iou_bbox(box1, box2):
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    
    x1 = np_max(b1_x1, b2_x1)
    y1 = np_max(b1_y1, b2_y1)
    
    x2 = np_min(b1_x2, b2_x2)
    y2 = np_min(b1_y2, b2_y2)
    
    inter = np_max(0, x2 - x1) * np_max(0, y2 - y1)
    s1 = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    s2 = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    union = s1 + s2 - inter
    iou = inter / union
    return iou

def non_maximum_supperssion(bboxes, iou_threshold = 0.2):
    if not bboxes:
        return []
    scores = [bbox[-1] for bbox in bboxes]
    sorted_indices = np.argsort(scores)[::-1]
    
    x_min = np.array([bbox[0] for bbox in bboxes])
    y_min = np.array([bbox[1] for bbox in bboxes])
    x_max = np.array([bbox[2] for bbox in bboxes])
    y_max = np.array([bbox[3] for bbox in bboxes])
    
    keep_bboxes = []
    while sorted_indices.size > 0:
        i = sorted_indices[0]
        keep_bboxes.append(bboxes[i])
        
        iou = iou_bbox(
            np.array([x_min[i], y_min[i], x_max[i], y_max[i]]),
            np.array([
                x_min[sorted_indices[1:]],
                y_min[sorted_indices[1:]],
                x_max[sorted_indices[1:]], 
                y_max[sorted_indices[1:]]
                ])
        )
        
        sorted_indices = sorted_indices[1:][iou < iou_threshold]
    return keep_bboxes