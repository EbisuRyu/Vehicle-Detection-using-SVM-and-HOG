import cv2
import joblib
from script.dataset import load_vehicle_dataset
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from script.feature_source import FeatureExtracter
image = cv2.imread('./dataset/vehicles/GTI_Far/image0000.png')
image = cv2.resize(image, (32, 32))
print(image.shape)
plt.imshow(image)
plt.show()