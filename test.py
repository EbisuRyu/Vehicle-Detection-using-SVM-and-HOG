import cv2
import joblib
from script.dataset import load_vehicle_dataset
from sklearn.preprocessing import LabelEncoder

image = cv2.imread('./dataset/vehicles/GTI_MiddleClose/image0002.png')
print(image.shape)
label_encoder = LabelEncoder()
X, y = load_vehicle_dataset()
