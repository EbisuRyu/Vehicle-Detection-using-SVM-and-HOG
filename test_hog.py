import cv2
from script.feature_source import FeatureExtracter
from script.helpers import show_images
import matplotlib.pyplot as plt 

start_frame = cv2.imread("dataset/vehicles/GTI_MiddleClose/image0077.png")
start_frame = cv2.imread("./test_images/highway-45-46.jpg")
sourcer_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb 
  'orientations': 9,        # 6 - 12
  'pixels_per_cell': 8,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'transform_sqrt': True,
  'block_norm': 'L2'
}

sourcer = FeatureExtracter(**sourcer_params)
features = sourcer.features(start_frame)
print("feature shape:", features.shape)

plt.imshow(start_frame)
plt.show()
'''f = sourcer.features(start_frame)
print("feature shape:", f.shape)

rgb_img, a_img, b_img, c_img = sourcer.visualize()
show_images([rgb_img, a_img, b_img, c_img], per_row = 4, per_col = 1, W = 10, H = 2)'''