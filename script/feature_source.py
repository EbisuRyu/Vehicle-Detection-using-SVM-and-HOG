from skimage import feature
import numpy as np
import cv2
from skimage.transform import resize

class FeatureExtracter:
  def __init__(self, spatial_size, orientations, pixels_per_cell, cells_per_block, transform_sqrt, block_norm = 'L2', color_hist=False, hog_visualize=False):
    self.spatial_size = spatial_size 
    self.orientations = orientations
    self.pixels_per_cell = (pixels_per_cell, pixels_per_cell)
    self.cells_per_block = (cells_per_block, cells_per_block)
    self.transform_sqrt = transform_sqrt
    self.block_norm = block_norm
    self.color_hist = color_hist
    self.hog_visualize = hog_visualize
    
    self.BGR_image = None
    self.hog_image = None
    
  def visualize(self):
    return self.BGR_image, self.hog_image
  
  def get_color_histogram(self, image, bins=9):
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
  
  def get_hog_features(self, image):
    if self.hog_visualize:
      hog_features, self.hog_image = feature.hog(
        image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell, 
        cells_per_block=self.cells_per_block, transform_sqrt=self.transform_sqrt, block_norm=self.block_norm, 
        visualize=self.hog_visualize, feature_vector=True
        )
    else:
      hog_features = feature.hog(
        image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell, 
        cells_per_block=self.cells_per_block, transform_sqrt=self.transform_sqrt, block_norm=self.block_norm, 
        visualize=self.hog_visualize, feature_vector=True
        )
    return hog_features
  
  def features(self, image):
    self.BGR_image = image
    if len(image.shape) > 2:
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)
    resized_gray_image = resize(gray_image, self.spatial_size, anti_aliasing=True)
    hog_features = self.get_hog_features(resized_gray_image)
    if self.color_hist:
      RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      rgb_image = RGB_image.astype(np.float32)
      resized_rgb_image = resize(rgb_image, self.spatial_size, anti_aliasing=True)
      color_hist_features = self.get_color_histogram(resized_rgb_image)
      combined_features = np.hstack((hog_features, color_hist_features))
      return combined_features
    else:
      return hog_features