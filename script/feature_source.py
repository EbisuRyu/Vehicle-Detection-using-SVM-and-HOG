from skimage import feature
from script.helpers import convert
import numpy as np
import cv2
from skimage.transform import resize

class FeatureExtracter:
  def __init__(self, spatial_size, orientations, pixels_per_cell, cells_per_block, transform_sqrt, block_norm = 'L2', hog_visualize = False):
    self.spatial_size = spatial_size 
    self.orientations = orientations
    self.pixels_per_cell = (pixels_per_cell, pixels_per_cell)
    self.cells_per_block = (cells_per_block, cells_per_block)
    self.transform_sqrt = transform_sqrt
    self.block_norm = block_norm
    self.hog_visualize = hog_visualize

  def features(self, image):
    if len(image.shape) > 2:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    
    resized_image = resize(image, self.spatial_size, anti_aliasing=True)
    
    hog_features = feature.hog(
      resized_image, 
      orientations=self.orientations, 
      pixels_per_cell=self.pixels_per_cell, 
      cells_per_block=self.cells_per_block, 
      transform_sqrt=self.transform_sqrt, 
      block_norm=self.block_norm, 
      visualize=self.hog_visualize,
      feature_vector=True
      )
    
    return hog_features