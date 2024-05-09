from skimage.feature import hog
from script.helpers import convert
import numpy as np
import cv2
from skimage.transform import resize

class FeatureExtracter:
  def __init__(self, color_model, orientations, pixels_per_cell, cells_per_block, transform_sqrt, block_norm = 'L2'):
    
    self.color_model = color_model    
    self.orientations = orientations
    self.pixels_per_cell = (pixels_per_cell, pixels_per_cell)
    self.cells_per_block = (cells_per_block, cells_per_block)
    self.transform_sqrt = transform_sqrt
    self.block_norm = block_norm
    self.ABC_img = None
    self.dims = (None, None, None)
    self.hogA, self.hogB, self.HogC = None, None, None
    self.hogA_img, self.hogB_img, self.hogC = None, None, None

  def hog(self, channel):
    features, hog_img = hog(channel, 
                            orientations = self.orientations, 
                            pixels_per_cell = self.pixels_per_cell,
                            cells_per_block = self.cells_per_block, 
                            transform_sqrt = self.transform_sqrt, 
                            visualize = True, 
                            block_norm = self.block_norm,
                            feature_vector = False)
    return features, hog_img

  def extract_hog(self, image):
    self.RGB_img = image 
    self.ABC_img = convert(image, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape
    
    self.hogA, self.hogA_img = self.hog(self.ABC_img[:, :, 0])
    self.hogB, self.hogB_img = self.hog(self.ABC_img[:, :, 1])
    self.hogC, self.hogC_img = self.hog(self.ABC_img[:, :, 2])
    
  def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):

    if h_pix is None and w_pix is None: 
      h_pix, w_pix = self.s, self.s
    
    h = h_pix // self.pixels_per_cell[0]
    w = w_pix // self.pixels_per_cell[0]
    y_start = y_pix // self.pixels_per_cell[0]
    x_start = x_pix // self.pixels_per_cell[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1
    
    return x_start, x_end, y_start, y_end
    
  def visualize(self):
    return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

  def features(self, image):
    # image = image.astype(np.float32)
    # Resize the image
    # image = resize(image, output_shape = (32, 32), anti_aliasing = True)
    self.extract_hog(image)
    x_start, x_end, y_start, y_end = self.pix_to_hog(0, 0, image.shape[1], image.shape[0])
    
    hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
    hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()
    hog = np.hstack((hogA, hogB, hogC))

    return hog 