import os
import cv2
import glob
import time
import numpy as np
import xml.etree.ElementTree as ET

def load_vehicle_dataset():
    t_start = time.time()
    vehicle_images, non_vehicle_images = [], []
    label_list = []
    for dirpath, dirnames, filenames in os.walk('./dataset/vehicles'):
        for filename in filenames:
                if filename.endswith('.png'):
                    path = os.path.join(dirpath, filename)
                    image = cv2.imread(path)
                    vehicle_images.append(image)
                    label_list.append('vehicle')

    for dirpath, dirnames, filenames in os.walk('./dataset/non-vehicles'):
        for filename in filenames:
                if filename.endswith('.png'):
                    path = os.path.join(dirpath, filename)
                    image = cv2.imread(path)
                    non_vehicle_images.append(image)
                    label_list.append('non-vehicle')
    t_end = time.time()
    vehicle_images, non_vehicle_images = np.asarray(vehicle_images), np.asarray(non_vehicle_images)
    print(f'Loaded dataset in {t_end - t_start:.2f} seconds')
    print("Vehicle images shape: ", vehicle_images.shape)
    print("Non-vehicle images shape: ", non_vehicle_images.shape)
    X = np.vstack((vehicle_images, non_vehicle_images))
    y = np.array(label_list)
    return X, y

def load_traffic_signboard_dataset():
    dataset_dir = './dataset/traffic_sign_board'
    image_dir = os.path.join(dataset_dir, 'images')
    annotation_dir = os.path.join(dataset_dir, 'annotations')
    # Read the dataset
    image_list = []
    label_list = []
    for annotation_file in os.listdir(annotation_dir):
        file_path = os.path.join(annotation_dir, annotation_file)
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        image_file = root.find('filename').text
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)
        
        for object in root.findall('object'):
            label = object.find('name').text
            if label == 'trafficlight':
                continue
            xmin = int(object.find('bndbox/xmin').text)
            ymin = int(object.find('bndbox/ymin').text)
            xmax = int(object.find('bndbox/xmax').text)
            ymax = int(object.find('bndbox/ymax').text)
            
            image_list.append(img[ymin:ymax, xmin:xmax])
            label_list.append(label)
    print('Number of objects: ', len(image_list))
    print('Classes: ', list(set(label_list)))
    return image_list, label_list