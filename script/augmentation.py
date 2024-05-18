import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_gaussian_noise(image):
    mean = 0
    sigma = 0.003 * 125  # Giảm mức độ nhiễu Gauss
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def median_blur(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def random_brightness_contrast(image):
    alpha = random.uniform(0.5, 1.5)
    beta = random.uniform(-50, 50)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def random_hue_saturation_value(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = random.uniform(-10, 10)
    saturation = random.uniform(0.5, 1.5)
    value = random.uniform(0.5, 1.5)
    
    hsv_image[..., 0] = (hsv_image[..., 0] + hue) % 180
    hsv_image[..., 1] = cv2.multiply(hsv_image[..., 1], saturation)
    hsv_image[..., 2] = cv2.multiply(hsv_image[..., 2], value)
    
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def apply_random_transformations(image):
    if random.random() < 0.5:
        image = median_blur(image)
    if random.random() < 0.5:
        image = random_brightness_contrast(image)
    return image


def plot_image(image, augmented_image):
    # Hiển thị ảnh gốc và ảnh sau khi tăng cường
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title('Augmented Image')
    plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    plt.show()
    
def generate_images(X_original, desired_length, augmented=True, plot=False):
    X_augmented = []
    while len(X_augmented) < desired_length:
        index = random.randint(0, len(X_original) - 1)
        image = X_original[index]
        if augmented:
            image_augmented = apply_random_transformations(image)
        else:
            image_augmented = image
        if plot:
            plot_image(image, image_augmented)
        X_augmented.append(image_augmented)
    return X_augmented