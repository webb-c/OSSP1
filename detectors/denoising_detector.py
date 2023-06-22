# Standard library imports
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Third party imports
import cv2
import numpy as np

# Local application imports
from networks.resnet import ResNet

resnet = ResNet()

def __denoising(origin, kernal_size):
    return cv2.medianBlur(src = origin, ksize = kernal_size)

def __calculate_difference(origin, denoising) :
    confidence_original = resnet.predict(origin)[0]
    confidence_denoising = resnet.predict(denoising)[0]
    diff = confidence_original - confidence_denoising
    abs_diff = np.abs(diff)
    diff_sum = np.sum(abs_diff) / 2
    return diff_sum

def is_attack(image, threshold = 0.00014) :
    return get_value(image) > threshold
    
def get_value(image) :
    decoded_image = __denoising(image, kernal_size = 3)
    diff_sum = __calculate_difference(image, decoded_image)
    return diff_sum