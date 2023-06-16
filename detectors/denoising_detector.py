'''
denoising detector
'''
import cv2
import numpy as np
import os
import sys
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')
import helper
from networks.resnet import ResNet

kernal_size = 3
resnet = ResNet()

def __denoising(origin):
    return cv2.medianBlur(src = origin, ksize = kernal_size)

def __calculate_difference(origin, denoising) :
    confidence_original = resnet.predict(origin)[0]
    confidence_denoising = resnet.predict(denoising)[0]

    diff = confidence_original - confidence_denoising
    abs_diff = np.abs(diff)
    diff_sum = np.sum(abs_diff)/2
    return diff_sum

def is_attack(image, threshold = 0.5) :
    return get_value(image) > threshold
    
def get_value(image) :
    decoded_image = __denoising(image)
    #helper.plot_image(decoded_image)

    diff_sum = __calculate_difference(image, decoded_image)
    return diff_sum

# if __name__ == "__main__":
#     origin_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/original/1037_automobile.png', cv2.IMREAD_COLOR)
#     origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
#     attack_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/attack/1037_automobile.png', cv2.IMREAD_COLOR)
#     attack_img = cv2.cvtColor(attack_img, cv2.COLOR_BGR2RGB)

#     origin = np.array(origin_img)
#     attack = np.array(attack_img)

#     decoded = get_value(origin)
#     decoded = get_value(attack)