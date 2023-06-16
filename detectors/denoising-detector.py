'''
denoising detector
'''
import cv2
import numpy as np
import sys
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')
import helper
import random
import os
from networks.resnet import ResNet

#class Denoising_detector:

def denoising(origin, kernal_size = 3):
    if origin is None:
        print('Image load failed!')
        sys.exit()
    print(origin.shape)
    img_median = cv2.medianBlur(src = origin, ksize = kernal_size)
    #helper.plot_image(img_median)    
    return img_median

def calculate_difference(origin, denoising) :
    resnet = ResNet()
    confidence_original = resnet.predict(origin)[0]
    confidence_denoising = resnet.predict(denoising)[0]
    
    diff = confidence_original - confidence_denoising
    abs_diff = np.abs(diff)
    diff_sum = np.sum(abs_diff)
    return diff_sum

def decide_attack_or_origin(value, threshold) :   # threshold 값만 나중에 잘 정해보기
    if value > threshold : print("attack")
    else : print("original")

def get_dataset():
    image_folder = "./image/"
    image_list = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # 이미지 파일 리스트에서 100개 랜덤 추출
    #random_images = random.sample(image_list, 100)

    # 이미지를 로딩하여 리스트에 저장
    loaded_images = []
    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            loaded_images.append(image)
    return loaded_images


random_images = get_dataset()
threshold = 0.5
print(random_images[0].shape)
for image in random_images:
    decoded = denoising(image)
    o_result = calculate_difference(image, decoded)
    decide_attack_or_origin(o_result, threshold)
    
def denoising(origin, kernal_size = 3):
    img_median = cv2.medianBlur(src = origin, ksize = kernal_size)
    helper.plot_image(img_median)    
    return img_median

def calculate_difference(origin, denoising) :
    resnet = ResNet()
    confidence_original = resnet.predict(origin)[0]
    confidence_denoising = resnet.predict(denoising)[0]
    
    diff = confidence_original - confidence_denoising
    abs_diff = np.abs(diff)
    diff_sum = np.sum(abs_diff)
    return diff_sum

def decide_attack_or_origin(value, threshold) :   # threshold 값만 나중에 잘 정해보기
    if value > threshold : print("attack")
    else : print("original")

def get_dataset():
    attack_path = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/lenet_sample/attack"
    attack_fileList = [f for f in os.listdir(attack_path) if f.endswith('.png')]
    random_images = random.sample(attack_fileList, 100)
    return random_images

random_images = get_dataset()
threshold = 0.5

for image in random_images:
    decoded = denoising(image)
    o_result = calculate_difference(image, decoded)
    decide_attack_or_origin(o_result, threshold)
