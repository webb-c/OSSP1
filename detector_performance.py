import cv2
import os
import random
from tqdm import tqdm
import sys
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')

import detectors.denoising_detector as denosing
import detectors.pca_detector as pca
import detectors.OPA2D_detector as opa2d
import detectors.binary_detector as binary

def __do_detect(image):
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    if denosing.is_attack(image) : denosing_detect_success_number += 1
    if pca.is_attack(image) : pca_detect_success_number += 1
    if binary.is_attack(image) : binary_detect_success_number += 1
    if opa2d.is_attack(image) : opa2d_detect_success_number += 1

def __init_variable():
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    denosing_detect_success_number = 0
    pca_detect_success_number = 0
    binary_detect_success_number = 0
    opa2d_detect_success_number = 0

def __load_image_file_name_list():
    global attack_image_folder, attack_file_name_list, total_image_number
    attack_image_folder = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/attack"
    attack_file_name_list = [f for f in os.listdir(attack_image_folder) if f.endswith('.png')]
    if total_image_number > len(attack_file_name_list):
        print("error : test image number is more than attack image number !!")
        exit()
    random.shuffle(attack_file_name_list)
def __load_image(index):
    global attack_image_folder, attack_file_name_list
    image_path = os.path.join(attack_image_folder, attack_file_name_list[index])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def __print_result():
    global total_image_number
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    print("==== Accuracy ====")
    print("Denosing : " + str(denosing_detect_success_number/total_image_number))
    print("Pca      : " + str(pca_detect_success_number/total_image_number))
    print("Binary   : " + str(binary_detect_success_number/total_image_number))
    print("OPA2D    : " + str(opa2d_detect_success_number/total_image_number))

if __name__ == "__main__":
    total_image_number = 100
    __init_variable()
    __load_image_file_name_list()
   
    for index in range(total_image_number):
        image = __load_image(index)
        __do_detect(image)
    __print_result()