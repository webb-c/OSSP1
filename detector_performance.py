import cv2
import os
import random
from tqdm import tqdm
import sys
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')

from detectors.binary_detector import ResNetforOSP
import detectors.denoising_detector as denosing
import detectors.pca_detector as pca
import detectors.OPA2D_detector as opa2d

binary = ResNetforOSP()

def __do_attack_detect(image):
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    if denosing.is_attack(image) : denosing_detect_success_number += 1
    if pca.is_attack(image) : pca_detect_success_number += 1
    if binary.is_attack(image) : binary_detect_success_number += 1
    if opa2d.is_attack(image) : opa2d_detect_success_number += 1

def __do_original_detect(image):
    global original_denosing_detect_success_number, original_pca_detect_success_number, original_binary_detect_success_number, original_opa2d_detect_success_number
    if not denosing.is_attack(image) : original_denosing_detect_success_number += 1
    if not pca.is_attack(image) : original_pca_detect_success_number += 1
    if not binary.is_attack(image) : original_binary_detect_success_number += 1
    if not opa2d.is_attack(image) : original_opa2d_detect_success_number += 1

def __init_variable():
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    global original_denosing_detect_success_number, original_pca_detect_success_number, original_binary_detect_success_number, original_opa2d_detect_success_number
    denosing_detect_success_number = 0
    pca_detect_success_number = 0
    binary_detect_success_number = 0
    opa2d_detect_success_number = 0
    original_denosing_detect_success_number = 0
    original_pca_detect_success_number = 0
    original_binary_detect_success_number = 0
    original_opa2d_detect_success_number = 0

def __load_image_file_name_list():
    global attack_image_folder, original_image_folder, attack_file_name_list, original_file_name_list, total_image_number, tmp_attack_file, tmp_original_file
    attack_file_name_list = []
    original_file_name_list = []
    attack_image_folder = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/attack"
    original_image_folder = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/original"
    data_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'ship']
    
    tmp_attack_file = [f for f in os.listdir(attack_image_folder) if f.endswith('.png')]
    random.shuffle(attack_file_name_list)

    tmp_original_file = [f for f in os.listdir(original_image_folder) if f.endswith('.png')]
    random.shuffle(tmp_original_file)

    for class_name in data_class :
        count = 0
        for file in tmp_attack_file :
            if file.split('_')[1] == class_name + '.png' :
                attack_file_name_list.append(file)
                count += 1
            if count == 10:
                break
        if count < 10:
            print("error : test image number is more than attack image number !!")
            exit()

    for class_name in data_class :
        count = 0
        for file in tmp_original_file :
            if file.split('_')[1] == class_name + '.png' :
                original_file_name_list.append(file)
                count += 1
            if count == 10:
                break
        if count < 10:
            print("error : test image number is more than original image number !!")
            exit()


def __load_attack_image(index):
    global attack_image_folder, attack_file_name_list
    image_path = os.path.join(attack_image_folder, attack_file_name_list[index])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def __load_original_image(index):
    global original_image_folder, original_file_name_list
    image_path = os.path.join(original_image_folder, original_file_name_list[index])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def __print_result():
    global total_image_number
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    global original_denosing_detect_success_number, original_pca_detect_success_number, original_binary_detect_success_number, original_opa2d_detect_success_number

    print("==== Attack Accuracy ====")
    print("Denosing : " + str(denosing_detect_success_number/total_image_number))
    print("Pca      : " + str(pca_detect_success_number/total_image_number))
    print("Binary   : " + str(binary_detect_success_number/total_image_number))
    print("OPA2D    : " + str(opa2d_detect_success_number/total_image_number))

    print("==== Original Accuracy ====")
    print("Denosing : " + str(original_denosing_detect_success_number/total_image_number))
    print("Pca      : " + str(original_pca_detect_success_number/total_image_number))
    print("Binary   : " + str(original_binary_detect_success_number/total_image_number))
    print("OPA2D    : " + str(original_opa2d_detect_success_number/total_image_number))

if __name__ == "__main__":
    global total_image_number
    total_image_number = 100
    __init_variable()
    __load_image_file_name_list()
   
    for index in tqdm(range(total_image_number)):
        image = __load_attack_image(index)
        __do_attack_detect(image)
    
    for index in tqdm(range(total_image_number)):
        image = __load_original_image(index)
        __do_original_detect(image)
    __print_result()