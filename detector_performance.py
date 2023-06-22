# Standard library imports
import os
import random

# Third party imports
import cv2
from tqdm import tqdm

# Local application imports
import detectors.denoising_detector as denosing
import detectors.pca_detector as pca
import detectors.OPA2D_detector as opa2d
from detectors.binary_detector import ResNetforOSP

count_per_class = 10
random.seed(42)
binary = ResNetforOSP()
path = os.path.dirname(os.path.abspath(__file__))
attack_image_folder =  os.path.join(path, 'images/resnet_sample/attack')
original_image_folder = os.path.join(path, 'images/resnet_sample/original')

def __do_attack_detect(idx, image):
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    if denosing.is_attack(image) : denosing_detect_success_number[idx] += 1
    if pca.is_attack(image) : pca_detect_success_number[idx] += 1
    if binary.is_attack(image) : binary_detect_success_number[idx] += 1
    if opa2d.is_attack(image) : opa2d_detect_success_number[idx] += 1

def __do_original_detect(idx, image):
    global original_denosing_detect_success_number, original_pca_detect_success_number, original_binary_detect_success_number, original_opa2d_detect_success_number
    if not denosing.is_attack(image) : original_denosing_detect_success_number[idx] += 1
    if not pca.is_attack(image) : original_pca_detect_success_number[idx] += 1
    if not binary.is_attack(image) : original_binary_detect_success_number[idx] += 1
    if not opa2d.is_attack(image) : original_opa2d_detect_success_number[idx] += 1

def __init_variable():
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    global original_denosing_detect_success_number, original_pca_detect_success_number, original_binary_detect_success_number, original_opa2d_detect_success_number
    denosing_detect_success_number =  [0 for _ in range(10)]
    pca_detect_success_number =  [0 for _ in range(10)]
    binary_detect_success_number =  [0 for _ in range(10)]
    opa2d_detect_success_number =  [0 for _ in range(10)]
    original_denosing_detect_success_number =  [0 for _ in range(10)]
    original_pca_detect_success_number =  [0 for _ in range(10)]
    original_binary_detect_success_number =  [0 for _ in range(10)]
    original_opa2d_detect_success_number =  [0 for _ in range(10)]

def __load_image_file_name_list():
    global attack_image_folder, original_image_folder, attack_file_name_list, original_file_name_list, total_image_number, tmp_attack_file, tmp_original_file, data_class
    attack_file_name_list = [[] for _ in range(10)]
    original_file_name_list = [[] for _ in range(10)]
    
    data_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    tmp_attack_file = [f for f in os.listdir(attack_image_folder) if f.endswith('.png')]
    random.shuffle(attack_file_name_list)

    tmp_original_file = [f for f in os.listdir(original_image_folder) if f.endswith('.png')]
    random.shuffle(tmp_original_file)

    for i, class_name in enumerate(data_class) :
        count = 0
        for file in tmp_attack_file :
            if file.split('_')[1] == class_name + '.png' :
                attack_file_name_list[i].append(file)
                count += 1
            if count == count_per_class:
                break
        if count < count_per_class:
            print("error : test image number is more than count_per_class !!")
            exit()

    for i, class_name in enumerate(data_class) :
        count = 0
        for file in tmp_original_file :
            if file.split('_')[1] == class_name + '.png' :
                original_file_name_list[i].append(file)
                count += 1
            if count == count_per_class:
                break
        if count < count_per_class:
            print("error : test image number is more than count_per_class !!")
            exit()


def __load_attack_image(class_idx, idx):
    global attack_image_folder, attack_file_name_list
    image_path = os.path.join(attack_image_folder, attack_file_name_list[class_idx][idx])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def __load_original_image(class_idx, idx):
    global original_image_folder, original_file_name_list
    image_path = os.path.join(original_image_folder, original_file_name_list[class_idx][idx])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def __print_result():
    global total_image_number, data_class
    global denosing_detect_success_number, pca_detect_success_number, binary_detect_success_number, opa2d_detect_success_number
    global original_denosing_detect_success_number, original_pca_detect_success_number, original_binary_detect_success_number, original_opa2d_detect_success_number
    
    for i, cls in enumerate(data_class) :
        print("\n==== class name : " + cls+"====")
        print("==== Attack Accuracy ====")
        print("Denosing : " + str(denosing_detect_success_number[i]/total_image_number[i]))
        print("Pca      : " + str(pca_detect_success_number[i]/total_image_number[i]))
        print("Binary   : " + str(binary_detect_success_number[i]/total_image_number[i]))
        print("OPA2D    : " + str(opa2d_detect_success_number[i]/total_image_number[i]))

        print("==== Original Accuracy ====")
        print("Denosing : " + str(original_denosing_detect_success_number[i]/total_image_number[i]))
        print("Pca      : " + str(original_pca_detect_success_number[i]/total_image_number[i]))
        print("Binary   : " + str(original_binary_detect_success_number[i]/total_image_number[i]))
        print("OPA2D    : " + str(original_opa2d_detect_success_number[i]/total_image_number[i]))

if __name__ == "__main__":
    global total_image_number
    total_image_number = [count_per_class] * 10
    __init_variable()
    __load_image_file_name_list()
    
    for class_idx in tqdm(range(10)) : 
        for idx in range(total_image_number[class_idx]):
            image = __load_attack_image(class_idx, idx)
            __do_attack_detect(class_idx, image)
        
        for idx in range(total_image_number[class_idx]):
            image = __load_original_image(class_idx, idx)
            __do_original_detect(class_idx, image)
        
    __print_result()