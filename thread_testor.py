import cv2
import os
import random
import sys
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')
import detectors

import detectors.denoising_detector as denosing
import detectors.pca_detector as pca
import detectors.OPA2D_detector as opa2d

attack_image_folder = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/attack"
origin_image_folder = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/original"
attack_image_list = [f for f in os.listdir(attack_image_folder) if f.endswith('.png')]
origin_image_list = [f for f in os.listdir(origin_image_folder) if f.endswith('.png')]

origin_OPA2D_value = 0
attack_OPA2D_value = 0
result_list = []
for index in tqdm(range(len(attack_image_list))):
    
    image_path = os.path.join(origin_image_folder, origin_image_list[index])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origin_denosing_value = denosing.get_value(image)
    origin_pca_value = pca.get_value(image)
    origin_OPA2D_value = opa2d.get_distance(image)
    print(image_path)

    image_path = os.path.join(attack_image_folder, attack_image_list[index])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    attack_denosing_value = denosing.get_value(image)
    attack_pca_value = pca.get_value(image)
    attack_OPA2D_value = opa2d.get_distance(image)
    print(image_path)
    
    result = [origin_denosing_value, attack_denosing_value, origin_pca_value, attack_pca_value, origin_OPA2D_value, attack_OPA2D_value]
    result_list.append(result)
    
df = pd.DataFrame(result_list, columns=["origin_denosing_value", "attack_denosing_value", "origin_pca_value", "attack_pca_value", "origin_OPA2D_value", "attack_OPA2D_value"])
df.to_csv('./threashold.csv', index=False)
# for str in result_list:
#     print(str)