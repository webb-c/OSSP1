import cv2
import os
import random
import sys
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import time
import datetime
from keras.datasets import cifar10

sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')
import detectors
import networks
import helper

import detectors.denoising_detector as denoising
import detectors.pca_detector as pca
import detectors.OPA2D_detector as opa2d
from detectors.binary_detector import ResNetforOSP
from networks.resnet import ResNet

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

random.seed(42)
resnet = ResNet()
binary= ResNetforOSP()

### get data
def get_data():
    _, (x_test, y_test) = cifar10.load_data()
    return (x_test, y_test)

### performance
def get_performance(detect_method=binary) :
    data = 0
    correct = 0
    total_time = 0
    x_test, y_test = get_data()
    # print(y_test[:15]) # [0, 1, 2, 3, 4 ,... , 9]
    # length = len(x_test)
    
    accuracy_List = []
    data_List = []
    total_time_List = []
    
    for i in tqdm(range(100)) :
        x = x_test[i]
        y = y_test[i][0]
        # origin
        start_time = time.time()
        # ret = detect_method.is_attack(x)
        ret_0 = denoising.is_attack(x)
        ret_1 = pca.is_attack(x)
        ret_2 = binary.is_attack(x)
        ret_3 = opa2d.is_attack(x)
        end_time = time.time()
        total_time += (end_time - start_time)
        pred = np.argmax(resnet.predict(x)[0])
        if not ret :
            data += 1
            if pred == y : correct += 1
        # attack
        copy_x = copy.deepcopy(x)
        attack_x = opa2d.reattack(copy_x, pred, resnet, maxiter=30, verbose=False)[-2]
        start_time = time.time()
        # attack_ret = detect_method.is_attack(attack_x)
        end_time = time.time()
        total_time += (end_time - start_time)
        if not attack_ret :
            data += 1
            attack_pred = np.argmax(resnet.predict(x)[0])
            if attack_pred == y : correct += 1
    
    accuracy = 100*(correct / data)
    return (data, accuracy, total_time)

### test
data, accuracy, total_time = get_performance()

print("==== Accuracy ====")
print("Denosing : " + str(accuracy[0]))
print("Pca      : " + str(accuracy[1]))
print("Binary   : " + str(accuracy[2]))
print("OPA2D    : " + str(accuracy[3]))

print("==== # of data ====")
print("Denosing : " + str(data[0]))
print("Pca      : " + str(data[1]))
print("Binary   : " + str(data[2]))
print("OPA2D    : " + str(data[3]))

print("==== total time for decide attack or not ====")
print("Denosing : " + str(total_time[0]))
print("Pca      : " + str(total_time[1]))
print("Binary   : " + str(total_time[2]))
print("OPA2D    : " + str(total_time[3]))