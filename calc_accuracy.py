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
    
    data = [0, 0, 0, 0]
    correct = [0, 0, 0, 0]
    total_time = [0, 0, 0, 0]
    detect_model = [denoising, pca, binary, opa2d]
    
    for i in tqdm(range(2500)) :
        x = x_test[i]
        y = y_test[i][0]
        
        # origin
        pred = np.argmax(resnet.predict(x)[0])
        
        # attack
        copy_x = copy.deepcopy(x)
        attack_x = opa2d.reattack(copy_x, pred, resnet, maxiter=30, verbose=False)[-2]
        
        # detect
        for i in range(4) :
            detect = detect_model[i]
            
            start_time = time.time()
            ret = detect.is_attack(x)
            end_time = time.time()
            total_time[i] += (end_time - start_time)
            if not ret :
                data[i] += 1
                if pred == y : correct[i] += 1
        
            start_time = time.time()
            attack_ret = detect.is_attack(attack_x)
            end_time = time.time()
            total_time[i] += (end_time - start_time)  
            if not attack_ret :
                data[i] += 1
                attack_pred = np.argmax(resnet.predict(attack_x)[0])
                if attack_pred == y : correct[i] += 1
    
    accuracy = []
    for i in range(4) :
        if data[i] == 0 :
            accuracy.append("NaN")
        else : accuracy.append(100*(correct[i] / data[i]))
    return (data, accuracy, total_time)

### test
start_time = time.time()
data, accuracy, total_time = get_performance()
end_time = time.time()

print("total running time :", end_time - start_time)

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