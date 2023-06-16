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

random.seed(42)
resnet = ResNet()
detect = ResNetforOSP()

### cuda setting
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
tf.test.is_gpu_available()

### get data
_, (x_test, y_test) = cifar10.load_data()
length = len(x_test)

### accuracy
def get_accuracy(detect_method=denoising) :
    n = 0
    for i in tqdm(range(length)) :
        x = x_test[i]
        y = y_test[i]
    
        print(y)
        ret = detect_method.is_attack(x)
        #if not ret :
        #    n += 1
        #    pred = resnet.predict(x)[0]
        
        #attack_x = opa2d.reattack()