"""
Testing DNN model' Accuracy on a CIFAR-10 dataset.
with one-pixel attack and detect by detect model (4-method)

Example of Usage :
    $ python accuracy.py --detector binary --model resnet --num 2500
    $ python accuracy.py -d pca -m lenet -n 500

Model Can use :         resnet, lenet
Detector Can use :     denoising, pca, binary, opa2d
Datasets:               https://www.cs.toronto.edu/~kriz/cifar.html 
"""

import argparse

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

# 상대경로로 추가
import detectors
import networks

# detect model
import detectors.denoising_detector as denoising
import detectors.pca_detector as pca
import detectors.OPA2D_detector as opa2d
from detectors.binary_detector import ResNetforOSP

# DNN model
from networks.resnet import ResNet
from networks.lenet import LeNet


tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)  # binary에서도 verbose False로 바꿔야됨

random.seed(42)


def __get_model(modelName):
    global model
    modelList = ["resnet", "lenet"]
    idx = modelList.index(modelName)
    if idx == 0:
        model = ResNet()
    elif idx == 1:
        model = LeNet()


def __get_detector(detectorName):
    global detector
    detectorList = ["denoising", "pca", "binary", "opa2d"]
    idx = detectorList.index(detectorName)
    if idx == 0:
        detector = denoising
    elif idx == 1:
        detector = pca
    elif idx == 2:
        detector = ResNetforOSP()
    elif idx == 3:
        detector = opa2d


def __get_data(dataLength):
    global dataset
    _, (x_test, y_test) = cifar10.load_data()
    dataset = (x_test[:dataLength], y_test[:dataLength])


def __parse_to_args(opt):
    global requireNum
    requireNum = opt.num
    __get_model(opt.model)
    __get_detector(opt.detector)
    __get_data(opt.num)


def __test():
    global detector, model, dataset, requireNum
    global inputNum, correctNum, totalTime
    inputNum = 0 
    correctNum = 0
    totalTime = 0
    x_test, y_test = dataset
    for i in tqdm(range(requireNum)):
        x = x_test[i]
        y = y_test[i][0]

        # attack
        copy_x = copy.deepcopy(x)
        pred = np.argmax(model.predict(x)[0])
        attack_x = opa2d.reattack(copy_x, pred, model, maxiter=50, verbose=False)[-2]

        # detect : original
        start_time = time.time()
        ret = detector.is_attack(x)
        end_time = time.time()
        totalTime += end_time - start_time
        if not ret:
            inputNum += 1
            if pred == y:
                correctNum += 1

        # detect : attack
        start_time = time.time()
        attack_ret = detector.is_attack(attack_x)
        end_time = time.time()
        totalTime += end_time - start_time
        if not attack_ret:
            inputNum += 1
            attack_pred = np.argmax(model.predict(attack_x)[0])
            if attack_pred == y:
                correctNum += 1


def __get_result():
    global inputNum, correctNum, totalTime
    if inputNum == 0 : accuracy = 0
    else : accuracy = correctNum / inputNum
    print("========= test result =========")
    print("Accuracy               : " + str(accuracy))
    print("# of input data        : " + str(inputNum))
    print("total time for detect  : " + str(totalTime))


def __parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--detector", type=str, default="binary", help="initial weights path"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet",
        help="select DNN model for measurement accuracy",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=500, help="number of data for measure accuracy"
    )
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = __parse_opt()
    __parse_to_args(opt)
    __test()
    __get_result()
