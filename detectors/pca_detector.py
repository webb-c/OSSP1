'''
PCA detector
'''
import keras
import cv2
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import cifar10
import sys
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')
import helper
import joblib
from networks.resnet import ResNet

resnet = ResNet()
pca = joblib.load('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/detectors/pca/pca_model.pkl')

def pca_train() :
    (x_train, _), _ = cifar10.load_data()
    x_train_vector = x_train.reshape(50000, -1)
    pca = PCA(n_components=768)
    pca.fit(x_train_vector)
    
    joblib.dump(pca, 'C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/detectors/pca/pca_model.pkl')

def pca_encode_decode(origin):
    origin_vector = origin.reshape(1, -1)
    encoded = pca.transform(origin_vector)
    decoded_vector = pca.inverse_transform(encoded)
    decoded = decoded_vector.reshape(origin.shape).astype(np.uint8)
    
    # 확인용 
    # helper.plot_image(origin)
    # helper.plot_image(decoded)
    
    return decoded

def calculate_difference(origin, decoded) :
    confidence_original = resnet.predict(origin)[0]
    confidence_pca = resnet.predict(decoded)[0]
    diff = confidence_original - confidence_pca
    abs_diff = np.abs(diff)
    diff_sum = np.sum(abs_diff)/2
    return diff_sum

def is_attack(image, threshold=0.000095) :   # threshold 값만 나중에 잘 정해보기
    value = get_value(image)
    if value > threshold : 
        return True
    else : return False

def get_value(img) :
    image = np.array(img)
    decoded = pca_encode_decode(image)
    result = calculate_difference(image, decoded)
    return result

# 테스트
# origin_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/original/19_frog.png', cv2.IMREAD_COLOR)
# attack_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/attack/19_frog.png', cv2.IMREAD_COLOR)
# origin = np.array(origin_img)
# attack = np.array(attack_img)

# decoded = pca_encode_decode(origin)
# o_result = calculate_difference(origin, decoded)
# decoded = pca_encode_decode(attack)
# a_result = calculate_difference(attack, decoded)

# print("original:", o_result, "attack:", a_result)