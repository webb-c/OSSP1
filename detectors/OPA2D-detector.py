'''
OPA2D-detector
'''

import cv2
import numpy as np
from keras.datasets import cifar10
import sys
import copy
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')
import helper
from networks.resnet import ResNet
from networks.lenet import LeNet
from differential_evolution import differential_evolution
# from parse import *

'''
1. attack , origin image 가져오기 
    attack,origin 이미지 복사 -> atack_copied , origin_copied 생성 
2. attack_copied, origin_copied 를 attack : (이미지들이 어떤 클래스 인지를 알아야 함) 하고
    -> reattacked_attack_img,reattacked_origin_img
3. | attack - reattacked_attack_img | , | origin - reattacked_origin_img |
    달라진 pixel 값 찾고 그 픽셀의 dist = max(|r-r'|,|g-g'|,|b-b'|) 를 attack,origin에 대해서 구함
4. 비교해서 더 큰게 origin 이다.  
'''
def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = helper.perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True
    
def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = helper.perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:, target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def color_distance(color1, color2) : # rgb값을 인자로 받아옴 -> color = (r,g,b)
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    dist = max(abs(r1-r2), abs(g1-g2), abs(b1-b2))
    return dist   

def reattack(img_id, input_img, input_class, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        # target_class = target if targeted_attack else self.y_test[img_id, 0]
        target_class = target if targeted_attack else input_class
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x ,dim_y = 32,32
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return predict_classes(xs, input_img, target_class, model, target is None)

        def callback_fn(x, convergence):
            return attack_success(x, input_img, target_class, model, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, input_img)[0]
        prior_probs = model.predict(np.array(input_img))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = input_class
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, class_names, predicted_class)

        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs,attack_image,attack_result.x]
# img , model load

# resnet = ResNet()
resnet = LeNet()
origin_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/sample/original_310.png')
attack_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/sample/attack_310.png')
# parse_result = parse("C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/sample/original_{}.png",)
origin_copied = copy.deepcopy(origin_img)
attack_copied = copy.deepcopy(attack_img)
img_id = 310 #162 # 8625 # 200 # 53 # 520 
origin_sample = np.array(origin_img)
origin_predicted_probs = resnet.predict(np.array([origin_sample]))[0]
origin_predicted_class = np.argmax(origin_predicted_probs)

attack_sample = np.array(attack_img)
attack_predicted_probs = resnet.predict(np.array([attack_sample]))[0]
attack_predicted_class = np.argmax(attack_predicted_probs)

# print("attack_sample",attack_sample.shape())
origin_copied_sample = copy.deepcopy(origin_img)
attack_copied_sample = copy.deepcopy(attack_img)

# attack img
reattacked_origin_result = reattack(img_id,origin_copied_sample ,origin_predicted_class,resnet)

reattacked_attack_result = reattack(img_id,attack_copied_sample, attack_predicted_class,resnet)

# 원본(origin,attack) 과 reattacked 비교 
x_pos,y_pos = reattacked_origin_result[-1][0],reattacked_origin_result[-1][1]
print(int(x_pos),int(y_pos))
origin_rgb = origin_sample[int(x_pos)][int(y_pos)] 
reattack_origin_rgb = reattacked_origin_result[-2][int(x_pos),int(y_pos)]
origin_dist = color_distance(origin_rgb,reattack_origin_rgb)

x_pos,y_pos = reattacked_attack_result[-1][0],reattacked_attack_result[-1][1]
attack_rgb = attack_sample[int(x_pos)][int(y_pos)]
reattack_attack_rgb = reattacked_attack_result[-2][int(x_pos),int(y_pos)]
attack_dist = color_distance(attack_rgb,reattack_attack_rgb)

# dist 큰게 origin 
origin = max(origin_dist,attack_dist)
attack = min(origin_dist,attack_dist)

print("origin img dist ", origin, "attack img dist ",attack)
