import tensorflow as tf
import keras
import sys
import pathlib
import matplotlib.pyplot as plt
sys.path.append('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras')

import os
import numpy as np
import random
import cv2
from PIL import Image
from keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers, regularizers

from networks.train_plot import PlotLearning

random.seed(42)

class ResNetforOSP:
    def __init__(self, epochs=200, batch_size=128, load_weights=True):
        self.name               = 'resnet'
        self.model_filename     = 'C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/detectors/binary/model_test.h5'
        
        self.stack_n            = 2  
        self.num_classes        = 2
        self.img_rows, self.img_cols = 32, 32
        self.img_channels       = 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 18000 // self.batch_size
        self.weight_decay       = 0.0001
        self.log_filepath       = 'C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/detectors/binary/log'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)
    
    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train,x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test

    def scheduler(self, epoch):
        if epoch < 80:
            return 0.1
        if epoch < 150:
            return 0.01
        return 0.001

    def residual_network(self, img_input,classes_num=1,stack_n=2):
        def residual_block(intput,out_channel):

            pre_bn   = BatchNormalization()(intput)
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            bn_1   = BatchNormalization()(conv_1)
            relu1  = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)            
            
            block = add([intput,conv_2])
            return block

        '''
        아래 주석 수정해줘야함!!
        '''
        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 3 by default, total layers = 32
        
        # input: 32x32x3 output: 32x32x16 - conv(16)
        x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        for _ in range(stack_n): # residual
            x = residual_block(x,16)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # input: 32x32x16 output: 32x32x32 - conv(32)
        x = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        # input: 32x32x32 output: 16x16x32 - max pooling
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        
        for _ in range(stack_n): # residual
            x = residual_block(x,32)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # input: 16x16x32 output: 16x16x64 - conv
        x = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        # input: 16x16x64 output: 8x8x64 - max pooling
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        
        for _ in range(stack_n): # residual
            x = residual_block(x,64)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # input: 8x8x64 output: 8x8x32 - conv(1*1)
        x = Conv2D(filters=32,kernel_size=(1,1),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # input: 8x8x32 output: 1x1x32 - GAP
        x = GlobalAveragePooling2D()(x)

        # input: 32 output: 1 - FC
        x = Dense(classes_num, activation='softmax',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        return x


    def train(self):
        # load data
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #y_train = keras.utils.to_categorical(y_train, self.num_classes)
        #y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        (x_train, y_train), (x_test, y_test) = self.get_dataset()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # build network
        img_input = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
        output    = self.residual_network(img_input,self.num_classes,self.stack_n)
        resnet    = Model(img_input, output)
        resnet.summary()

        # set optimizer
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # set callback
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint(self.model_filename, 
                monitor='val_loss', verbose=0, save_best_only= True, mode='auto')
        # plot_callback = PlotLearning()
        cbks = [change_lr,tb_cb,checkpoint]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=0.125,
                                    height_shift_range=0.125,
                                    fill_mode='constant',cval=0.)

        datagen.fit(x_train)

        # start training
        resnet.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                            steps_per_epoch=self.iterations,
                            epochs=self.epochs,
                            callbacks=cbks,
                            validation_data=(x_test, y_test))
        resnet.save(self.model_filename)

        self._model = resnet
        self.param_count = self._model.count_params()

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = self.get_dataset()
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        # color preprocessing
        _, x_test = self.color_preprocessing(x_train, x_test)
        return self._model.evaluate(x_test, y_test, verbose=1)[1]
    
    def get_dataset(self):
        attack_path = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/lenet_sample/attack"
        attack_fileList = [f for f in os.listdir(attack_path) if f.endswith('.png')]
        attack_fileList_1 = attack_fileList[:len(attack_fileList)//2]
        attack_fileList_2 = attack_fileList[len(attack_fileList)//2:]
        img_list = []

        for f in attack_fileList_1 :
            temp = Image.open(os.path.join(attack_path, f))
            img_list.append(image.img_to_array(temp))
            temp.close()
        for f in attack_fileList_2 :  
            temp = Image.open(os.path.join(attack_path, f))
            img_list.append(image.img_to_array(temp))
            temp.close()
        attack_label = np.zeros(shape=(len(attack_fileList),), dtype=np.int8)
        
        origin_path = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/lenet_sample/original"
        origin_fileList = [f for f in os.listdir(origin_path) if f.endswith('.png')]
        origin_fileList_1 = origin_fileList[:len(origin_fileList)//2]
        origin_fileList_2 = origin_fileList[len(origin_fileList)//2:]

        for f in origin_fileList_1 :
            temp = Image.open(os.path.join(origin_path, f))
            img_list.append(image.img_to_array(temp))
            temp.close()
        for f in origin_fileList_2 :  
            temp = Image.open(os.path.join(origin_path, f))
            img_list.append(image.img_to_array(temp))
            temp.close()
        origin_label = np.ones(shape=(len(origin_fileList),), dtype=np.int8)
        
        x = img_list
        y = np.concatenate((attack_label, origin_label)).tolist()
        
        temp_list = list(zip(x, y))
        random.shuffle(temp_list)
        x, y = zip(*temp_list)
        x, y = np.array(list(x)), np.array(list(y))
        
        split_index = int(len(x)*0.9)
        x_train, x_test = np.split(x, [split_index])
        
        split_index = int(len(y)*0.9)
        y_train, y_test = np.split(y, [split_index])
        
        return (x_train, y_train), (x_test, y_test)
    
    def get_dataset_usetf(self): 
        # cifar랑 비교
        
        dir_path = "C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/lenet_sample/"
        # img_gen = ImageDataGenerator(validation_split=0.2)
        
        #train_set = img_gen.flow_from_directory(directory=dir_path, target_size=(self.img_rows, self.img_cols), class_mode='binary', subset='training', batch_size=self.batch_size)
        #vaild_set = img_gen.flow_from_directory(directory=dir_path, target_size=(self.img_rows, self.img_cols), class_mode='binary', subset='validation',batch_size=self.batch_size)
        train_set = tf.keras.utils.image_dataset_from_directory(
            directory=dir_path,
            image_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            validation_split=0.3,
            subset='training',
            seed=1015
        )
        test_set = tf.keras.utils.image_dataset_from_directory(
            directory=dir_path,
            image_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            validation_split=0.3,
            subset='validation',
            seed=1015
        )
        #확인
        class_names = train_set.class_names
        print(class_names)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        i=0
        for images, labels in train_set:
            x_train.append(images)
            y_train.append(labels)
            i+=1
            print(i)
        
        i=0
        for images, labels in test_set:
            x_test.append(images)
            y_test.append(labels)
            i+=1
            print(i)
        
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)
        
        return (x_train, y_train), (x_test, y_test)
    
def is_attack(img):
    conf = model.predict(img)
    predict = np.argmax(conf)
    if predict == 0 : return True
    else : return False

model = ResNetforOSP()

# testing용 코드
test_img = cv2.imread('C:/Users/CoIn240/VSCpython/2023OSP/one-pixel-attack-keras/resnet_sample/original/1037_automobile.png', cv2.IMREAD_COLOR)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
print(is_attack(test_img))

# predict_origin = model.predict(np.array(origin_img))[0]
# predict_attack = model.predict(np.array(attack_img))[0]
# predict_origin = ['{:.4f}'.format(num) for num in predict_origin]
# predict_attack = ['{:.4f}'.format(num) for num in predict_attack]

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(origin_img)
# axes[0].axis('off')
# axes[0].annotate(str(predict_origin), xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -30), textcoords='offset points', fontsize=12, ha='center')

# axes[1].imshow(attack_img)
# axes[1].axis('off')
# axes[1].annotate(str(predict_attack), xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -30), textcoords='offset points', fontsize=12, ha='center')

# plt.tight_layout()
# plt.show()
