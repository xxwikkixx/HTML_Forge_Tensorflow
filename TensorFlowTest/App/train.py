import numpy as np
import os
import cv2
import random
import pickle
import time
from tqdm import tqdm # progress bar library
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

# Directories with the datasets
DATADIR = "/run/media/wikki/01D1F98FEBABCF00/Users/wikki/Desktop/COLLEGE/COLLEGE 2019 14th Spring Penn State/CS 488 Capstone/HTML_Forge_Tensorflow/TensorFlowTest/App/dataset"
DATADIRWin = r"A:\COLLEGE\COLLEGE 2019 14th Spring Penn State\CS 488 Capstone\HTML_Forge_Tensorflow\TensorFlowTest\App\dataset"

# Labels of the images
CATEGORY = ['Header', 'Title', 'Plain_Image_Gallery',
              'Paragraph', 'IMG_Top_Text_Bottom', 'IMG_Right_Text_Left',
              'IMG_Left_Text_Right', 'ImageFlipWithPreview', 'Image_Flip', 'Footer']

# CATEGORY = ['Header', 'Title', 'Plain_Image_Gallery','Paragraph', 'IMG_Top_Text_Bottom', 'IMG_Right_Text_Left']

training_data = []

def create_train_data():
    for categories in CATEGORY: # go through the categories and their folders
        path = os.path.join(DATADIRWin, categories) # Path to the labels directories
        classification_num = CATEGORY.index(categories) # get the classification number for the two categories we are training
        # iterates through the images
        for img in tqdm(os.listdir(path)): # iterate over each image per header and footer
            try:
                # convert imgaes to an array
                    #joins the path to the image
                        #convert the image to grayscale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (90, 90))
                training_data.append([img_array, classification_num])
            except Exception as e:
                pass


if __name__ == "__main__":

    create_train_data()
    print(len(training_data))

    #shuffle the training data list
    random.shuffle(training_data)
    for sample in training_data:
        print(sample[1])

    X = [] # feature set
    y = [] # Labels set

    for features, label in training_data:
        X.append(features)
        y.append(label)

    # -1 - catch all the features from the images
    X = np.array(X).reshape(-1, 90, 90,  1)

    pickle_out = open("X.pickle", "wb") # makes a file for X
    pickle.dump(X, pickle_out) #outputs the data
    pickle_out.close()

    pickle_out = open("y.pickle", "wb") # makes a file for y
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle", 'rb')
    y = pickle.load(pickle_in)

    X = X/255

    dense_layers = [0]
    layer_sizes = [64]
    conv_layers = [3]

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                print(NAME)

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )

                model.fit(X, y,
                          batch_size=32,
                          epochs=75,
                          validation_split=0.3,
                          callbacks=[tensorboard])

    model.save(r'A:\COLLEGE\COLLEGE 2019 14th Spring Penn State\CS 488 Capstone\HTML_Forge_Tensorflow\TensorFlowTest\App\64x3-CNN.model')