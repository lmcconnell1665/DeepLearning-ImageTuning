"""
BZAN 544 - Deep Learning - Group Project 4
Tuning Neural Network for Images
Aileen Barry, Luke McConnell, and Harry Zheng
"""

#The goal is to build the best possible model to predict the class of the image as measured on the test set.

# When tuning across the tuning grid, the team is splitting up as follows:
# Luke 0:8
# Harry 9:16
# Aileen 17:24

########################
#### IMPORT MODULES ####
########################
import ssl
import csv
import tensorflow as tf
import itertools as it


########################
##### GET SSL CERT #####
########################
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


#######################
##### IMPORT DATA #####
#######################
fashion_data = tf.keras.datasets.fashion_mnist
(X_train, y_train),(X_test, y_test) = fashion_data.load_data()

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

#Let's look at the first image:
class_names[y_train[0]]
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap = 'binary')

#How many images in train and their size?
X_train.shape
# Images: 60000
# Size: 28 x 28 pixels

#scaling the features:
X_train = X_train / 255.0
X_test = X_test / 255.0

#Let's build a multi-class classifier
#Unique classes?
import numpy as np
np.unique(y_train.astype(np.uint8))
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)


############################
##### CREATE TUNE GRID #####
############################
tune_filter_number = [64, 128, 256]
tune_stride = [1, 2, 4, 7] #must max out at 7
tune_zero_padding = ['same', 'valid']

grid = []

grid.extend(tuple(it.product(tune_filter_number,
                            tune_stride,
                            tune_zero_padding)))


#######################################
#### SPECIFY ARCHITECTURE AND TUNE ####
#######################################

#As we go deeper toward the output it makes sense to increase the number of filters
#since the number of low-level features is often low (circles, lines, ...), but there are many
#different ways to combine low-level features into higher-level features (nose, ear, ...).
#It is common to double the number of filters after each pooling layer (e.g., 64, 128, 256) as the pooling layer #divides each spatial dimension by 2. This prevents growth in the computational load.

results = []

for i in grid:
    #shape is 28,28,1 because grayscale (single color channel)
    inputs = tf.keras.layers.Input(shape=(28,28,1), name='input') 
    #Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
    x = tf.keras.layers.Conv2D(filters=i[0],kernel_size = 7, strides = i[1], padding = i[2], activation = "relu")(inputs)
    #MaxPooling2D: pool_size is window size over which to take the max
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2*i[1], padding = "valid")(x)
    x = tf.keras.layers.Conv2D(filters=2*i[0],kernel_size = 3, strides = i[1], padding = i[2], activation = "relu")(x)
    x = tf.keras.layers.Conv2D(filters=2*i[0],kernel_size = 3, strides = i[1], padding = i[2], activation = "relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2*i[1], padding = "valid")(x)
    x = tf.keras.layers.Conv2D(filters=4*i[0],kernel_size = 3, strides = i[1], padding = i[2], activation = "relu")(x)
    x = tf.keras.layers.Conv2D(filters=4*i[0],kernel_size = 3, strides = i[1], padding = i[2], activation = "relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2*i[1], padding = "valid")(x)
    #dense layers expect 1D array of features for each instance so we need to flatten.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    x = tf.keras.layers.Dense(64, activation = 'relu')(x)
    yhat = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = inputs, outputs = yhat)
    model.summary()
    
    #Compile model
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.SGD(lr = 0.001))

    #Fit model
    model.fit(x=X_train,y=y_train, batch_size=1, epochs=2) 

    #Compute multiclass accuray
    yhat = model.predict(x=X_test)
    yhat_sparse = [int(np.where(yhat_sub ==np.max(yhat_sub))[0]) for yhat_sub in yhat]
    results.append([i[0], i[1], i[2], sum(yhat_sparse == y_test) / len(y_test)])
     
    #Save results
    file = open('results.txt', 'a')
    file.write(str(results) + '\n')
    file.close()
    

    