#TODO CHANGE LOSS METHOD & FUNCTION
#TODO FIND A WAY TO DISPLAY LOSS

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from PIL import Image
from keras.preprocessing.image import img_to_array
import tensorflow as tf

print("TensorFlow Version: {}".format(tf.__version__))
print("Eager Execution: {}".format(tf.executing_eagerly()))

# importing the dataset
cat_data_path = "D:/Coding/ML-CatsAndDogs/PetImages/Cat"
dog_data_path = "D:/Coding/ML-CatsAndDogs/PetImages/Dog"

onlyfilescat = [f for f in os.listdir(cat_data_path) if os.path.isfile(os.path.join(cat_data_path, f))]
del onlyfilescat[-1]
print("Cat Data Loaded. Working with {0} images".format(len(onlyfilescat)))
onlyfilesdog = [f for f in os.listdir(dog_data_path) if os.path.isfile(os.path.join(dog_data_path, f))]
del onlyfilesdog[-1]
print("Dog Data Loaded. Working with {0} images".format(len(onlyfilesdog)))

shuffle(onlyfilescat)
shuffle(onlyfilesdog)
allfiles = onlyfilescat + onlyfilesdog
shuffle(allfiles)

train_files = []
train_y = []
test_files = []
test_y = []
i=0
train_sample_size = 500
test_sample_size = 50
# change the amount of files it is loading for adjusting set sizes


def checkLabel(label):
    if label == "cat ":
        return True
    else:
        return False


for _file in allfiles[0:train_sample_size*2]:
    train_files.append(_file)
    file_label = _file.find("(")
    train_y.append(checkLabel(_file[0:file_label]))

for _file in onlyfilescat[train_sample_size:test_sample_size+train_sample_size]:
    test_files.append(_file)
    file_label = _file.find("(")
    test_y.append(checkLabel(_file[0:file_label]))

for _file in onlyfilesdog[train_sample_size:test_sample_size+train_sample_size]:
    test_files.append(_file)
    file_label = _file.find("(")
    test_y.append(checkLabel(_file[0:file_label]))

print("Training Dataset Size: %d" %len(train_files))
print("Testing Dataset Size: %d"  %len(test_files))

# IMAGE DIMENSIONS
image_width = 128
image_height = 128

channels = 3
nb_classes = 1

train_dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels), dtype=np.float32)
test_dataset = np.ndarray(shape=(len(test_files), image_height, image_width, channels), dtype=np.float32)
i=0
for _file in train_files:
    if _file[0:file_label] == "cat ":
        img = Image.open(cat_data_path + "/" + _file)
    else:
        img = Image.open(dog_data_path + "/" + _file)
    img = img.resize((128, 128), Image.ANTIALIAS)
    x = img_to_array(img)
    # print("Loading image ", i, " with shape ", x.shape, " at filename ", _file)
    x = x.reshape((image_height, image_width, channels))
    x = (x-128.0) / 128.0
    train_dataset[i] = x
    i += 1
    if i % 200 == 0:
        print("200 new images loaded into training array. Currently at %d images in array." % i)

i = 0
for _file in test_files:
    file_label = _file.find("(")
    if i < len(test_files)/2:
        img = Image.open(cat_data_path + "/" + _file)
    else:
        img = Image.open(dog_data_path + "/" + _file)
    img = img.resize((128, 128), Image.ANTIALIAS)
    x = img_to_array(img)
    x = x.reshape((image_height, image_width, channels))
    x = (x-128.0)/128.0
    test_dataset[i] = x
    i += 1
    if i % 100 == 0:
        print("100 new images loaded into testing array. Currently at %d images in array." % i)

print("All image files loaded into arrays.")

# IMPORTANT
steps = 200
batch_size = 20


class CNN:
    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels])
        # tochange here
        self.conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=98, kernel_size=[2, 2], padding="same",
                                             activation=tf.nn.relu)
        self.pooling_layer_1 = tf.layers.max_pooling2d(self.conv_layer_1, pool_size=[2, 2], strides=2)
        self.flattened_pooling = tf.layers.flatten(self.pooling_layer_1)
        self.dense_layer = tf.layers.dense(self.flattened_pooling, 1024, activation=tf.nn.relu)
        self.dropout = tf.layers.dropout(self.dense_layer, rate=0.4, training=True)
        self.output = tf.layers.dense(self.dropout, num_classes)

        self.choice = tf.argmax(self.output, axis=1)
        self.probability = tf.nn.softmax(self.output)
        self.labels = tf.placeholder(dtype=tf.bool, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)
        self.one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.mean_squared_error(labels=self.one_hot_labels, predictions=self.output)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        self.train_operation = self.optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())


cnn = CNN(image_height, image_width, channels, 2)
plotlist=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    currentStep = 0
    while currentStep < steps:
        print(sess.run((cnn.train_operation, cnn.accuracy_op),
              feed_dict={cnn.input_layer: train_dataset[currentStep:currentStep+batch_size],
                         cnn.labels: train_y[currentStep:currentStep+batch_size]}))
        plotlist.append(sess.run((cnn.train_operation, cnn.accuracy_op),
              feed_dict={cnn.input_layer: train_dataset[currentStep:currentStep+batch_size],
                         cnn.labels: train_y[currentStep:currentStep+batch_size]}))
        currentStep = batch_size + currentStep

plt.plot(plotlist)
plt.ylim(0, 1.0)
plt.show()

# TODO TESTING
