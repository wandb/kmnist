#!/usr/bin/env python
# -*- coding: utf-8 -*-

# cnn_kmnist.py
#----------------
# Train a small CNN to identify 10 Japanese characters in classical script
# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function

#import sys
#sys.path.append('/home/stacey/KerasDropconnect')
#from ddrop.layers import DropConnect

import tensorflow as tf
from tensorflow.keras import layers
import argparse
import numpy as np
import os
from utils import load_train_data, load_test_data, load, KmnistCallback
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
print(K.image_data_format())
# default configuration / hyperparameter values
# you can modify these below or via command line
MODEL_NAME = ""
DATA_HOME = "./dataset" 
BATCH_SIZE = 128
EPOCHS = 10
L1_SIZE = 150
L2_SIZE = 400
L3_SIZE = 10
DROPOUT_1_RATE = 0.5
DROPOUT_2_RATE = 0.5
FC1_SIZE = 10
NUM_CLASSES = 10
#NUM_CLASSES_K49 = 49

# input image dimensions
img_rows, img_cols = 28, 28
# ground truth labels for the 10 classes of Kuzushiji-MNIST Japanese characters 
LABELS_10 =["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"] 
LABELS_49 = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち",
"つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め"
"も","や","ゆ","よ","ら","り","る","れ","ろ","わ","ゐ","ゑ","を","ん","ゝ"]


def build_dc_model(args, input_shape):
  model = tf.keras.Sequential()
  #model.add(layers.Conv2D(args.l1_size, kernel_size=(3,3), activation='relu', input_shape=input_shape))
  model.add(layers.Flatten(input_shape=input_shape))
  model.add(layers.Dense(args.l1_size, activation="relu"))
  model.add(layers.Dropout(args.dropout_1))
  model.add(layers.Dense(args.l2_size, activation="relu"))
  model.add(layers.Dropout(args.dropout_2))
  model.add(layers.Dense(args.num_classes, activation = 'softmax'))
  learn_opt = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)
  model.compile(loss="categorical_crossentropy", optimizer=learn_opt, metrics=["accuracy"])
  return model

def build_fc_model(args):
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(args.l1_size, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
 

  model.add(layers.Conv2D(args.l2_size, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(args.dropout_1))
  model.add(layers.Flatten())
  model.add(layers.Dense(args.fc1_size, activation='relu'))
  model.add(layers.Dropout(args.dropout_2))
  model.add(layers.Dense(args.num_classes, activation='softmax'))

  model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

# next improvements from SOTA:
# 1) crop to random 24 x 24 loc
# 2) rotate & scale these
# 3) 700-200-100 schedule
# 4) subtract image mean
def build_sota(args, input_shape):
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(args.dropout_1))
  model.add(layers.Flatten())
  model.add(layers.Dense(args.l1_size, activation="relu"))
  model.add(layers.Dropout(args.dropout_2))
  #model.add(layers.Dense(args.l2_size, activation="relu"))
  #model.add(layers.Dropout(args.dropout_2))
  model.add(layers.Dense(args.num_classes, activation = 'softmax'))
  learn_opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
   

  #model.add(layers.Dense(128, activation='relu'))
  #model.add(layers.Dropout(0.5))
  #model.add(layers.Dense(args.num_classes, activation='softmax'))

  model.compile(loss="categorical_crossentropy",
                optimizer=learn_opt,
                metrics=['accuracy'])
  return model
 
def build_base_model(args, input_shape):
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(args.l1_size, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
  model.add(layers.Conv2D(args.l2_size, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(args.dropout_1))
  model.add(layers.Flatten())
  model.add(layers.Dense(args.fc1_size, activation='relu'))
  model.add(layers.Dropout(args.dropout_2))
  model.add(layers.Dense(args.num_classes, activation='softmax'))

  model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])
  return model

def train_cnn(args):
  # initialize wandb logging to your project
  wandb.init()
  config = {
    "model_type" : "fc_dropout",
    "batch_size" : args.batch_size,
    "num_classes" : args.num_classes,
    "epochs" : args.epochs,
    "l1_size": args.l1_size,
    "l2_size" : args.l2_size,
    "l3_size" : L3_SIZE,
    "dropout_1" : args.dropout_1,
    "dropout_2" : args.dropout_2,
    "fc1_size" : args.fc1_size,
    "zoom" : 0.0,
    "shear" : 10.0
  }
  wandb.config.update(config)

  # Load the data form the relative path provided
  x_train, y_train = load_train_data(args.data_home)
  x_test, y_test = load_test_data(args.data_home)

  # reshape to channels last
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  if args.quick_run:
    MINI_TR = 6000
    MINI_TS = 1000
    x_train = x_train[:MINI_TR]
    y_train = y_train[:MINI_TR]
    x_test = x_test[:MINI_TS]
    y_test = y_test[:MINI_TS]
 
  N_TRAIN = len(x_train)
  N_TEST = len(x_test)
  wandb.config.update({"n_train" : N_TRAIN, "n_test" : N_TEST})
  print('{} train samples, {} test samples'.format(N_TRAIN, N_TEST))

  # Convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

  model = build_sota(args, input_shape)
  # try this with augmentation??
  # basic data augmentation
  train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    #samplewise_center=True,
    #samplewise_std_normalization=True,
    #zoom_range=0.15,
  #  horizontal_flip=True)
    shear_range=10)
  test_datagen = ImageDataGenerator()

  #train_datagen.fit(x_train)
# model.fit(x_train, y_train,
#            batch_size=args.batch_size,
#            epochs=args.epochs,
#            verbose=1,
#            validation_data=(x_test, y_test),
#            callbacks=[KmnistCallback(), WandbCallback(data_type="image", labels=LABELS_10)])
   # fits the model on batches with real-time data augmentation:
  model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=args.batch_size),
                      steps_per_epoch=N_TRAIN / args.batch_size,
                      epochs=args.epochs,
                      validation_data = test_datagen.flow(x_test, y_test, batch_size=args.batch_size),
                      validation_steps = N_TEST/args.batch_size,
                      callbacks=[KmnistCallback(), WandbCallback(data_type="image", labels=LABELS_10)])
#, validation_data=test_datagen.flow(x_test, y_test, batch_size=args.batch_size))])

  train_score = model.evaluate(x_train, y_train, verbose=0)
  test_score = model.evaluate(x_test, y_test, verbose=0)
  print('Train loss:', train_score[0])
  print('Train accuracy:', train_score[1])
  print('Test loss:', test_score[0])
  print('Test accuracy:', test_score[1])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "--data_home",
    type=str,
    default=DATA_HOME,
    help="Relative path to training/test data")
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="batch size")
  parser.add_argument(
    "--dropout_1",
    type=float,
    default=DROPOUT_1_RATE,
    help="dropout rate for first dropout layer")
  parser.add_argument(
    "--dropout_2",
    type=float,
    default=DROPOUT_2_RATE,
    help="dropout rate for second dropout layer")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="number of training epochs (passes through full training data)")
  parser.add_argument(
    "--fc1_size",
    type=int,
    default=FC1_SIZE,
    help="size of fully-connected layer")
  parser.add_argument(
    "--l1_size",
    type=int,
    default=L1_SIZE,
    help="size of first conv layer")
  parser.add_argument(
    "--l2_size",
    type=int,
    default=L2_SIZE,
    help="size of second conv layer")
  parser.add_argument(
    "--num_classes",
    type=int,
    default=NUM_CLASSES,
    help="number of classes (default: 10)")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "--quick_run",
    action="store_true",
    help="train quickly on a tenth of the data")   
  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # create run name from command line
  if args.model_name:
    os.environ['WANDB_DESCRIPTION'] = args.model_name

  train_cnn(args)

