from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.xception import Xception
#from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.resnet import ResNet152
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.mobilenet import MobileNet
import numpy as np
import tensorflow as tf
import pandas as pd
import ast

###############################setting up some variable################################
train_data_dir="/home/t107360144/ML_hw2/theSimpsons-train/train/"
img_height = 112
img_width = 112
batch_size = 256
nb_epochs =10

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)

#################################load train data#######################################
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range = 30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

def _fixup_shape(images, labels, weights):
    images.set_shape([None, None, None, 3])
    labels.set_shape([None, 50]) # I have 19 classes
    weights.set_shape([None])
    return images, labels, weights

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data
train_dataset = tf.data.Dataset.from_generator(lambda:train_generator,(tf.float32, tf.float32))
validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data
valid_dataset = tf.data.Dataset.from_generator(lambda:validation_generator,(tf.float32, tf.float32))
print(train_generator.class_indices)
print(train_generator.labels)

###############################trainning module######################################
#"""Either choose to train module or load the old one"""

#load module from google drive
#from tensorflow import keras
#model = keras.models.load_model('/content/drive/MyDrive/colab_share/model')
#import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
#setup module
#cnn_base = InceptionV3(weights="imagenet",include_top=False,input_shape=(224, 224, 3))
cnn_base = ResNet152(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
#cnn_base = NASNetLarge(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
cnn_base.trainable = True
classify = Sequential()
classify.add(Flatten())
classify.add(Dense(512,activation='relu'))
classify.add(Dropout(0.2))
classify.add(Dense(50,activation='softmax'))
model = Sequential()
model.add(cnn_base)
model.add(classify)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.summary()
#train_dataset = train_dataset.repeat(nb_epochs)
#train_dataset = train_dataset.batch(32)
#train module
checkpoint = ModelCheckpoint("/home/t107360144/resnet", monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(
    train_dataset,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = valid_dataset, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    verbose=1,
    callbacks=callbacks_list)
model.save("/home/t107360144/ML_hw2/resnet")

######################################Predict#############################################
#read test data
test_data_dir = "/home/t107360144/ML_hw2/theSimpsons-test/"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size = 1,
    shuffle = False,
    class_mode = 'categorical')

#predict
Predict = model.predict_generator(test_generator,steps=len(test_generator),verbose=1)
result = np.argmax(Predict,axis=1)#以最後一層dense的50個參數中找出最大值

import ast
my_dict2 = dict((y,x) for x,y in train_generator.class_indices.items())#字典key,value交換
#cl = np.round(Predict)
# print(my_dict2)
# print("one hot encoding")
# print(result)
Y_predict = [my_dict2[k] for k in result]#使用字典將原有的one hot encoding轉回角色名稱
# for i in range(len(result)):
#   Y_predict.append(my_dict2[result[i]])
# print("character name")
# print(Y_predict)
id = test_generator.filenames#讀取檔案名稱
id = [s.replace("test/", "") for s in id]#去除資料夾名test/與副檔名.jpg
id = [s.replace(".jpg", "") for s in id]
idd = [int(s) for s in id]#轉成數字
select_df = pd.DataFrame({"id": idd,"character": Y_predict})
select_df = select_df.sort_values(by=['id'], ascending=True)#數字重新排序
select_df.to_csv('test_resnet.csv',index= False)#存檔成csv
