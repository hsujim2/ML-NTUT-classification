##########################import###################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import numpy as np
import tensorflow as tf
import pandas as pd
import ast

######################setting up some variables####################
train_data_dir="/home/t107360144/ML_hw2/theSimpsons-train/train/"
img_height = 112
img_width = 112
batch_size = 128
nb_epochs = 16

#To prevent keras from using all vram
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

####################read train data###############################
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data
print(train_generator.class_indices)
print(train_generator.labels)

#Either choose to train module or load the old one

#load module from google drive
from tensorflow import keras
model1 = keras.models.load_model('/home/t107360144/ML_hw2/xception')
model2 = keras.models.load_model('/home/t107360144/ML_hw2/densenet')
model3 = keras.models.load_model('/home/t107360144/ML_hw2/inceptionresnetv2')
model4 = keras.models.load_model('/home/t107360144/ML_hw2/resnet')
#import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


#######################setup module##############################
#cnn_base = InceptionV3(weights="imagenet",include_top=False,input_shape=(224, 224, 3))
#use pre-trained model for input
# cnn_base = Xception(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
# cnn_base.trainable = False#keep pre-trained model parameter
# #add some dense layers for output
# classify = Sequential()
# classify.add(Flatten())
# classify.add(Dense(512,activation='relu'))
# classify.add(Dropout(0.1))
# classify.add(Dense(50,activation='sigmoid'))
# model = Sequential()#combine two model
# model.add(cnn_base)
# model.add(classify)
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# model.summary()

#####################train module#########################
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs,
#     verbose=1)
# model.save("/home/hsujim/model")

#####################read test data########################
test_data_dir = "/home/t107360144/ML_hw2/theSimpsons-test/"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size = 1,
    shuffle = False,#important!! Test data will misorder if suffle is True, which is keras's default
    class_mode = 'categorical')

Predict1 = model1.predict_generator(test_generator,steps=len(test_generator),verbose=1)

Predict2 = model2.predict_generator(test_generator,steps=len(test_generator),verbose=1)

Predict3 = model3.predict_generator(test_generator,steps=len(test_generator),verbose=1)

#Predict4 = model4.predict_generator(test_generator,steps=len(test_generator),verbose=1)

Predict = Predict1 * 0.97364 + Predict2 * 0.95986 + Predict3 * 0.97449
Predict = Predict / 3
result = np.argmax(Predict,axis=1)#find the maximum value in 50 categorical

my_dict2 = dict((y,x) for x,y in train_generator.class_indices.items())#exchange the keys and values in dictionary
#datagen will use one hot encoding to encode category
Y_predict = [my_dict2[k] for k in result]#It needs to use dictionary to decoding and get charactor's name

id = test_generator.filenames#read filenames
id = [s.replace("test/", "") for s in id]#remove folder's name test/
id = [s.replace(".jpg", "") for s in id]#remove file extension .jpg
idd = [int(s) for s in id]#str to int
select_df = pd.DataFrame({"id": idd,"character": Y_predict})
#files will arrange in ASCII code as default(1,10,100,1000,10000,11......)
select_df = select_df.sort_values(by=['id'], ascending=True)#sort filename with int datatype(1,2,3...)
select_df.to_csv('combine_test.csv',index= False)#save to csv
