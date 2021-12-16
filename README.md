# ML-NTUT-classification
***This is the homework2 in Machining Learning class @ NTUT.<br>***
***It's a kaggle compititation for 50 Simpson charactors prediction.<br>***
***It's a big dataset, colab can't handle it, so I ran it locally.<br>***
student:電子四甲 107360144  許智棋

## Environment<br>
### hardware<br>
>cpu: i7 8700<br>
>ram: DDR4 32G 3200<br>
>gpu: RTX2060-6G<br>

### Software
>Windows 11 Insider Preview build 22518<br>
>with WSL2 and WSLg installed<br>
>GPU_driver 496.76 with WSL support<br>
>Ubuntu 20.04.3 LTS<br>
>anaconda 4.10.3<br>
>tensorflow-gpu 2.4.1<br>
>tensorflow-estimator 2.6.0<br>
>cudnn 7.6.5<br>
>cudatoolkit 10.1.243<br>
>Python 3.9.7<br>
>Pillow 8.4.0<br>
>Pandas 1.3.4<br>
## Set up environment
Download [Nvidia Graphic Card Driver with WSL support](https://developer.nvidia.com/cuda/wsl)<br>
[Enable hyper-v features](https://docs.microsoft.com/zh-tw/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v)<br>
Open Windows Terminal<br>
### Install WSL

    wsl --install
    wsl --set-default-version 2

[Install ubuntu 20.04 in Windows store.](https://www.microsoft.com/zh-tw/p/ubuntu-2004-lts/9n6svws3rx71?activetab=pivot:overviewtab)<br>
Open Ubuntu in Windows Terminal([WSLg](https://github.com/microsoft/wslg) is optional).<br>
### Install anaconda

    cd
    sudo apt-get install curl
    curl –O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
    bash Anaconda3-2020.02-Linux-x86_64.sh
    
Press Enter many times and wait for installation.<br>
Close WSL and relogin.<br>
### Setting up virtual environment

    conda create –n your_vir_env_name
    conda activate your_vir_env_name
    conda install tensorflow-gpu=2.4.1
    conda install tensorflow-estimator=2.6.0
    conda install cudnn=7.6.5
    conda install pillow
    conda install panda

### Checking Environment
Check if GPU driver works.<br>

    nvidia-smi

![](https://i.imgur.com/EkHLnWe.png"Results")
Check if tensorflows-gpu works.<br>
```Python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print("################################################")
print(cuda, cudnn)
```
![](https://i.imgur.com/ul9WnJC.png"Results")
Now your pc is ready to train tensorflow model.<br>
## Run Python Code(未完成)

    sudo apt install git
    pip3 install kaggle
    
Download kaggle api from kaggle->account.<br>
Put the json file to ~/.kaggle.<br>

    git clone https://github.com/hsujim2/ML-NTUT-classification
    cd ML-NTUT-classification
    kaggle competitions download -c machine-learningntut-2021-autumn-classification
    unzip -qq machine-learningntut-2021-autumn-classification.zip
    python3 hw2.py
    
The Result will save as test.csv in ML-NTUT-classification folder.<br>

## Python Code
### Import
```Python
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
```
### Setting up variables
Setting image size(it can resize using Imagedatagen), epochs and batch size.<br>
Setting batch_size to 256 will ran out of vram and stop training.<br>

```Python
######################setting up some variables####################
train_data_dir="/home/hsujim/theSimpsons-train/train/"
img_height = 224
img_width = 224
batch_size = 128
nb_epochs = 16

#To prevent keras from using all vram
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
```

### Read training data
Use image datagen to load image, it will read all folder and label them automatically.<br>
Also, it can help split validation data and rotate, shear, zoom image randomly.<br>

```Python
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
```

### Setup and trainning
I use [Xception](https://www.cnblogs.com/sariel-sakura/p/13402056.html) pretrained model and two Dense layers as output.<br>
```Python
#######################setup module##############################
#cnn_base = InceptionV3(weights="imagenet",include_top=False,input_shape=(224, 224, 3))
#use pre-trained model for input
cnn_base = Xception(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
cnn_base.trainable = False#keep pre-trained model parameter
#add some dense layers for output
classify = Sequential()
classify.add(Flatten())
classify.add(Dense(512,activation='relu'))
classify.add(Dropout(0.1))
classify.add(Dense(50,activation='sigmoid'))
model = Sequential()#combine two model
model.add(cnn_base)
model.add(classify)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.summary()

#####################train module#########################
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    verbose=1)
model.save("/home/hsujim/model")
```

### Predict
Setting shuffle = True when using datagen for test data.<br>
Shuffle will change the order of data in order to get better trainning result.<br>
But the order of filenames won't be change at the same time.<br>
```Python
#######################read test data#########################
test_data_dir = "/home/hsujim/theSimpsons-test/"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size = 1,
    shuffle = False,#important!! Test data will misorder if suffle is True, which is keras's default
    class_mode = 'categorical')
    
###########################Pridect############################
Predict = model.predict_generator(test_generator,steps=len(test_generator),verbose=1)
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
select_df.to_csv('test.csv',index= False)#save to csv
```
## Result
### Category
There are 77560 images in training dataset and 19369 images in validation dataset with 50 classes.<br>
![](https://i.imgur.com/tPWBBcc.png)
### Module
![](https://i.imgur.com/61xdSVt.png)
### Training
2060 needs above 4 hours to train this module.<br>
![](https://i.imgur.com/svHAQsZ.jpg)
### Kaggle result
![](https://i.imgur.com/2fBmRzf.png)
