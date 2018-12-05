import cv2
import os 
import numpy as np
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt


	#Creating image list
Image=[]
Id=[]
x=os.listdir('data/')
for name in x:
	fname='data/'+name
	img=cv2.imread(fname)
	Image.append(img)
	n=name.split('_')
	Id.append(n)
# now we have list of image and also list of ids

# Loading pretrained model of yolo
model = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

# Using model on the image set

coYolo, imYolo = data.transforms.presets.yolo.load_test(Image, short=512)
#coYolo, imYolo are respectively storing is a n-Dimensional array with shape (batch_size,channels,Height ,width) and
#image stored in numpy array format 


#Obtaining class number , confidence score and bounding box description 
class_num, confidence,box = model(coYolo)

images = utils.viz.plot_bbox(imYolo, box[0], confidence[0],
                         class_num[0], class_names=model.classes)



	input_layer = Input(shape=input_img_dim, name="convolutional_Layer_input")

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)

   # model = Model(inputs, x, name='vgg16')
   # model.compile(Adam(lr=00001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
   x=LSTM(100 return_sequences=True, return_state=True , unroll=True)(x)
   x=LSTM(100 return_sequences=True, return_state=True , unroll=True)(x)
   x=LSTM(100 return_sequences=True, return_state=True , unroll=True)(x)
   x=LSTM(100 return_sequences=True, return_state=True , unroll=True)(x)


   model=Model(input_layer,x)
   model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
   model.fit(images, y_train,batch_size=batch_size, epochs=num_epochs,validation_data=(x_val, y_val))



