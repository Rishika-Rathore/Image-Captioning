#Preprocessing Creating Bounding Box
import cv2
import os
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




