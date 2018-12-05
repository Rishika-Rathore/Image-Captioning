#Image captioning Model:
This Model has three parts specifically :
			1.Creating bounding box
			2.Creating a Dense Representation of the image
			3.Creating Caption for the input images 
##Breif Description of Files :
**model.py** It contains all the files of different networks used, put together in order at one place.**bounding_box.py** It 
contains code for creating bounding box around the elements present in the images.**preprocess.py** It contaims the code necessary to 
preprocess the data before being fed to the network.


## Dataset :
COCO is a large-scale object detection, segmentation, and captioning dataset. We have used 2014 datatset here.
 Five annotations are available for each image .Images contain complex everyday scenes containing common objects in their natural context. Objects are labeled using per-instance segmentations to aid in
precise object localization. Dataset contains photos of 91 objects types that would be easily recognizable . With a total of 2.5 million 
labeled instances in 328k images.

## Dependencies:
	*Python 3.x
	*Opencv
	*Numpy
	*Keras
	
 
