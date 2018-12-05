#Preprocessing of Captions and images in a format to feed into the network

import json
from pprint import pprint

with open('captions_val2014.json') as f:
    data = json.load(f)

 #Now we will extract 'id' column as id of the image and 'caption ' column to get cation of the image.

item=data['annotations']
#
ids=[]
captions=[]

for i in item:
	ids.append(item['id'])
	captions.append(item['captions'])
