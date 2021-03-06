#Importing all the neccessary Libraries
#Importing the neccessary Libraries
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

img=cv2.imread('outfile.jpg')

#Creating the pretrained Restnet Model
resnet152 = models.resnet152(pretrained=True)
modules=list(resnet152.children())[:-1] #Striped of the fully connected layer 
resnet152=nn.Sequential(*modules)

#Model is set to the evaluation mode 
resnet152.eval()

# Transform the image 
scaler = transforms.Scale((224, 224)) 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
imagespath = 'D:\bdaproject\fashion-dataset\fashion-dataset\images'

Features = []
i = 0
#Extracting the features
for img_file in os.listdir(imagespath):
	img = Image.open(os.path.join(imagespath,img_file))
	if len(img.getbands()) !=3:
		continue
	t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
	feature = resnet152(t_img)[0,:,0,0]
	Features.append(feature.detach().numpy())
	if (i+1)%100 == 0:
		print(i, " Features Extracted")
		np.save('Features150.npy',Features)
	i += 1
np.save('Features150.npy',Features)
print(len(Features))
