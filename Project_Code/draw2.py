#Import All the Neccesary Libraries
import cv2
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
#Images are loaded 
imagespath = 'D:/bdaproject/fashion-dataset/images'
Features = np.load('Features50_1.npy')
#Normalizing the Images 
for i in range(Features.shape[0]):
	Features[i] = Features[i] / norm(Features[i])
print(Features.shape)

images2display = 12
#list = os.listdir(imagespath) # dir is your directory path
#list = np.loadtxt('names.txt', dtype=)
with open('names.txt', 'r') as file:
    list = file.read().replace('\n', ' ')
list = list.split()

number_images = Features.shape[0]#len(list)

# randomly selects indeces of images 
indeces = np.random.uniform(0,number_images,images2display).astype(int) 

images = []
for i in range(images2display):#loading the images to display
	#print(cv2.imread(os.path.join(imagespath,list[indeces[i]])).shape)
	images.append(cv2.resize(cv2.imread(os.path.join(imagespath,list[indeces[i]])),(120,160)))

#aligning the images in horizontal axis to display 
Hori1 = np.concatenate((images[0], images[1],images[2]), axis=1) 
Hori2 = np.concatenate((images[3], images[4],images[5]), axis=1)  
Hori3 = np.concatenate((images[6], images[7],images[8]), axis=1)
Hori4 = np.concatenate((images[9], images[10],images[11]), axis=1)
  
print(Hori1.shape)

#Fitting the extracted Features 
neighbors = NearestNeighbors(n_neighbors=12,algorithm='auto',metric='euclidean')
#print(Features[0])
#print(Features[1])
#print(Features.shape)
neighbors.fit(Features)


img_back = np.zeros((160,360,3), np.uint8)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 3
fontColor              = (0,0,255)
thickness              = 10
lineType               = 2

cv2.putText(img_back,'Back', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

Verti = np.concatenate((Hori1, Hori2,Hori3,Hori4), axis=0) #aligning the horizonatlly aligned images in vertical axes 
  
cv2.imshow('Select An Item', Verti)

Main_page = 0
Count = 0 #activates once a click is made on selction page
def on_click(event, x, y, p1, p2):
	global Main_page
	global Count
	global indeces
	global images
	if event == cv2.EVENT_LBUTTONDOWN:
		#cv2.destroyAllWindows()
		if Main_page == 0:
			i = x // 120
			j = y //160
			
			if y<=800 and y>642:
				Main_page = 1

			if Count == 1:
				return
			Count = 1
			
			index = i + j *3
			#Caluculting the nearest neighbours based on the features extracted 
			
			distances,indeces_found = neighbors.kneighbors([Features[indeces[index]]])

			Found_images = []
			for i in range(12):
				Found_images.append(cv2.resize(cv2.imread(os.path.join(imagespath,list[indeces_found[0][i]])),(120,160)))

			Hori1 = np.concatenate((images[index], Found_images[1],Found_images[2]), axis=1) #aligning the images in horizontal axis to display 
			Hori2 = np.concatenate((Found_images[3], Found_images[4],Found_images[5]), axis=1)  
			Hori3 = np.concatenate((Found_images[6], Found_images[7],Found_images[8]), axis=1)
			Hori4 = np.concatenate((Found_images[9], Found_images[10],Found_images[11]), axis=1)

			Verti = np.concatenate((Hori1, Hori2,Hori3,Hori4,img_back), axis=0)
			cv2.imshow('Select An Item', Verti)
			
			
		else:
			indeces = np.random.uniform(0,number_images,images2display).astype(int) # randomly selects indeces of images 

			images = []
			for i in range(images2display):#loading the images to display
				#print(cv2.imread(os.path.join(imagespath,list[indeces[i]])).shape)
				images.append(cv2.resize(cv2.imread(os.path.join(imagespath,list[indeces[i]])),(120,160)))

			Hori1 = np.concatenate((images[0], images[1],images[2]), axis=1) #aligning the images in horizontal axis to display 
			Hori2 = np.concatenate((images[3], images[4],images[5]), axis=1)  
			Hori3 = np.concatenate((images[6], images[7],images[8]), axis=1)
			Hori4 = np.concatenate((images[9], images[10],images[11]), axis=1)
			
			
			Main_page = 0
			Count = 0

			Verti = np.concatenate((Hori1, Hori2,Hori3,Hori4), axis=0)
			cv2.imshow('Select An Item', Verti)
		
		#cv2.circle(lastImage, (x, y), 3, (255, 0, 0), -1)
cv2.setMouseCallback('Select An Item', on_click)

cv2.waitKey(0)

