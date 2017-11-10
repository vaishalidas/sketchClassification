import cv2
import numpy as np
from os import listdir
import os
  
for folder in listdir("dataset"):
	count = 0
	for file in os.listdir("dataset/"+str(folder)):
		if not (str(file).startswith('.')):
		#file = str(file)[2:]
			image = cv2.imread('dataset/'+str(folder)+'/'+str(file), 0)
			img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
			ret,thresh = cv2.threshold(img,127,255,0)
			im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			cnt = contours[0]
			x,y,w,h = cv2.boundingRect(cnt)
			
			small_img = img[x:x+w,y:y+h]
		
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			count = count + 1
			if str(folder)== 'cat':
				path_train = 'E:/IT/Major/images/train/cat'
				path_test = 'E:/IT/Major/images/test/cat'
				if count < 19:
					cv2.imwrite(os.path.join(path_test , file), img)
					print 'ddd'
				else:
					cv2.imwrite(os.path.join(path_train , file), img)
			else:
				path_train = 'E:/IT/Major/images/train/shoe'
				path_test = 'E:/IT/Major/images/test/shoe'
				if count < 19:
					cv2.imwrite(os.path.join(path_test , file), img)
				else:
					cv2.imwrite(os.path.join(path_train , file), img)
					
			
