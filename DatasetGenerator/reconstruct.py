import numpy as np
import cv2

images = []

#Anger = 0,Fear = 2, Happy = 3, Sad = 4, Surprise = 5, Neutral = 6
m = 0

with open("fer2013.csv", "r") as filestream:
	for line in filestream:
		currentline = line.split(",")
		if int(currentline[0]) == 0:
			arr = map(int,currentline[1].split())
			if len(arr) == 2303:
				arr.append(0)
			elif len(arr) == 2305:
				arr.pop()
			images.append(arr)
			m = m + 1
		if m == 4000:
			break
m = 3000
#print len(images)
while m<4000:
	img = np.ones((48,48), np.uint8)
	k = 0
	for i in range(0,48):
		for j in range(0,48):
			img[i,j] = (images[m][k])
			k = k + 1
			#print(k)
	string = 'anger'
	cv2.imwrite(string+str(m)+'.jpg',img)
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	m = m + 1
	#print(m-1)

print ("Finished")

