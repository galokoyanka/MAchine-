import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_data=[]
data_set='./New Folder'
cnt=0
while True:
	ret,frame=cap.read()
	if ret == False:
		continue

	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


	gray=cv2.flip(gray,1)
	frame=cv2.flip(frame,1)
	faces=face_cascade.detectMultiScale(frame,1.3,5)

	for (x,y,w,h) in  faces:
		face_section=frame[y-10:y+10+h,x-10:x-10+w]
		face_section=cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.imshow('fr2',face_section)
		
		if (cnt%10==0):
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow('FRa',frame)
	#cv2.imshow('fra2',gray)
	

	key= cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()