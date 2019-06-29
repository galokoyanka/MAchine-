import cv2
import numpy as np
cap=cv2.VideoCapture(0)
Face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_data=[]
dataset='./Facedata'
file_name=input("write your name/n")

cnt=0
while True:
	ret,frame=cap.read()
	if ret==False:
		continue

	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	frame=cv2.flip(frame,1) 
	gray=cv2.flip(gray,1) 
	faces=Face_cascade.detectMultiScale(frame,1.3,2)
	#faces2=Face_cascade.detectMultiScale(gray,1.3,2)
	for face in faces:
		x,y,w,h=face
		face_section=frame[y-10:y+10+h,x-10:x+10+w]
		face_section=cv2.resize(face_section,(100,100))
		cv2.imshow('cut',face_section)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(240,240,0),3)
		if cnt%10==0:
			face_data.append(face_section)
			print(len(face_data))
	cnt+=1
    

	#for face in faces2:
		#x,y,w,h=face
		#face_section=gray[y-10:y+10+h,x-10:x+10+w]
		#face_section=cv2.resize(face_section,(100,100))
		#cv2.imshow('cuto',face_section)
		#cv2.rectangle(gray,(x,y),(x+w,y+h),(240,240,0),3)



	cv2.imshow('FRame',gray)

	cv2.imshow('Frame',frame)

	if (cv2.waitKey(1) & 0xFF)==ord('q'):
		break
	#convert array tp numpt array
#face_data=np.asarray(face_data)
#face_data=face_data.reshape(face_data.shape[0],-1)
#print(face_data)
#np.save(dataset+file_name+'.npy',face_data)


cap.release()
cv2.destroyAllWindows()