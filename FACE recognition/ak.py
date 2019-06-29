import cv2
import numpy as nu

cap=cv2.VideoCapture(0)
Face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data=[]
cnt=0
user_name=input("write your namw")


while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	
	faces=Face_cascade.detectMultiScale(frame,1.3,2)

	for face in faces:
		x,y,w,h=face
		face_section=frame[y-10:y+10+h,x-10:x+10+w]
		try:
			face_section=cv2.resize(face_section,(100,100))
		except:
			pass
		cv2.rectangle(frame,(x,y),(x+w,y+h),(240,240,220),3)
		cv2.imshow("cutout",face_section)
		if (cnt%10)==0:
			face_data.append(face_section)
			print(len(face_data))
		cnt+=1

	key_pressed=cv2.waitKey(1)&0xFF
	if key_pressed==ord('q'):
		break
	cv2.imshow('Frame',frame)	
print("total faces",len(face_data))
face_data=nu.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))

nu.save("Facedata/test"+user_name+".npy",face_data)
print("saved at Facedata"+user_name+".npy")
cap.release()
cv2.destroyAllWindows()