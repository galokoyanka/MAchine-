import cv2
import numpy as np

cam = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_data=[]
cnt=0
user_name=input("enter your name")
while True:
	ret, frame = cam.read()
	if ret==False:
		#print("something went wrong")
		continue
	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed ==ord('g'):
		break
	faces=face_cascade.detectMultiScale(frame,1.3,5)

	if(len(faces)==0):
		cv2.imshow("video",frame)
		continue
	for face in faces:
		x,y,w,h=face
		face_section=frame[y-10:y+h+10,x-10:x+w+10]
		face_section=cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		if cnt%10==0:
			print("taking picture",int(cnt/10))
			face_data.append(face_section)
	cnt+=1

	bright_image=frame+100
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow("video title",frame)
	cv2.imshow("video GRAy",face_section)
print("total faces",len(face_data))
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
np.save("Facedata/test"+user_name+".npy",face_data)
print("saved at Facedata/"+user_name+".npy")
print(face_data.shape)
cam.release()
cv2.destroyAllWindows()