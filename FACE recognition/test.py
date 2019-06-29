import cv2
import numpy as np 
import os

def dist(p,q):
	return np.sum((q-p)**2)**0.5
def knn(train,test, k=6):
	dista =[]

	for i in range(train.shape[0]):
		ix=train[i,:-1]
		iy=train[i,-1]

		d=dist(test,ix)
		dista=d.append((d,iy))

	dista= np.array(sorted(dista))[:,k]
	t =np.unique(dista,return_counts=True)
	index=np.argmax(t[1])
	return t[0][index]

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cnt=0
face_data=[]
dataset_path='./Facedata/'
class_id=0
label=[]
names={}

for f in os.listdir(dataset_path):
	if f.endswith('npy'):
		names[class_id] = f[:-4]
		print("Loading file ",f)
		data=np.load(dataset_path+f)
		face_data.append(data)

		target=class_id*np.ones((data.shape[0],))
		label.append(target)
		class_id+=1

x=np.concatenate(face_data,axis=0)
y=np.concatenate(label,axis=0)

trainset = np.concatenate((x,y.reshape(-1,1)),axis=1)

while True:
	ret, frame=cap.read()

	if ret == False:
		continue

	if (cv2.waitKey(0) & 0xFF)==ord('q'):
		break
	faces=face_cascade.detectMultiScale(frame,1.3,3)

	for face in faces:
		x,y,w,h=face
		face_section=frame[y-10:y+h+10,x-10:x-10+w]
		face_section=cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(240,240,204),3)
		cv2.imshow("cutout", face_section)
		pred=knn(trainset,face_section.flatten())
		name=names[int(pred)]
		cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	cv2.imshow("frame",frame)
cap.release()
cv2.destroyAllWindows()	





    
