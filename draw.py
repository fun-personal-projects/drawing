import cv2
import numpy as np
import pickle
import mathrecog as mr
cap = cv2.VideoCapture(0)

points = []

flag = 0

while(1):

  _,frame = cap.read()
  frame = cv2.flip(frame,1)

  filterFrame = cv2.GaussianBlur(frame,(35,35),25)

  hsvFrame = cv2.cvtColor(filterFrame,cv2.COLOR_BGR2HSV)
  with open('range.pickle','rb') as f:
    t = pickle.load(f)
  lower_bound = np.array([t[0],t[1],t[2]])
  upper_bound = np.array([t[3],t[4],t[5]])


  threshImg = cv2.inRange(hsvFrame,lower_bound,upper_bound) 

  _,contours,_ = cv2.findContours(threshImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  finalImg = cv2.bitwise_and(frame,frame,mask=threshImg)

  finalImg = cv2.drawContours(finalImg,contours,-1,(255,255,0),1)

  c,X,Y=0,0,0

  key = cv2.waitKey(1)

  if flag==1:
    for item in contours:
      for i in item :
        X += i[0][0]
        Y += i[0][1]
        c += 1

    try:      
      points.append([int(X/c),int(Y/c)])    
    except:
      pass

  if (key & 0xFF == ord('s')) and flag == 0:
    flag = 1

  elif key & 0xFF == ord('s') and flag == 1 :
    flag = 0
  for p in range(1, len(points):
      if points[i-1] is None or points[i] is None:
      	continue
      cv2.line(finalImg, points[i-1], points[i], (0,255,0), 5) 

  cv2.imshow('Draw',finalImg)

  if key & 0xFF == ord('q'):
    cv2.imwrite('new.jpg', cv2.bitwise_not(finalImg))
    cap.release()
    cv2.destroyAllWindows()
    break

cv2.destroyAllWindows()
#cap.release()

mr.test()


