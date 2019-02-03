import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from skimage.feature import hog
 #Used for initialising the cnn model
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn import cluster
from sklearn import datasets
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
import os

#Constructing the arguement parse and parsing the arguements

#Initialising the number of epochs to train for, initial learning rate, and batch size

def train():
    #Initialising the data and labels
    data=[]
    labels=[]

    #Loop over the input images
    print('Getting img data')
    for root, dirs, files in os.walk(os.getcwd()+'/data_tot', topdown=False):
        for a in files:
            if ('Store' not in a):
                #print(a)
                #load the image, pre-process it, and store it in the data list
                ima = cv2.imread(os.path.join(root, a),0)
                im_gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                # Threshold the image
                blur = cv2.GaussianBlur(im_gray,(5,5),0)
                #ima = np.asarray(ima)
                #new = [int(b) for b in ima]
                
                #p =np.ndarray.flatten(ima)
                data.append(blur)
                #print(p)

                #extract the class label from the image path and update the labels list
                label = a[0]
                #print(label)
                if label=='0':
                    label = 0
                elif label =='1':
                    label=1
                elif label =='2':
                    label=2
                elif label =='3':
                    label=3     
                elif label =='4':
                    label=4    
                elif label =='5':
                    label=5
                elif label =='6':
                    label=6
                elif label =='7':
                    label=7
                elif label =='8':
                    label=8
                elif label =='9':
                    label=9
                elif label =='+':
                    label=10
                elif label =='-':
                    label=11
                elif label=='*':
                    label=12
                elif label=='=':
                    label = 13
                labels.append(label)        
    """
        elif label =='*':
            label=12
        elif label =='/':
            label=13
    """
        
    img_rows, img_cols = 28, 28

    #Scale the raw pixel intensities to the range [0,1]
    print('Putting in arr')
    data = np.array(data, dtype = 'int')/255.0
    labels= np.array(labels,dtype = 'int')

    #Partition the data into training and testing splits using 75% of the data
    #for training and the remaining 25% for testing

    #(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                    #  test_size=0.25, random_state =42)
    '''
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    '''
    #convert the labels from integers to vectors
    #trainY = to_categorical(trainY, num_classes=14)
    #testY = to_categorical(testY, num_classes=14)
    #trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    #testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    list_hog_fd = []
    for feature in data:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')
    (trainX, testX, trainY, testY) = train_test_split(hog_features, labels,test_size=0.25, random_state =42)
    print('training')
    #clf =LinearSVC()
    clf = svm.LinearSVC()
    # fitting our model
    clf.fit(trainX, trainY)

    # saving our modelx§
    print('Saving model')
    joblib.dump(clf, "test.pkl")
    f = open('Statistics.txt','a+')
    f.write('svm.SVC -gamma="scale", decision_function_shape="ovo"'+'\n')
    print('Stats:')
    y_pred = clf.predict(testX)
    print(confusion_matrix(testY, y_pred))
    f.write(str(confusion_matrix(testY, y_pred)))
    f.write('\n')
    print(classification_report(testY, y_pred))
    f.write(str(classification_report(testY, y_pred)))
    f.write('\n')
    f.close()

def chk_dig(n):
    if(n.isdigit()==True):
        return str(n)
    else:
        return n
def math_part(l):
    s = ''
    for a in l:
        print(a)
        s+=str(a)
    print(l)
    try:
        return eval(s)
    except:
        return s

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
 
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
 
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
 
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
 
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
    # draw the countour number on the image
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2)
 
    # return the image with the contour number drawn on it
    return image

def test():
    clf = joblib.load('test.pkl')
    im = cv2.imread('new.jpg')
    im = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
    #r = cv2.selectROI(im)
    #im_gray = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    # Threshold the image
    blur = cv2.GaussianBlur(im_gray,(5,5),0)
    ret,im_th =cv2.threshold(blur,90, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours in the image
    _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width = im.shape[0]
    max_height = im.shape[1]
    nearest = max_height * 1.4
    # show the original, unsorted contour image
    # sort the contours according to the provided method
    (ctrs, boundingBoxes) = sort_contours(ctrs, "left-to-right")
    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    pre = []
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im_gray, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        
        dic = {10:'+',11:'-',12:'*',13:'='}
        if nbr>9:
            nbr = dic[nbr[0]]
        pre.append(nbr[0])
        cv2.putText(im, str(nbr[0]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX,4,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(im_gray,'Answer:'+str(math_part(pre)),(im.shape[0]//2,im.shape[1]//2),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow("Resulting Image with Rectangular ROIs", im_gray)
    cv2.waitKey()
    # extract featuress

    #print(predictions)
    cv2.imwrite('e_edited.jpg',im)
#train()
#test()
