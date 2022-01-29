import os
import numpy as np
import cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#from Horse_Face_Train_CV import horses


# Function to crop image based on ear location
def image_crop(read_path, save_path):
    img = cv2.imread(read_path)
    #img = cv2.resize(img, (500,677))
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar Cascade for horse ear detection created by HABIT-Horse Project
    haar_cascade = cv2.CascadeClassifier('/Users/macbook/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_horse_ear.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces_rect:
        ###################### FULL FACE CROP ############################################
        #cv2.rectangle(img, (x,(y+(h//5))), ((x+w),(y+(h*5)//2)), (0,255,0), thickness = 2)
        #crop = gray_scale[(y+(h//5)):(y+(h*5)//2), x:x+w]
        #cv2.imshow('Crop Image', crop)

        ####################### REDUCED CROP ############################################# 
        cv2.rectangle(img, (19*(x+w)//32,(y+(h*5//8))), (x+(12*w//16),(y+(h*9)//4)), (0,255,0), thickness = 2)
        crop = gray_scale[(y+(h*5//8)):(y+(h*9)//4), 19*(x+w)//32:x+(12*w//16)]
        cv2.imshow('Crop Image', img)

        c = cv2.waitKey(0) % 256

        if c == ord('y'):
            cv2.imwrite(save_path, crop) 
            break 



horses = ['Casanova', 'Superbad', 'Turbo', 'Vago', 'Charly']

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


# Get file name as input from user
print('File should be saved in "Testing" folder as .jpeg')
pic_name = input('Please enter file name (including extension): ')
crop_req = input('Does the picture need to be cropped? (y/n)')

file_path = 'Testing/' + pic_name
save_path = 'Testing/crop.jpeg'

if crop_req == 'y':
    # Crop image to match dimensions of training data
    image_crop(file_path,save_path)
    img = cv2.imread(save_path)
    disp_img = cv2.imread(file_path)
    disp_img = cv2.resize(disp_img, (500,677))

else:
    img = cv2.imread(file_path)
    disp_img = cv2.resize(img, (200,800))
    

#img_small = cv2.resize(img,(50,200))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

label, confidence = face_recognizer.predict(gray)
print(f'Label = {horses[label]} with a confidence of {confidence}')

cv2.putText(disp_img, str(horses[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

cv2.imshow('Detected Face', disp_img)

cv2.waitKey(0)