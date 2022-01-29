import cv2
import os
import numpy as np

def image_crop(read_path, save_path):
    img = cv2.imread(read_path)
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
            print('saved') 


# List of folders/horses in the training set (uncropped)
horses = []
directory = 'Training/Full'
for subdirectory in os.listdir(directory):
    if not subdirectory.startswith('.'):
        horses.append(subdirectory)

# Iterate through each horse's Full size training images to crop
# Save to horse's cropped folder

for horse in horses:
    directory = 'Training/Full/' + horse
    i = 0
    for pic in os.scandir(directory):
        print(os.path.relpath(pic))
        file_path = os.path.relpath(pic)
        save_path = 'Training/Cropped/' + horse + '/' + str(i) + '.jpeg'
        if file_path[-4:] == 'jpeg':
            image_crop(file_path,save_path)
            i += 1
