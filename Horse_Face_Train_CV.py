import os
import cv2
import numpy as np
from PIL import Image

# Create list of all horses with cropped photos
horses = []

directory = 'Training/Cropped'
for subdirectory in os.listdir(directory):
    if not subdirectory.startswith('.'):
        horses.append(subdirectory)

print(horses)

features = []
labels = []

def create_train():
    for horse in horses:
        path = os.path.join(directory, horse)
        label = horses.index(horse)
        print(label)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv2.imread(img_path)
            if img_array is None:
                continue 

            #img_small = cv2.resize(img_array, (500,677))
            #print(img_array)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            features.append(gray)
            labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
