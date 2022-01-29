Set pwd to 'FINAL_CODE' folder.
Run 'Horse_Face_Train_CV.py' to create 'features.npy', 'labels.npy', and 'face_trained.yml'
After training, run 'Horse_Face_Recognize_CV.py'
    Enter file name in Testing folder, such as 'cas.jpeg'
    If the image needs to be cropped then press 'y'
    The image will display with a region of interest. To discard press any key. To use the region of interest press 'y'.
    If no region of interest is created it will not create a new 'crop.jpeg' file. If there is no file it will terminate,
        if a previous run's 'crop.jpeg' is still present it will use that file and return the previous result.
    Once the identification is complete, press any key to close the image window and terminate the script.
