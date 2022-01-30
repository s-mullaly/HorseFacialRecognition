# HorseFacialRecognition

This project was completed as part of the MS Analytics program at Georgia Tech.

The project had two phases. The first was to identify a horse in an image and crop the image to a meaningful representation for identification. The second phase was to then train a model with a number of horses and then identify which horse was present in a new image presented to the model.

The report created for this project is included along with the code, which can be run by following these steps:

Run 'Horse_Face_Train_CV.py' to create 'features.npy', 'labels.npy', and 'face_trained.yml'

After training, run 'Horse_Face_Recognize_CV.py'

Enter file name in Testing folder, such as 'cas.jpeg'

If the image needs to be cropped then press 'y'

The image will display with a region of interest. To discard press any key. To use the region of interest press 'y'.

If no region of interest is created it will not create a new 'crop.jpeg' file. If there is no file it will terminate,
if a previous run's 'crop.jpeg' is still present it will use that file and return the previous result.

Once the identification is complete, press any key to close the image window and terminate the script.
