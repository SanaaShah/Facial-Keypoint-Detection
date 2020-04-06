# Facial-Keypoint-Detection
This project is a part of Udacity's Nano-degree program. In this project, an input image is provided which is fed into to algorithm 
providing an output containing the key points on the face. 
* It can detect facial keypoints on multiple faces.
* It has been implemented using python and pytorch.

Below is an example of the output of this project:


![Input Image](/picturesToDemonstrate/InputImage.JPG)

![Faces detected](/picturesToDemonstrate/detectedFaces.JPG)

![Keypoints detected](/picturesToDemonstrate/detectedKeypoints.JPG)

## Additional Feature
I have also added a feature to place sunglasses over the image on the detected keypoints.

![Input Image](/picturesToDemonstrate/pictureForglasses.JPG)

![Input Image](/picturesToDemonstrate/outputGlasses.JPG)

## Future Aspect
I want to implement more features to this as:
* Initialize the weights of your CNN by sampling a normal distribution or by performing Xavier initialization so that a particular
input signal does not get too big or too small as the network trains.
* In Notebook 4, create face filters that add sunglasses, mustaches, or any .png of your choice to a given face in the correct location.
* Use the keypoints around a person's mouth to estimate the curvature of their mouth and create a smile recognition algorithm .
* Use OpenCV's k-means clustering algorithm to extract the most common facial poses (left, middle, or right-facing, etc.).
* Use the locations of keypoints on two faces to swap those faces.
* Add a rotation transform to our list of transformations and use it to do data augmentation.

## Liscense
The contents of this repository are covered under the [MIT License](https://github.com/udacity/ud777-writing-readmes/blob/master/LICENSE).

If you want to know more about this nano,-degree, visit [this](https://www.udacity.com/course/computer-vision-nanodegree--nd891) link.


