## Writeup Report

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1-1]: ./output_images/car_notcar.png
[image1-2]: ./output_images/hog_output.png
[image4]: ./output_images/pipeline_result.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file `feature.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1-1]

I then explored different color spaces and different `skimage.hog()` parameters
(`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel
for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image1-2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and different features, applying those on test\_images/test4.jpg
and see if they work.
HOG with 8 pixels and 8 cells works well without making it a big one.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature function is defined in line 28 through 37 in file `train.py`.
I trained a linear SVM using `sklearn.svm.LinearSVC` class, along with C=0.001 as its parameters,
which is chosen by the `GridSearchCV` method.
I used `train_test_split` method to split training and testing sets.
A model and a scaler were saved after training(code is in line 40 through 75 of file `train.py`).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search through the image at 3 different settings of y-axis scopes and window sizes
(in lines 109 through 113 of the file `find_car.py`). The small window sizes locate
in upper range of y-axis in the image because the bigger vehicles appear in the image, the closer to the camera they are.
The overlaps are set to 0.75 for the balance between accuracy and processing speed.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using HSV 3-channel HOG features plus histograms of color in the feature vector,
which provided a nice result.
Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
The resulting heatmap is calculated according to the weighted average of the heatmaps of several latest frames.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle. 
I constructed bounding boxes to cover the area of each blob detected.  
The related code is in file `heat_map.py`.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are still some false positive cases from the processed video.
The noises could be caused by dirty road surface and fences, etc.
I believe a stronger classifier such as well-trained CNN structure would help a lot for robustness.
Maybe some method like Kalman Filter can help to filter out several false positives by
statistics over nearby frames.

Sometimes the bounding boxes flicker in the result video.
For further improvement, I would like to try a Gaussian mixture model to fit the heatmap
in order to get smoother bounding boxes.

