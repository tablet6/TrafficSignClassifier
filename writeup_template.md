#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:<br />
* Load the data set (see below for links to the project data set)<br />
* Explore, summarize and visualize the data set<br />
* Design, train and test a model architecture<br />
* Use the model to make predictions on new images<br />
* Analyze the softmax probabilities of the new images<br />
* Summarize the results with a written report<br />


[//]: # (Image References)

[image1]: ./results/Classes.png
[image2]: ./web/z113.jpg
[image3]: ./web/z114.jpg
[image4]: ./web/z136_10.jpg
[image5]: ./web/z205.jpg
[image6]: ./web/z274.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

The Project HTML is here: [Traffic\_Sign\_Classifier.html](http://htmlpreview.github.com/?https://github.com/tablet6/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.html)

---


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


Loaded the data. Below are the samples in each training set. </br>
Training Set:   34799 samples</br>
Validation Set: 4410 samples</br>
Test Set:       12630 samples</br>

There are 43 unique classes in the data set and each Image is of shape = (32, 32, 3). Images are RGB color.


####2. Include an exploratory visualization of the dataset.

Below screenshot is displaying one image in each class. Total 43 classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To pre-process the image data - I used the below normalization.
image\_data = (image\_data - mean(image\_data))/std(image\_data). </br>
This is for calculating the standard deviations from the mean with given: image\_data, mean and Standard Deviation, which is also called standard or z-score.

I started with (image_data-128.)/128, but the model did not generate good validation accuracy. So settled on standard score normalization.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is based on LeNet. </br>
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Valid padding, outputs 14x14x6 	|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride, Valid padding, outputs 5x5x16 	|
| Flatten               | Input = 5x5x16. Output = 400                  |
| Fully connected		| Input = 400. Output = 120        				|
| RELU                  |                                               |
| DROPOUT               |                                               |
| Fully connected		| Input = 120. Output = 84        				|
| RELU                  |                                               |
| DROPOUT               |                                               |
| Fully Connected       | Input = 84. Output = 43                       |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
learning\_rate = 0.001, calculated cross_entropy, loss, used AdamOptimizer and minimize for Gradient Descent. 

Used 10 Epochs and a batch size of 128.



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After training, my model results had Validation accuracy of 0.944 and Test Accuracy of 0.924.
I used the LeNet arcihtecture with 2 Dropouts between FullyConnected layers. 
This was done when the accuracy wasn't going beyong 0.89. Dropouts is a regularization technique to randomly set half of activations to zero. 
I also experimented with Gray Scale images, but that didn't make much of a difference with the accuracy, so reverted back to RGB.

Sometimes I used to see 0.05 accuracy, which used to blow my mind. But that was probably because of doing normalization of the image data multiple times. 
So I ended up with bad data. In those cases, I used to restart the kernel and the results used to settle down. 
I played with learning rate of 0.001, but the accuray dropped a lot. The standrd deviation of 0.12 instead of 0.1 gave me good accuracy even though I went back to 0.1.

I think the model is performing well, but needs improvement.  

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

These images were originally 120x80 and I resized to 32x32, which makes them pixelated and difficult for the model to classify.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Beware of ice/snow     			| Children crossing 										|
| Slippery Road					| Dangerous curve to the left											|
| Children crossing	      		| Children crossing					 				|
| Yield			| Yield      	
| 60 km/h			| 60 km/h      	


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares not much favorably to the accuracy on the test set of 92%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


First Image: <br />
Original: Beware of ice/snow<br />
Prediction: Correct. Probability of 0.98.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98     			| Beware of ice/snow										|
| 0.01					| Dangerous curve to the right											|
| 0.0003	      		| Slippery Road					 				|
| 0.0001			| Childrens Crossing      	
| 0.0000003			| Bicycles Crossing  


Second Image: <br />
Original: Slippery Road<br />
Prediction: Wrong. A close call between Dangerous curve (0.63) and Slippery Road (0.46).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.53     			| Dangerous curve to the left 										|
| 0.46					| Slippery Road											|
| 1.8e-04	      		| Double curve					 				|
| 2.1e-06			| Wild animals crossing      	
| 0.0000003			| Road work 


Third Image: <br />
Original: Children crossing<br />
Prediction: Correct. Probability of 0.99.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99     			| Children crossing										|
| 5.8e-04					| Bicycles crossing											|
| 2.1e-09	      		| Traffic signals					 				|
| 3.9e-10			| Beware of ice/snow      	
| 5.8e-12			| Bumpy road


Fourth Image: <br />
Original: Yield.<br />
Prediction: Correct. Probability of 1.00.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0     			| Yield										|
| 1.6e-36				| Ahead only											|
| 0.0	      		| Speed limit (20km/h)					 				|
| 0.0		| Speed limit (30km/h)      	
| 0.0			| Speed limit (50km/h) 


Fifth Image: <br />
Original: Speed limit (60km/h).<br />
Prediction: Correct. Probability of 0.99.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99    			| Speed limit (60km/h)										|
| 2.5e-04				| Speed limit (80km/h)											|
| 7.3e-05	      		| Speed limit (50km/h)					 				|
| 4.1e-12			| Speed limit (30km/h)     	
| 2.3e-17			| Ahead only 


