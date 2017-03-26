#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/hist.png "Sign Type Histogram of the Training Set"
[image2]: ./img/hist2.png "Sign Type Histogram of the Validation Set"
[image3]: ./img/hist3.png "Sign Type Histogram of the Test Set"
[gray]: ./img/gray.png "Grayscaled Traffic Sign"
[bypass]: ./img/lecun_bypass.png "Sermanet/LeCun Layer2 Bypass"
[uev]: ./1_ueberholverbot.png "No Passing"
[s100]: ./2_100_wvz.png "Speed Limit 100 km/h"
[row]: ./3_rakete.png "Right of Way at Next Intersection"
[s50]: ./4_50.png "Speed Limit 100 km/h"
[stop]: ./5_stop.png "Stop"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

[project code](https://github.com/baninho/CarND-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cells in the "Step 1" section of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed onto the different types of traffic signs.

![alt text][image1]

The data appears to be unevenly distributed among the different types of traffic signs. Poor performance in recognizing less frequently occurring signs may result from this.

![alt text][image2]

Similar distribution appear in both the the validation und test datasets. 

![alt text][image3]

It's possible this will translate into the trained model preferring the more frequent signs over less frequently appearing ones in its predictions.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the first code cell of the "Step 2" section in the IPython notebook.

I included the training data augmentation in this code cell. I used small translations, rotation, blur and addition of random noise to make the model more robust against disturbances in the image. This approach has been found by Sermanet/LeCun to improve model performance.

As a first step, I decided to convert the images to grayscale because this reduces the complexity at no cost in recognition accuracy (Sermanet/LeCun).

Here is an example of a traffic sign image after grayscaling.

![alt text][gray]

This image has also been augmented with a slight counter-clockwise rotation.

As a last step, I normalized the image data to reduce influence from different lighting conditions.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cell labeled "Model Architecture" of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 28x28x68 	|
| Sigmoid				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x68 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x68	|
| Sigmoid				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x68 					|
| Concatenation			| Layer2 bypass									|
| Fully connected		| width 75        								|
| Fully connected		| width 75        								|
| Output Layer			| width 43        								|
| Softmax				| 	        									|

I also used a Layer2 Bypass similar to the Sermanet/LeCun approach (shown below), which enables the network to evaluate both the low-level features found in the first layer and the higher-level features identified in the second layer.

![alt text][bypass]
Sermanet/LeCun Layer2 Bypass 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the last three cells of the "Step 2" section of ipython notebook. 

I used a learning rate of 0.001 and batch size of 256 for model training. 

I first train the model with dropout disabled for ten epochs which gave me a validation accuracy of 96.1%. I subsequently train the model with dropout enabled, only saving the new result if the validation accuracy increases, otherwise restoring the previously saved state and start again from there.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.962
* test set accuracy of 0.948

I chose to use a slightly modified Sermanet/LeCun implementation because they show the efficacy of their architecture in the suggested paper. I experimented with the layer width, finally compromising on an acceptable training duration while keeping the most prominent architecture features. My boundary condition was meeting a validation accuracy of 0.93 or higher as required. I use sigmoid activation instead of relus because me first results with relu activation units showed close to random accuracies both in training and validation, not improving during ten epochs of training. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][uev] ![alt text][s100] ![alt text][row] 
![alt text][s50] ![alt text][stop]

The first two images might be difficult to classify because they are shown on digital displays, inverting the color of background and pictograms/digits.

The third sign has a slight difference in lighting, while the last two signs should be easily recognised.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model can be found in the code cell following the label "Predict the Sign Type for Each Image".

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Passing      		| Stop sign   									| 
| 100 km/h     			| U-turn 										|
| Right of Way Next	Intersection| Yield									|
| 50 km/h	      		| Bumpy Road					 				|
| Stop					| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|