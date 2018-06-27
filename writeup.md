# **Traffic Sign Recognition** 

## Writeup

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

[summary]: ./traffic-signs-data/result/summary.png "Summary"
[rgb2gray1]: ./traffic-signs-data/result/rgb2gray1.png "Priority road"
[rgb2gray2]: ./traffic-signs-data/result/rgb2gray2.png "Stop"
[rgb2gray3]: ./traffic-signs-data/result/rgb2gray3.png "Traffic signals"

[50km]: ./traffic-signs-data/test_imgs/2.jpg "Speed limit(50km/h)"
[nopass_3.5]: ./traffic-signs-data/test_imgs/10.jpg "No passing for vehicles over 3.5 metric tons"
[stop]: ./traffic-signs-data/test_imgs/14.jpg "Stop"
[slippery]: ./traffic-signs-data/test_imgs/23.jpg "Slippery road"
[bicycle]: ./traffic-signs-data/test_imgs/29.jpg "Bicycles crossing"

[rotate_0]: ./traffic-signs-data/result/rotate_0.png
[rotate_1]: ./traffic-signs-data/result/rotate_1.png
[rotate_2]: ./traffic-signs-data/result/rotate_2.png
[rotate_3]: ./traffic-signs-data/result/rotate_3.png
[rotate_4]: ./traffic-signs-data/result/rotate_4.png
[rotate_5]: ./traffic-signs-data/result/rotate_5.png
[rotate_6]: ./traffic-signs-data/result/rotate_6.png
[rotate_7]: ./traffic-signs-data/result/rotate_7.png
[rotate_8]: ./traffic-signs-data/result/rotate_8.png

[pred2]: ./traffic-signs-data/result/pred2.png
[pred10]: ./traffic-signs-data/result/pred10.png
[pred14]: ./traffic-signs-data/result/pred14.png
[pred23]: ./traffic-signs-data/result/pred23.png
[pred29]: ./traffic-signs-data/result/pred29.png

[conv1]: ./traffic-signs-data/result/vis_conv1.png
[conv2]: ./traffic-signs-data/result/vis_conv2.png

[l5]: ./traffic-signs-data/result/l5.png
[l5_dropout]: ./traffic-signs-data/result/l5_dropout.png
[l5-over]: ./traffic-signs-data/result/l5-over.png
[l5-over_dropout]: ./traffic-signs-data/result/l5-over_dropout.png
[l5-aug]: ./traffic-signs-data/result/l5-aug.png
[l5-aug_dropout]: ./traffic-signs-data/result/l5-aug_dropout.png
[l5-over-aug]: ./traffic-signs-data/result/l5-over-aug.png
[l5-over-aug_dropout]: ./traffic-signs-data/result/l5-over-aug_dropout.png

[my]: ./traffic-signs-data/result/my.png
[my_dropout]: ./traffic-signs-data/result/my_dropout.png
[my-over]: ./traffic-signs-data/result/my-over.png
[my-over_dropout]: ./traffic-signs-data/result/my-over_dropout.png
[my-aug]: ./traffic-signs-data/result/my-aug.png
[my-aug_dropout]: ./traffic-signs-data/result/my-aug_dropout.png
[my-over-aug]: ./traffic-signs-data/result/my-over-aug.png
[my-over-aug_dropout]: ./traffic-signs-data/result/my-over-aug_dropout.png
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? 32 x 32 x 3
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many records for each classes 

![Class summary][summary]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I found color didn't affect the judgement. When I convet these images to grayscale, we still can identify each sign by its shape. 

Here is an example of a traffic sign image before and after grayscaling.

![Priority road][rgb2gray1]
![Stop][rgb2gray2]
![Traffic signals][rgb2gray3]

I put the normalized step in network because we use numpy.uin8 to store data. It can save lots of memory.

I decided to generate additional data because unbalanced problem. I choose oversampling technique to duplicate less size class data.

To add more data to the the data set, I used the augmentation techniques because we need more data to avoid overfitting for small dataset. So I randomly add rotation (-15 ~ 15 degress) to images.

Followings are rotation examples:

![][rotate_0]
![][rotate_1]
![][rotate_2]
![][rotate_3]
![][rotate_4]
![][rotate_5]
![][rotate_6]
![][rotate_8]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In my experiments, I implement 2 network: LeNet5 and MyCNN(customized network).
The architecture is like classic LeNet5. The difference is activation function. I choose leaky relu and alpha is 1e-2.
And MyCNN model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 |
| LeakyRELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 |
| LeakyRELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 |
| LeakyRELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64 |
| LeakyRELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x32|
| Dropout	      	| |
| Fully connected		| etc.        									|
| Softmax				| etc.        									|

The reason for why I put 2 convolution in layer1 and layer2 before pooling is from VGG network. I did some experiments and found multiple convolution before pooling can give better result.

The loss function is consist of cross entropy for softmax and regularization. The first part for regularization is `sparse`. I use absolute function to control the output of each convolution layer. Like spase coding, we want convolution layer just give significant output. Second part is L2 regulaization for fully connected layer. with this 2 regulaization terms, my model can optimize very quickly.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam optimizer because adam is good at cnn problem from other research results.
I set 50 epochs because I just want to know if we optimize more does it gives us better result.
There are 43 classes in training data. If we do randomly sampling with small mini batch, some batch could not get enough data for each calsses. So the size is 128.
I choose dropout rate: 0.5 before output layer to avoid overfitting.
The final thing is about learning rate. All experiments start with learning rate: 1e-2 and decay rate: 0.9 after each epochs. The network should start from large learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I did 24 experiments with (oversampling, augmentation, dropout) for 2 models. And I found dropout could avoid fully connected layer overfitting. The oversampling could improve validation accuracy because this is unbalanced problem. The augmentation also achieve good result. When I combine them together, I get the highest validation and test accuracy for both models.

LeNet5 model (oversampling, random augmentation, dropout) results were:
* training set accuracy of ? 0.9984
* validation set accuracy of ? 0.9723
* test set accuracy of ? 0.9546

MyCNN model (oversampling, random augmentation, dropout) results were:
* training set accuracy of ? 0.9995
* validation set accuracy of ? 0.9923
* test set accuracy of ? 0.9795

Followings are detailed for each experiments:

| | LeNet5 | MyCNN |
:----------------------------:|:----------------------------:|:--------------------------:
Raw | ![][l5] | ![][my]
Oversampling| ![][l5-over] | ![][my-over]
Augmentation| ![][l5-aug] | ![][my-aug]
Oversampling + Augmentation| ![][l5-over-aug] | ![][my-over-aug]

| | LeNet5 Dropout | MyCNN Dropout |
:----------------------------:|:----------------------------:|:--------------------------:
Raw | ![][l5_dropout] | ![][my_dropout]
Oversampling| ![][l5-over_dropout] | ![][my-over_dropout]
Augmentation| ![][l5-aug_dropout] | ![][my-aug_dropout]
Oversampling + Augmentation| ![][l5-over-aug_dropout] | ![][my-over-aug_dropout]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Speed limit(50km/h)][50km] ![No passing for vehicles over 3.5 metric tons][nopass_3.5] ![Stop][stop] 
![Slippery road][slippery] ![Bicycles crossing][bicycle]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![Speed limit(50km/h)][pred2]
![No passing for vehicles over 3.5 metric tons][pred10]
![Stop][pred14] 
![Slippery road][pred23]
![Bicycles crossing][pred29]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The followings are visualization of feature maps of layer1 and layer2 in MyCNN. I found network emphasize the edge of sign like we did in edge detection.
![Layer1][conv1]
![Layer2][conv2]


## References
1. https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/
1. https://github.com/Goddard/udacity-traffic-sign-classifier/
