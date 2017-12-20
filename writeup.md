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

[image1]: ./img/visualization.png "Visualization"
[image2]: ./img/grayscale.png "Grayscaling"
[image3]: ./img/augment.png "Image Augmentation"
[image4]: ./img/wild.png "Images from internet"
[image5]: ./img/prediction.png "Top 3 probablities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/prasad-pillai/traffic-sign-deeplearn/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a random image map with class name and image dimension

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**My preprocessing steps are as follows **

Convertion to grayscale. I choose this after few rounds of experimentation. More accuracy was obtained on grayscale images when compared to color images. Grayscaling also reduces the size of the dataset to 1/3rd this making it easier and less time consuming to train.
    
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data in the range (-1,1). Normalization is a must have preprocessing process, as explianed in the class. I choose this methord because it is fairly easy and also as it is shown in the lessons.


I decided to generate additional data because i think more data will help create a better model, also because data augmentation has been proven to help the model better generalise rather than to memorize.

To add more data to the the data set, I used a random combination of translation, scaling, skew and rotate.

Here is an example of an original image and an augmented image:

![alt text][image3]

The new dataset has equal number of samples for all the 43 classes.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


    5x5 convolution (32x32x1 in, 28x28x6 out)
    ReLU
    2x2 max pool (28x28x6 in, 14x14x6 out)
    5x5 convolution (14x14x6 in, 10x10x16 out)
    ReLU
    2x2 max pool (10x10x16 in, 5x5x16 out)
    5x5 convolution (5x5x6 in, 1x1x400 out)
    ReLu
    Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
    Concatenate flattened layers to a single size-800 layer
    Dropout layer
    Fully connected layer (800 in, 43 out)



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Adam optimizer which is a more complex and accurate algorithm campared to the traditional Gradient Descentent algorithm. The hyperparameter settings where as follows

    BATCH SIZE: 100
    EPOCH: 50
    LEARNING RATE: 0.001
    MU: 0
    SIGMA: 0.1
    DROPOUT KEEP PROBABLITY: 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.993
* test set accuracy of 0.935

I started with the lenet model from the lessons and started tweking the hyperparameters to see how validation accuracy changes. Though playing around with the paramaters alone did not give me enough accuracy. Later i used grayscaling which gave me an efficiency boost. But was a real surprise that the image depth from 3 to 1 which is infact data loss could contribute to increase in accuracy. Later i tried data augmentation. Initialy i tried augmenting the existing data, ie not adding any more data but rather augment the existing datasets. This did not give me any good news rather this decresed the accuracy. Later i tried doubling the dataset with the newly added data being augmented. This also did not improve my accuracy. Later i tried equalizing the number of data for each label by adding as many augmented data sets to make all the sample a purticular size. This method performed better than other two augmentations but did not give me enough accuracy as i wanted.

As the Lenet architecture could not give me as much accuracy as i wanted, i moved to try other architecture, i manually tried adding and removing layers but the efficiency was not improving much. Later i came across with the sermanet/LeCunn article. I implemented that architecture and found the efficieny to have improved greatly. I also experimented adding more augmented data with this model. After all those experimentation i came up with the final model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] 

I have used more clearer and brighter images from the internet. This maight be a source of confussion for the model. Again, the first image i have used is actually not in the dataset, the closest match is the turn left label. I wanted to see which label will it get predicted to be. The 6th image is with blue borders in the dataset but here i have used red only assuming that when converted to grayscale, they both will look the same.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of ~70%. This is quite less compared with the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top five soft max probabilities are shown below

![alt text][image5] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


