# Cat-Dog-Classifier

## Introduction

Through image recognition of the image dataset, we hope to use Convolution Neural Network (CNN) to classify images into 'cat' and 'rabbit' categories. We used CNN network. We also applied the HOG model to the code.



---



## Pre-process

We cut the photos into size of 128*128 for decreasing computing difficulty and reserving enough information of previous photo.



---



## Methodology

### Traditional method

#### K-Nearest Neighbors(KNN)

![KNN](/KNN.jpg)



K-Nearest Neighbors(KNN) find the k nearest data points to a given data, and use the feature of these k data points to predict the feature of given data.

In our project, we  used the KNN algorithm to classify whether an image represents a cat or a dog. First, we converted each 128\*128 image into a 128\*128 dimensional vector, where each component represents the grayscale value of the corresponding pixel. We then split the cat and dog data into a training set and a test set. To ensure balance, we selected 8000 cat and dog images for training, and 1997 cat and 2288 dog images for testing.

We loaded the data, split it into train and test sets, and converted each image into a vector using OpenCV. We then implemented the KNN classifier from scikit-learn, with different values of k from 1 to 47. For each value of k, we fitted the KNN model on the training data and predicted the class labels for the test data. We calculated the accuracy, confusion matrix, and classification report for each k value.

Finally, we printed out the accuracy, confusion matrix, and classification report for each k value, and stored them in the accuracy, confusion matrix, and classification_report lists, respectively. We also used seaborn and pandas to visualize the results and summarize the performance of the KNN classifier.



#### Histogram of Oriented(HOG)

![HOG](/HOG.png)





Histogram of Oriented Gradients (HOG) is a feature descriptor used in computer vision and image processing for object detection and recognition tasks. HOG works by computing histograms of image gradient orientations in localized portions of an image. The histogram represents the distribution of gradients in the image, and it captures the edge and texture information of the object.

In our project, we used Histogram of Oriented Gradients (HOG) to extract features from images for object classification. We first loaded the preprocessed image data from pickle files and reshaped them into a 2D array, where each row represents an image.

Next, we chose a subset of the training set to train the model due to limited computational resources. We then defined the HOG parameters, including the window size, block size, block stride, cell size, and number of histogram bins, and created a HOG descriptor using the OpenCV HOG Descriptor class.

We computed the HOG features for each image in the training and test sets using the HOG descriptor and stored them in separate arrays. We then saved the HOG features as numpy arrays for future use.



#### Scale-invariant feature transform(SIFT)

![SIFT](/SIFT.jpg)

SIFT (Scale-Invariant Feature Transform) is a popular feature descriptor used in computer vision and image processing for object recognition, image matching, and scene reconstruction tasks. SIFT works by detecting and describing local features in images that are invariant to scale, rotation, and illumination changes.

 In our project, we used Scale-Invariant Feature Transform (SIFT) to extract features from images for object classification. We loaded the preprocessed image data from pickle files and selected a subset of the training and test sets due to limited computational resources.

We created a SIFT object using the OpenCV SIFT_create() function and applied it to each image in the training and test sets to extract keypoints and descriptors. We normalized the descriptors by dividing them by the Euclidean norm to ensure scale invariance. We then concatenated the descriptors of all keypoints in each image and stored them in separate arrays along with their corresponding labels.

Finally, we saved the extracted SIFT features and labels as pickle files for future use. We also selected a subset of the training set for training the classifier, using a random permutation of the SIFT features and labels to ensure randomness.



#### Support Vector Machine(SVM)

![SVM](/SVM.png)

Support Vector Machine(SVM) is a popular machine learning algorithm used for classification and regression tasks in various domains, such as computer vision, natural language processing, and bioinformatics. The SVM algorithm works by finding an optimal hyperplane that maximally separates the data into different classes, based on the geometrical properties of the data.

In our project, the code provided defines a function named `hog_svm` that trains a support vector machine (SVM) model using the HOG features extracted from an image dataset. The function takes two parameters: the kernel type (linear, rbf,  poly or sigmoid) and the regularization parameter C(0.01,0.1,1,10,100).

The function then fits the SVM model to the training data and makes predictions on the test set. It computes various performance metrics, such as accuracy, precision, recall, F1-score, AUC score, and the confusion matrix. These metrics are printed out to the console, and the function returns them in a dictionary object named params.

The function is then called multiple times with different kernel types and regularization parameters to train and evaluate the SVM model on the same dataset. The performance of the SVM model is compared across different values of the kernel and C parameters. The results are printed out to the console and stored in the params dictionary object.



### Neural Networks

Neural network is a type of machine learning model that is inspired by the structure and function of the human brain. It consists of interconnected layers of nodes (also known as neurons), each of which performs a mathematical operation on the input it receives and passes the result on to the next layer. By adjusting the weights and biases of these connections, the network can learn to perform complex tasks.

#### Convolutional Neural Network(CNN)

![naiveCNN](/naiveCNN1.jpg)

Convolutional Neural Networks (CNNs) are a type of deep neural network that are commonly used for image recognition tasks. They are designed to automatically learn and extract features from images, which are then used to make predictions.

CNNs are composed of multiple layers, including convolutional layers, pooling layers, and fully connected layers. The convolutional layers use filters to extract features from the input images, while the pooling layers reduce the spatial size of the feature maps. The fully connected layers then use the extracted features to make predictions.

The picture demonstrates the structure of our initial CNN(naive CNN).

##### Use different kernels in a convolution layer

![difker](/difker.png)



Using different convolutional kernels in the same convolutional layer can bring several benefits in extracting image features. Firstly, it enables the network to capture multiple types of features at different scales within a single layer. For example, one kernel may be specialized in detecting edges, while another kernel is focused on identifying textures or shapes. By combining the outputs of these different kernels, the network can capture richer and more diverse features, leading to better performance in tasks such as object recognition or image classification.

Secondly, using multiple kernels in the same layer can increase the number of parameters in the model, which can improve the model's ability to fit complex data distributions. However, this increase in parameters must be carefully balanced with the risk of overfitting, which can occur if the model becomes too complex relative to the size of the training data.

This is the structure of our network after adding different kernels.

##### Data Argumentation

We randomly choose part of dataset to vertical(horizontal) flip and rotate to generate more data.(However it is found that the data after argumentation has more difference compare to the test set.)

#### Ensemble Learning

We trained 10 models which use 80% train set for each model.

And use the best model in `Different Kernel` part, voting for the result.

---





## Reproduction

If you want to reproduce our result, please follow the steps below:

### KNN

In KNN part, we only used the first part of our dataset with about 16000 photos.

Just run the **`KNN.ipynb`** ,similar result will be shown.



---



Before trying other methods except KNN, please run the **`Get array and label.ipynb`** to Generate the files including **`X`,`y`,`X_train`,`y_train`,`X_test`,`y_test`**.

### HOG+SVM

Run the **`HOG_SVM.ipynb`**.

In first part, the code will generate **`hog_features_train.npy`** and **` hog_features_test.npy`**.

You can use them to reproduce the second part which check the performance of SVM with different parameters(`kernel` and `C value`).

### SIFT+SVM

Run the **`SIFT_SVM.ipynb`** .

In first part, the code will generate **`X_train_descriptors`** , **` y_train_labels`**,**`X_test_descriptors`** and **` y_test_labels`**.

However it does not performance well.( It dose not have value to reproduce.)

### SIFT+ Neural Networks

First part, same as **`SIFT+SVM`**, to obtain 4 files.

Then, run the **`SIFT_Neuret.ipynb`**.



### CNN

The folder `naive_CNN_model` contains the model of different epochs with the method of naive CNN.

The folder `difker_model` contains the model weight of different epochs with the method of using different kernels in CNN.

And `arg_10e.h5` and `arg_100e.h5` are the model weight after using argued data to train the model.

If reproducing, just run **`CNN.ipynb`**( naive CNN ) or  **`Different_Kernel.ipynb`** ( Different Kernel ) or **`Data_Arg.ipynb`**( data argumentation )

### Ensemble Learning

Run the **Ensemble_Learn_preprocess.ipynb** to obtain the training sets of models.

Move these files and the weight of best model we trained in `difker_model`(**difker_3020e.h5**) to the folder `Ensemble Learning` .

Run the  **Emsemble_learning.ipynb** in this folder to train 10 models.

Run the **Emsemble_test.ipynb** to evaluate the performance of model.



---



## Display

You can try your own photo to test whether it is considered as a dog or a cat.

Open the **display_version.ipynb** and put your own photo of a cat or a dog.

```
img=cv.imread('photo3.jpg',0)#change it to your photo name
```

Change the `photo3.jpg` to your own photo name.

Run the code cell by cell and the followed cell will print the result.

```
#show result，dog is 0 and cat is 1
print(prediction)
```

