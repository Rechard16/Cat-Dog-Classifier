# Reproduction

If you want to reproduce our result, please follow the steps below:

## KNN

In KNN part, we only used the first part of our dataset with about 16000 photos.

Just run the **`KNN.ipynb`** ,similar result will be shown.



---



Before trying other methods except KNN, please run the **`Get array and label.ipynb`** to Generate the files including **`X`,`y`,`X_train`,`y_train`,`X_test`,`y_test`**.

## HOG+SVM

Run the **`HOG_SVM.ipynb`**.

In first part, the code will generate **`hog_features_train.npy`** and **` hog_features_test.npy`**.

You can use them to reproduce the second part which check the performance of SVM with different parameters(`kernel` and `C value`).

## SIFT+SVM

Run the **`SIFT_SVM.ipynb`** .

In first part, the code will generate **`X_train_descriptors`** , **` y_train_labels`**,**`X_test_descriptors`** and **` y_test_labels`**.

However it does not performance well.( It dose not have value to reproduce.)

## SIFT+ Neural Networks

First part, same as **`SIFT+SVM`**, to obtain 4 files.

Then, run the **`SIFT_Neuret.ipynb`**.



## CNN

The folder `naive_CNN_model` contains the model of different epochs with the method of naive CNN.

The folder `difker_model` contains the model weight of different epochs with the method of using different kernels in CNN.

And `arg_10e.h5` and `arg_100e.h5` are the model weight after using argued data to train the model.

If reproducing, just run **`CNN.ipynb`**( naive CNN ) or  **`Different_Kernel.ipynb`** ( Different Kernel ) or **`Data_Arg.ipynb`**( data argumentation )

## Ensemble Learning

Run the **Ensemble_Learn_preprocess.ipynb** to obtain the training sets of models.

Move these files and the weight of best model we trained in `difker_model`(**difker_3020e.h5**) to the folder `Ensemble Learning` .

Run the  **Emsemble_learning.ipynb** in this folder to train 10 models.

Run the **Emsemble_test.ipynb** to evaluate the performance of model.