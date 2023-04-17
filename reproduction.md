# Reproduction

If you want to reproduce our result, please follow the steps below:

Please make sure that the required files and the target program are in the same folder before running the program.

## KNN

In KNN part, we only used the first part of our dataset with about 16000 photos.

Just run the **`KNN part/KNN.ipynb`** ,similar result will be shown.



---



Before trying other methods except KNN, please run the **`Preprocess/Get array and label.ipynb`** to Generate the files including **`X`,`y`,`X_train`,`y_train`,`X_test`,`y_test`**.

## HOG+SVM

Run the **`SVM part/HOG part/HOG_SVM.ipynb`**.

In first part, the code will generate **`hog_features_train.npy`** and **` hog_features_test.npy`**.

You can use them to reproduce the second part which check the performance of SVM with different parameters(`kernel` and `C value`).

## SIFT+SVM

Run the **`SVM part/SIFT part/SIFT_SVM.ipynb`** .

In first part, the code will generate **`X_train_descriptors`** , **` y_train_labels`**,**`X_test_descriptors`** and **` y_test_labels`**.

However it does not performance well.( It dose not have value to reproduce.)

## SIFT+ Neural Networks

First part, same as **`SIFT+SVM`**, to obtain 4 files.

Then, run the **`SVM part/SIFT part/SIFT_Neuret.ipynb`**.



## CNN

The folder `CNN part/naive_CNN_model` contains the model of different epochs with the method of naive CNN.

The folder `CNN part/difker_model` contains the model weight of different epochs with the method of using different kernels in CNN.

And  `CNN part/DataArgumentation_model` the model weight after using argued data to train the model.

If reproducing, just run **`CNN part/CNN.ipynb`**( naive CNN ) or  **`CNN part/Different_Kernel.ipynb`** ( Different Kernel ) or **`CNN part/Data_Arg.ipynb`**( data argumentation )

## Ensemble Learning

Run the **CNN part/Ensemble Learning/Ensemble_Learn_preprocess.ipynb** to obtain the training sets of models.

Move these files and the weight of best model we trained in `difker_model`(**difker_3020e.h5**) to the folder `Ensemble Learning` .

Run the  **CNN part/Ensemble Learning/Emsemble_learning.ipynb** in this folder to train 10 models.

Run the **CNN part/Ensemble Learning/Emsemble_test.ipynb** to evaluate the performance of model.

Run the **CNN part/Ensemble Learning/Voting** to train voting model.