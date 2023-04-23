# Cat-Dog-Classifier

## Introduction

Image classification is a machine learning problem that involves categorizing images based on their visual characteristics. In image classification, an algorithm learns to recognize patterns and features associated with each category and define the category of image. 

Image classification has widespread applications in many fields, including facial recognition, object detection, medical imaging, and more.

In our project, we choose a binary classification problem to classify whether the picture is more likely a cat or a dog.

We used a series of methods to solve this problem including traditional computer vision method like HOG and SIFT and novel method like CNN.

---

## Instruction and Document

![general](/mdpic/general_structure.png)

### EDA (Pre-processing)

Exploratory data analysis and preprocess we do for data.

[EDA&Preprocess Document](/EDA&Preprocess.md)

### Methodology

Machine learning techniques used.

[Methodology Document](/methodology.md)

### Reproduction

Tell you how to reproduce our project.

[Reproduction Document](/reproduction.md)



---

## Best Model Performance

| Metrics   | Train    | Test     |
| --------- | -------- | -------- |
| Accuracy  | 0.999614 | 0.940046 |
| Precision | 1        | 0.950562 |
| Recall    | 0.99922  | 0.929058 |
| F1 score  | 0.99961  | 0.939687 |

![CM](/mdpic/CNNcm.png)

![ROC](/mdpic/CNNROC.png)

---

## Conclusion

- Traditional models' performance may not relate to the size of data( the performances in small dataset and large one are similar.) But Neural Network's performance is obviously related to size of data.
- The input with more information is more precise in model training.
- CNN model performance is obviously better than the traditional method( maybe that is the reason why they become a `traditional` method).
- The running time of traditional method is much more longer than the CNN model. Some of traditional methods have a time complexity of O(n^2).
- Dropout and regulation can help to avoid over-fitting (our best model trained more than 3000 epochs and not over-fitting), but also may decrease the learning speed of model.
- Use different types of kernels in a convolution layer can capture more useful features.
- Data Argumentation does not always work, maybe the argumentation process lead to a bias of data.
- Ensemble Learning can performance better than one single model, but not the under-fitting models. And it need to learn how to vote.



---

## What did we learn from this project?

- Collaborating using GitHub

- Use cmd commands to batch process files

- Use Git to maintain the repository

- Different feature description method, like SIFT and HOG

- Machine learning algorithms, including KNN and SVM

- Convolution Neural Network and its improvement

-  Ensemble learning method

- `steganographia` and `cryptography` (it might be strange but we use it to protect our originality)

  

---



## Display

You can try to input your own photo and test whether it is considered as a dog or a cat.

You do not need to reproduce by yourself.

Open the **Display/display_version.ipynb** and put your own photo of a cat or a dog.

```
img=cv.imread('photo3.jpg',0) #change it to your own photo name
```

Change the `photo3.jpg` to your own photo name.

Then, run the code cell by cell. The cell below will print the result.

```
#show result，dog is 0 and cat is 1
print(prediction)
```



---

## Contributors

@clintonluvs29 - data preprocess, video making

@kengle-18 - CNN, KNN

@Rechard16 - traditional method, CNN improvement

---



## References

[Link to our video!](https://youtu.be/lzi8So_inkQ)


Datasets:

[Dog and Cat Image Dataset](https://www.kaggle.com/competitions/dogs-vs-cats)

[Cat Image Dataset](https://www.kaggle.com/datasets/crawford/cat-dataset)

[Dog Image Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

[CatvRabbit Image Dataset](https://www.kaggle.com/code/utkarshsaxenadn/catvsrabbit-classification-mobilenet-acc-100/input)



Techniques used:

[Learning CNN](https://www.youtube.com/watch?v=WvoLTXIjBYU); [Understanding CNN](https://towardsdatascience.com/understand-the-architecture-of-cnn-90a25e244c7); [Tips on CNN Image Classification](https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/); [CNN](https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7); [More CNN](https://medium.com/@himadrisankarchatterjee/a-basic-introduction-to-convolutional-neural-network-8e39019b27c4); [CNN in detail](https://www.ibm.com/topics/convolutional-neural-networks)

[HOG technique](https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f); [HOG_SVM technique](https://github.com/CHNicelee/HOG_SVM); [About SVM](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/); [SIFT Technique](https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/) 

[Explanation of KNN](https://www.ibm.com/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20algorithm%2C%20also%20known%20as%20KNN%20or,of%20an%20individual%20data%20point)

[Deep Learning](https://b23.tv/8uak8EH); [Backpropagation](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)

[Ensemble Learning](https://www.inquirer.com/life/upbringing-teamwork-learning-together-20201227.html); [AUCROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) 



Information in video:

[Application in Autonomous Vehicles](https://www.fia.com/autonomous-vehicles); [Cameras](https://computer.howstuffworks.com/internet/tips/facebook-photo-tags.htm#:~:text=The%20site%20can%20do%20this,does%20often%20work%20very%20well); [Drones](https://www.robotics247.com/article/precision_ai_raises_20m_for_drone_based_computer_vision_for_sustainable_agriculture)

[How Image Classification Came About](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/); [2013 Kaggle Dog vs Cat Competition](https://www.kaggle.com/c/dogs-vs-cats); [Pierre Sermanet CNN Paper](https://arxiv.org/abs/1312.6229)



Images used:

[Camera](https://snapshot.canon-asia.com/article/eng/how-to-maximise-your-cameras-vari-angle-monitor); [Cat](https://icatcare.org/advice/thinking-of-getting-a-cat/); [Dog](https://unsplash.com/s/photos/baby-dog); [CAPTCHA](https://stytch.com/blog/what-is-captcha/); [Turing test](https://www.techtarget.com/searchenterpriseai/definition/Turing-test); [HOG technique](https://www.researchgate.net/figure/HOG-calculation-a-gradients-in-a-cell-b-histogram-of-gradients_fig1_269074001); [ReLu](https://www.semanticscholar.org/paper/Convexity-in-Image-Based-3D-Surface-Reconstruction-Kolev/b1df0cd796034ca9ba3bc018474e44ee60fd7855); [Padding](https://datahacker.rs/what-is-padding-cnn/); [Data Augmentation Cat](https://wikidocs.net/166068); [Cat2](https://www.bbc.com/future/article/20230217-should-new-zealand-cats-be-kept-indoors); [Easter egg](https://pngtree.com/so/easter-eggs)



Others:

[Magic Tool](http://www.atoolbox.net/Tool.php?Id=857)

---

<img src="/mdpic/egg.jpg" alt="e" style="zoom:10%; float: left;" />
