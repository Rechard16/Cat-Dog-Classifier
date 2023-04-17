# Cat-Dog-Classifier

## Introduction

Through image recognition of the image dataset, we hope to use Convolution Neural Network (CNN) to classify images into 'cat' and 'rabbit' categories.



---



## EDA (Pre-processing)

### 1) Data-cleansing

Removed some abnormal data, including irrelevant data and misclassified data.

### 2) Analysis of the size of photos and deciding how to best resize them

We analysed the size of pictures.

<img src="/mdpic/size_distribution_cat.png" alt="cat1" style="zoom:50%;" />

cat height and weight distribution (boxplots)

<img src="/mdpic/weight_height_cat.png" alt="weight_height_cat" style="zoom:50%;" />

cat height and weight scatter plot

<img src="/mdpic/size_distribution_dog.png" alt="dog" style="zoom:50%;" />

dog height and weight distribution (boxplots)

<img src="/mdpic/weight_height_dog.png" alt="weight_height_dog" style="zoom:50%;" />

dog height and weight scatter plot

We cut the photos into size of 128*128 so that we can reduce the computing difficulty while preserving essential information of the previous photo file.

### Data skewness

We have 22497 cat pictures and  22786 dog pictures, which only have a 1.2% difference between them. 

Thus, it is not necessary to process class imbalance.

### Random selection of photos for checking

We then randomly chose 20 photos from each category to check if our process has badly influenced our raw data set.

#### Cat

![cat](/mdpic/catsample.png)

### Dog

![dog](/mdpic/dogsample.png)

---

## Methodology

Machine learning techniques used.

[Methodology Document](/methodology.md)



---

## Reproduction

Reproduction document.

[Reproduction Document](/reproduction.md)



---



## Display

You can try your own photo to test whether it is considered as a dog or a cat.

Open the **display_version.ipynb** and put your own photo of a cat or a dog.

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



<img src="/mdpic/egg.jpg" alt="e" style="zoom:10%; float: left;" />



--- 



## References
Datasets:

https://www.kaggle.com/datasets/crawford/cat-dataset

https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

https://www.kaggle.com/c/dogs-vs-cats

https://www.kaggle.com/code/utkarshsaxenadn/catvsrabbit-classification-mobilenet-acc-100/input

https://www.kaggle.com/competitions/dogs-vs-cats

Techniques used:

https://www.youtube.com/watch?v=WvoLTXIjBYU 

https://github.com/CHNicelee/HOG_SVM

http://t.csdn.cn/MnF6B 

https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/

https://www.analyticsvidhya.com/blog/2014/12/image-processing-python-basics/ 

https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

https://arxiv.org/abs/1312.6229 

https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00471-5 

Extra information in video:

https://www.robotics247.com/article/precision_ai_raises_20m_for_drone_based_computer_vision_for_sustainable_agriculture

https://www.fia.com/autonomous-vehicles
