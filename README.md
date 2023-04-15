# Cat-Dog-Classifier

## Introduction

Through image recognition of the image dataset, we hope to use Convolution Neural Network (CNN) to classify images into 'cat' and 'rabbit' categories. We used CNN network. We also applied the HOG model to the code.



---



## EDA & Pre-process

### clean data

Removed some abnormal data, including irrelevant data and misclassified data.

### photo size analysis

Then, we analyzed the size of pictures.

<img src="/mdpic/size_distribution_cat.png" alt="cat1" style="zoom:50%;" />

cat height and weight distribution

<img src="/mdpic/weight_height_cat.png" alt="weight_height_cat" style="zoom:50%;" />

cat height and weight scatter plot

<img src="/mdpic/size_distribution_dog.png" alt="dog" style="zoom:50%;" />

dog height and weight distribution

<img src="/mdpic/weight_height_dog.png" alt="weight_height_dog" style="zoom:50%;" />

 dog height and weight scatter plot

We cut the photos into size of 128*128 for decreasing computing difficulty and reserving enough information of previous photo.

### Data skew

We have 22497 cat pictures and  22786 dog pictures, only has a 1.2% difference. 

So it is not necessary to process class imbalance.

### Random choose to recheck

We randomly choose 20 photos from each category ,to check whether our process has a bad influence on previous data set.

#### Cat

![cat](/mdpic/catsample.png)

### Dog

![dog](/mdpic/dogsample.png)

---

## Methodology

[Methodology Document](/methodology.md)



---

## Reproduction

[Reproduction Document](/reproduction.md)



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



---



<img src="/mdpic/egg.jpg" alt="e" style="zoom:10%; float: left;" />
