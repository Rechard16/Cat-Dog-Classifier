# EDA (Pre-processing)

## 1) Data-cleansing

Removed some abnormal data, including irrelevant data and misclassified data.

## 2) Analysis of the size of photos and deciding how to best resize them

We analysed the size of pictures.

<img src="/mdpic/size_distribution_cat.png" alt="cat1" style="zoom:50%;" />

cat height and weight distribution

<img src="/mdpic/weight_height_cat.png" alt="weight_height_cat" style="zoom:50%;" />

cat height and weight scatter plot

<img src="/mdpic/size_distribution_dog.png" alt="dog" style="zoom:50%;" />

dog height and weight distribution

<img src="/mdpic/weight_height_dog.png" alt="weight_height_dog" style="zoom:50%;" />

dog height and weight scatter plot

We cropped the photos into size of 128*128 so that we can reduce the computing difficulty while preserving essential information of the previous photo file.

## 3)Data skewness

We have 22497 cat pictures and  22786 dog pictures, which only have a 1.2% difference between them. 

Thus, it is not necessary to process class imbalance.

## 4)Random selection of photos for checking

We then randomly chose 20 photos from each category to check if our process has badly influenced our raw data set.

### Cat

![cat](/mdpic/catsample.png)

### Dog

![dog](/mdpic/dogsample.png)