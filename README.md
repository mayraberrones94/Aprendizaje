# Automated learning
Repository for my automated learning's course. The course description and activities can be found [here](https://github.com/satuelisa/StatisticalLearning) 

---

## **Homework 1: Introduction**

> **Instructions:** Identify one or more learning problems in your thesis work and identify goals and elements.

Reading the first chapter of the recommended book, we saw several components that could be applied to our thesis work. How can we differentiate between supervised and unsupervised learning, and when to use each, with different types of techniques. 

In our case, our main objective is to use computational tools to help classify and diagnose breast cancer images. Our input data consists of four data sets, of which only three have been made publicly available.

For this objective we could identify the following problems and elements:

- **Classifying images into normal and abnormal:** This requires having a separate data set, where the model knows which images belong to the normal and abnormal sets, and corrects the output with each iteration. Here we need only two types of output, where the ones closest to 1 are determined as abnormal, and the ones closest to 0 are normal.

- **Segmentation of the image:** Here is where we single out the data in the image that we find to be abnormal. The output expected from this could be a patch of the image the model considered to be abnormal. The input of this would be the images that are considered abnormal from our last model.

- **Classifying anomalies:** The input of this step would be the patches of images with the different anomalies. The model then has to classify into their separate categories of possible malignancy with the BIRADs scale that goes from 0 to 6 different categories (number 4 having another three scales of low, moderate, and high). 

And finally, we have discussed more recently that a separation of the data set by density of breast tissue could potentially help us reduce the false negatives and false positives in our models so:

- **Classifying density of breast tissue:** The input on this problem would be the complete image, and as an output, it would classify into the four different densities of tissue stated by the [ACS](https://www.cancer.org/cancer/breast-cancer/screening-tests-and-early-detection/mammograms/breast-density-and-your-mammogram-report.html).  

---


## **Homework 2: Supervised learning**

> **Instructions:** First carry out Exercise 2.8 of the textbook with their ZIP-code data and then replicate the process the best you manage to some data from you own problem that was established in the first homework.


**Ex. 2.8:** Compare the classification performance of linear regression and k-nearest neighbor classification on the ZIP-code data. In particular, consider only the 2's and 3's and k = 1, 3, 5, 7 and 15. show both training and test error for each choice. The zipcode data are available from the book [website](https://hastie.su.domains/ElemStatLearn/datasets/).

For this example, we use the ZIP-code dataset for [training](https://github.com/mayraberrones94/Aprendizaje/blob/main/Datasets/zip.train) and [testing](https://github.com/mayraberrones94/Aprendizaje/blob/main/Datasets/zip.test). We begin by separating the data, focusing on elements 2 and 3, as indicated in the instructions.

```python
import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

def filtered_data(path):
    data_all = np.loadtxt(path)
    mask = np.in1d(data_all[:, 0], (2, 3))
    data_x = data_all[mask, 1: ]
    data_y = data_all[mask, 0]
    return data_x, data_y

train_x, train_y = filtered_data('/content/drive/MyDrive/Datasets/zip.train')
test_x, test_y = filtered_data('/content/drive/MyDrive/Datasets/zip.test')
```

Then we determine the k points we are going to be using:

```python
k_points = [1, 3, 5, 7, 15]      
```

### Linear model:

First we import all the libraries:

```python
from pandas import read_csv

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score    
```

Then we call the LinearRegresion function to train our model:

```python
#https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/

rmse_val = [] #to store rmse values for different k
rmse_valtrain = []
for K in k_points:
    model = LinearRegression()

    model.fit(train_x, train_y)  #fit the model
    pred_train=model.predict(train_x) #make prediction on train set
    pred=model.predict(test_x) #make prediction on test set
    error_train = sqrt(mean_squared_error(train_y,pred_train)) #calculate rmse
    error = sqrt(mean_squared_error(test_y,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    rmse_valtrain.append(error_train)
    print('RMSE value for train set k = ', K, 'is:', error_train)
    print('RMSE value fortest set k= ' , K , 'is:', error)
    print('\n')     
```
The value of K does not really interviene at this point, but we have them in the [Notebook]() for this exercise, algonside the corresponding plot.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/Linearplot_1.png)


