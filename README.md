# Automated learning
Repository for my automated learning's course. The course description and activities can be found [here](https://github.com/satuelisa/StatisticalLearning) 
---

+ [Chapter 1: Introduction](#homework-1-introduction)
+ [Chapter 2: Supervised learning](#homework-2-supervised-learning)
+ [Chapter 3: Linear Regresion](#homework-3-linear-regresion)
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
The value of K does not really interviene at this point, but we have them in the [Notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/Homework2_ex2_8.ipynb) for this exercise, algonside the corresponding plot.

```python
fig = plt.figure(figsize = (8, 8))
axes = fig.add_subplot(1, 1, 1)

axes.plot(k_points, rmse_val, '-', color = 'orange', label = 'linear-train')
axes.plot(k_points, rmse_valtrain, '-', color = 'green', label = 'linear-test')

axes.legend()
axes.set_xlabel("k")
axes.set_ylabel("Error")
plt.show()
```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/Linearplot_1.png)

### K-Nearest neighboor model:

Following a similar procedure as in the LinearRegresion, we use the sklearn library to call KNeighborsRegresion. In this case, the K points are going to change the results.

```python
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt

rmse_val = [] #to store rmse values for different k
rmse_valtrain = []
for K in k_points:
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

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
And in our plot we see how it is not just a line:

```python
fig = plt.figure(figsize = (8, 8))
axes = fig.add_subplot(1, 1, 1)

axes.plot(k_points, rmse_val, '-', color = 'orange', label = 'knn-train')
axes.plot(k_points, rmse_valtrain, '-', color = 'green', label = 'knn-test')

axes.legend()
axes.set_xlabel("k")
axes.set_ylabel("Error")
plt.show()
```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/knnplot1.png)

### Linear and knn models for our data:

In this examples we are going to use the dataset MiniMias, which is the smallest dataset and a public one. It can be downloaded from [here](https://www.kaggle.com/kmader/mias-mammography). Since we are dealing with images, we wanted to try something we had considered before to reduce the file size of our datasets, and that is to convert our images into csv files. 

In our first try, with the images of size 1024x1024 pixels, the code to turn images to csv took too much time and RAM resources (using google colab), so we decided to first change the size of the images to a smaller one.

```python
import cv2
from imutils import paths
import os

Hg = 200
Lng = 80

imagePaths = list(paths.list_images('/content/drive/MyDrive/BD/Minimias/MIAS_Normal'))
data = []
labels = []
i = 0
for imagePath in imagePaths:
    i = i + 1
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (Hg, Lng))
    try:
      cv2.imwrite('/content/drive/MyDrive/BD/Minimias/MIAS_sa/{}mias.png'.format(i), image)
    except AttributeError:
      print("Not found {}".format(img))
```

We repeat this procedure with the file for the images with anomalies. Now that they are smaller, is faster to turn them into csv files (and the files do not take forever to load afterwards).

```python

#https://stackoverflow.com/questions/49070242/converting-images-to-csv-file-in-python
from PIL import Image
import numpy as np
import sys
import os
import csv

#Useful function
def createFileList(myDir, format='.png'):
  fileList = []
  print(myDir)
  for root, dirs, files in os.walk(myDir, topdown=False):
    for name in files:
      if name.endswith(format):
        fullName = os.path.join(root, name)
        fileList.append(fullName)
  return fileList
```
The full code can be found [here](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/Aprendizaje_2.ipynb). Parallel to converting the images to a csv file, we created a list where we appended a 0 for the normal images and a 1 for the images with anomalies.

Using the sklearn library we separated the data into training and test sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(read_file, y_label, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, len(y_train), len(y_test))

```

Then using the codes we used for the exercise we have the following plots:

Linear error

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/linear2.png)

Knn error:

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/knn2.png)

### Comments and conclusions:

For this second homework we were tasked to complete one exercise, but also to try and recreate it with one of the problems we discussed in the first homework. For this, we wanted to experiment with one of the projects we had in mind, and that was to turn images into csv sets. We decided to try it with the Mini MIAS data set because is public and the smallest one. (A small dataset was important because of the processing needed to turn the images).

The regular scoring for all the problems discussed in HW1 just focus on the accuracy score, but it would be interesting to go back to these messurements of error for the results of the regular model training we do (CNN). 

## **Homework 3: Linear Regresion:**

> **Instructions:** Repeat the steps of the prostate cancer example in Section 3.2.1 using Python, first as a uni-variate problem using the book's data set and then as a multi-variate problem with data from your own project. Calculate also the p-values and the confidence intervals for the model's coefficients for the uni-variate version. Experiment, using libraries, also with subset selection. 

For this problem we are using the prostate cancer dataset. The data came from a study that examines the correlation between the level of prostate specific antigen (PSA) and number of mesures. The goal is to predict the log of PSA (lpsa).

Here is a brief description of the data:

Column name | Description
-------------------|------------------
lpsa       | Log of prostate specific antigen
lcavol     | Log cancer volume
lweight     | Log prostate weight
age     | Age of patient
lbph     | Log of benign prostatic hyperplasia
svi     | Seminal vesicle invarion
lcp     | Log of capsular penetration
gleason     | Gleason score
pgg45  |Percent of Gleason scores 4 or 5



First we load our [dataset](https://github.com/mayraberrones94/Aprendizaje/blob/main/Datasets/prostate.data):

```python
import numpy as np
import pandas as pd

from scipy import stats

dataset = pd.read_csv('/content/drive/MyDrive/Datasets/prostateData.csv')
print(dataset.shape)

dataset.drop('id', axis=1, inplace=True)
```

Reading further into the example of prostate cancer we realize, since the goal is to examine the correlation of `lpsa`, so this becomes our target variable. All the other columns are the features that we need to analyse in order to see which ones will help us in our models. The `train` coulmn marks the division between our training and data set. 

Next up we divide and name our training and test sets and we begin with the correlations of the variables .

```python
target = 'lpsa'
# based on the following features
features = ['lcavol', 'lweight', 'age', 'lbph',
            'svi', 'lcp', 'gleason', 'pgg45']

train_st = dataset.train == 'T'
X, y = dataset[features].values, dataset[target].values
X_train, y_train = X[train_st], y[train_st]
X_test, y_test = X[~train_st], y[~train_st]

df_corr = dataset[train_st].corr()
```

|        |	lcavol| lweight|   age|	lbph   |svi	  |lcp	   |gleason	   |pgg45|
|--------|---------|--------|------|--------|------|--------|-----------|--------|
|lweight |	0.300 |	       |	  |	       |	  |	       |           |   |
|age     |	0.286 |	0.317 |	  |		   |	  |        |           |	|		|	
|lbph	| 0.063   |	0.437  | 0.287|		   |	  |		   |           ||
|svi     |	0.593 |	0.181  |0.129 |	-0.139 |	  |		   |	       ||
|lcp	    |0.692	  |0.157   |0.173 |	-0.089 |0.671 |		   |	       ||
|gleason	|0.426	  |0.024   |0.366 |	0.033  |0.307 |	0.476  |		   ||
|pgg45	|0.483	  |0.074   |0.276 |	-0.03  |0.481 |	0.663  |	0.757  |	|
|lpsa	|0.733	  |0.485   |0.228 |	0.263  |0.557 |	0.489  |	0.342  |	0.448|

Table 3.1 in the book shows the correlation of predictors in the dataset. We wanted it to look similar to what we had in the book, so we print it as a traingle correlation matrix. For the entire code, refer to the [notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/HW_3_Univariant_example.ipynb) 

Seeing this table we can already tell which variables are strongly correlated. In the book they mention a scatterplot matrix that show that the svi variable is binary, and gleason is an ordered categorical value. So we plot this matrix with the help of seaborn.

```python
import seaborn as sns

sns.set_theme(style = "ticks")
sns.set(font_scale=1.6)
sns.pairplot(dataset)
```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/sns_plot.png)

Then for table 3.2, they fit a linear model to the lpsa after first standarizing the predictors to have unit variance.

```python
#https://scikit-learn.org/stable/modules/preprocessing.html

from sklearn import preprocessing
import numpy as np

scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#https://www.statsmodels.org/0.6.1/examples/notebooks/generated/ols.html
import statsmodels.api as sm

model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
ls_params = model.params

result = zip(['Intercept'] + features, ls_params, model.bse, model.tvalues)
print('      Term   Coefficient   Std. Error   Z Score')
print('-----------------------------------------------')
for term, coefficient, std_err, z_score in result:
    print(f'{term:>10}{coefficient:>14.2f}{std_err:>13.2f}{z_score:>10.2f}')
```

The resulting table shows the coefficient, std. error and Z score of each feature.

|Term  | Coefficient  | Std. Error  | Z Score|
|-------|-------------|-------------|--------------|
 |Intercept |         2.46  |       0.09 |    27.60|
 |   lcavol |         0.68 |        0.13 |     5.37|
 |  lweight |         0.26 |        0.10 |     2.75|
 |      age |        -0.14 |        0.10 |    -1.40|
 |     lbph |         0.21 |        0.10 |     2.06|
  |     svi |         0.30 |        0.12 |     2.47|
  |     lcp |        -0.29 |        0.15  |   -1.87|
  | gleason  |       -0.02  |       0.14  |   -0.15|
   |  pgg45  |        0.27  |       0.15  |    1.74|
   
In the book they mention that the Z-scores mesure the effect that the model will have if we drop a variable. A Z-score greater than 2 in absolute value is approx. significant at 5% level. Based on the Z-score we can consider `age`, `lcp`, `gleason`, and pgg45 as variables we can get rid off.

They mention the F-statistic to test for the exclusion of several features at the same time. The F-statistic mesures the change in residual sum of squares per aditional parameter in the bigger model. 

We have as F-statistic:

$F = \frac{(RSS_0 - RSS_1) / (p_1 - p_0)}{RSS_1/(N-p_1-1)}$   

Repeating the code from above but only with the features that have significance in the Z score we have the resulting table (complete code in notebook):

|Term   |Coefficient  | Std. Error  | Z Score|
|---------|-------------|------------|-------------|
| Intercept|          2.46  |       0.09 |    27.60|
|    lcavol |         0.68 |        0.13  |    5.37|
|   lweight |         0.26  |       0.10  |    2.75|
|      lbph |        -0.14  |       0.10  |   -1.40|
|       svi  |        0.21  |       0.10  |    2.06|

We can now calculate the residual sum for both of this results and use the F-statistic formula.

```python
rss0 = sum((model1.predict(sm.add_constant(X_train1)) - y_train1) ** 2)
p1 = len(features) + 1
p0 = len(features1) + 1
N = len(y_train)
f_statistic = ((rss0 - rss1)/ (p1 - p0))/(rss1 / (N - p1 - 1))
prob = 1 - stats.f.cdf(f_statistic, (p1 - p0), (N - p1) )
```

```
RSS01 =  29.4263844599084
RSS0 =  32.81499474881555
F =  1.6409660073161834
p-value =  0.1762476548551386
```

