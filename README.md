# Automated learning
Repository for my automated learning's course. The course description and activities can be found [here](https://github.com/satuelisa/StatisticalLearning) 
---

+ [Chapter 1: Introduction](#homework-1-introduction)
+ [Chapter 2: Supervised learning](#homework-2-supervised-learning)
+ [Chapter 3: Linear Regresion](#homework-3-linear-regresion)
+ [Chapter 4: Linear methods for Classification](#homework-4-linear-methods-for-classification)
+ [Chapter 5: Basis expansion and regularization](#homework-5-basis-expansion-and-regularization)
+ [Chapter 6: Kernel smoothing methods](#homework-6-kernel-smoothing-methods)
+ [Chapter 7: Model assesment and selection](#homework-7-model-assesment-and-selection)
+ [Chapter 8: Model inference and averaging](#homework-8-model-inference-and-averaging)
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

Now that we have all the examples from the book and the p-values, we see the coeficient intervals of our model.

```python
coef_intervals = model.conf_int(0.05)
low = []
high = []
for (x, y) in coef_intervals:
  low.append(x)
  high.append(y)

data = {'Term':  ['Intercept','lcavol', 'lweight', 'age', 'lbph',
            'svi', 'lcp', 'gleason', 'pgg45'],
        'Low coef': low,
        'High coef': high}

df = pd.DataFrame(data)
df
```
 And we have the following table as a result:
 
| 	|Term|	Low coef	|High coef|
|---|-----|-----------|----------------- |   
|0	|Intercept|	2.286150|	2.643716|
|1	|lcavol	|0.423851|	0.928182|
|2	|lweight|	0.071262|	0.452125|
|3	|age|	-0.342544|	0.061077|
|4	|lbph|	0.005504	|0.412617|
|5	|svi|	0.057489|	0.549758|
|6	|lcp	|-0.594727	|0.020724|
|7	|gleason|	-0.310436|	0.268046|
|8	|pgg45	|-0.040326|	0.571478|

Lastly, the instructions asks us to experiment a little bit with libraries and subset selection. Thanks to the page provided in the courses page, we have very good examples to start with. All the experimentations where made in the notebook, as we think we did not have too much variables. The subset selection comes in handy on the next part of the experiment, when we are using our own data.

### Breast cancer dataset:

For our data (all our image datasets) we did not find features that would match the things asked for this experimentation, so we turn to a similar data set that could actually help us understand better the correlations between certain features. In this case this data set is usally used for classification and prediction processes. 

The data set can be found in [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). Same as before, we start by setting up our data set.

Here is a brief description of the data:

Column name | Description
-------------------|------------------
radius       | Mean of distances from center to points on the perimeter
texture    | Standard deviation of gray scale values
perimeter     | Perimeter of anomaly
area     | Area of anomaly
smoothness     | Local variation in radius lenght
compactness     | perimeter^2 /area - 1.0
concavity     | Severity of concave portions of the contour
concave points     | Number of concave portions of the countour
symmetry  | Symmetry of anomaly
fractal dimension | Costline aproximation - 1

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.



In order to be able to use it with the same propose as the data set form the book, we took other variable as the target variable. Instead on focusing on the diagnosis we focus on the `radius_mean`. The reason why we did not use diagnosis as our target was because, after some [research](https://towardsdatascience.com/the-difference-between-classification-and-regression-in-machine-learning-4ccdb5b18fd3) we realized that we where seeing the problem as a classification problem, and not a regression one. What can this variable help us in our data set?

The Mini-MIAS data set has a companion [document](http://peipa.essex.ac.uk/info/mias.html) where they detail the center of the anomaly, and the aproximate radius of the circle that encompases the abnormality. With this analysis we expect to find which other features can be necesary to develop a similar file to the breast cancer of wisconsin, but with the Mini-Mias images. This could give us an idea also in the segmentation part of our task.

As described in the table above, all the features have mean, standard error and worst or largest as aditional features. Fow this, in our data set we use `radius_mean` as target value, and `radius_se`, and `radius_worst`.

```python
X1 = dataset.drop(columns=['radius_mean', 'radius_se', 'radius_worst'])
columns = X1.columns
from sklearn.model_selection import train_test_split 
target = 'radius_mean'

X, y = dataset[features].values, dataset[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

We keep diagnosis as a feature in the data set, so we can plot a correlation plot with the added tool of the diagnosis separation with seaborn with only the core parameters.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/sns_diagnosis.png)

Following the same steps as before we get the coeficients, Std error and Z score of our remaining features. (Complete code in [notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/HW3_Breastcancer.ipynb))

|                   Term|   Coefficient  | Std. Error|   Z Score|
|------------------------|--------------|--------------|-----------|
 |            Intercept  |        0.13   |      0.15  |    0.86|
|             diagnosis |         0.04    |     0.01  |    2.84|
|          texture_mean |        -0.00   |      0.00  |   -0.50|
 |       perimeter_mean  |        0.16   |      0.00  |   61.72|
|             area_mean |        -0.00   |      0.00 |    -4.05|
|       smoothness_mean  |        1.91  |       0.70  |    2.72|
|      compactness_mean  |       -4.59   |      0.42   | -10.94|
|        concavity_mean  |       -2.42  |       0.35   |  -6.88|
|   concave points_mean  |        0.69  |       0.73  |    0.95|
|         symmetry_mean  |        0.24  |       0.26   |   0.93|
|fractal_dimension_mean  |        2.54   |      1.98   |   1.28|
|            texture_se  |       -0.01  |       0.01   |  -0.74|
|          perimeter_se  |        0.00   |      0.01   |   0.08|
|               area_se  |       -0.00   |      0.00  |   -1.56|
|         smoothness_se  |        1.50   |      2.25   |   0.67|
|        compactness_se  |       -0.96  |       0.72   |  -1.32|
|          concavity_se  |        2.84  |       0.43   |   6.63|
|     concave points_se  |       -4.09  |       1.91   |  -2.14|
|           symmetry_se   |       2.05  |       0.98   |   2.08|
|  fractal_dimension_se   |      -1.05   |      3.77    | -0.28|
|         texture_worst   |       0.00  |       0.00    |  0.44|
|       perimeter_worst   |      -0.01  |       0.00  |   -4.23|
|            area_worst   |       0.00  |       0.00   |   5.26|
|      smoothness_worst   |      -0.43  |       0.49  |   -0.86|
|     compactness_worst   |       0.21  |       0.13   |   1.60|
|       concavity_worst   |      -0.05  |       0.09  |   -0.49|
|  concave points_worst   |       0.68  |       0.33   |   2.08|
|        symmetry_worst   |      -0.04 |        0.17  |   -0.26|
|fractal_dimension_worst   |       0.23  |       0.85  |    0.28|

Seeing this results, we can see which features we can get rid of some features that do not have enough significance. Then, repeating the procedure of the first part, we separate the features and get the F-statistic score:

|    Term  | Coefficient  | Std. Error   |Z Score|
|-----------|--------------|-------------|---------------------|
 |          Intercept  |        0.13 |        0.15 |     0.86|
 |          diagnosis  |        0.04  |       0.01 |     2.84|
|      perimeter_mean  |       -0.00  |       0.00 |    -0.50|
 |    smoothness_mean  |        0.16  |       0.00 |    61.72|
 |       concavity_se  |       -0.00  |       0.00  |   -4.05|
|         symmetry_se  |        1.91 |        0.70 |     2.72|
|          area_worst  |       -4.59  |       0.42 |   -10.94|
|concave points_worst  |       -2.42  |       0.35 |    -6.88|

```python
from scipy import stats

rss1 = sum((model.predict(sm.add_constant(X_train)) - y_train) ** 2)
rss0 = sum((model1.predict(sm.add_constant(X_train1)) - y_train1) ** 2)
p1 = len(features) + 1
p0 = len(features1) + 1
N = len(y_train)

f_statistic = ((rss0 - rss1)/ (p1 - p0))/(rss1 / (N - p1 - 1))
prob = 1 - stats.f.cdf(f_statistic, (p1 - p0), (N - p1) )
```

```
RSS01 =  2.3867283504900967
RSS0 =  10.44504733766965
F =  68.32994927470597
p-value =  1.1102230246251565e-16
```

We tried the code of the best subset regression from the page recomended in the homework, just to check if it matched what we did so far, but with all the variables we gave the code 30 min and it did not finish, so we gave only the last significant variables and this is the result:

|index |num_features   |            features   |    MAE|
|------|----------------|----------------------|-----------|
|0      0  |          6 |    [1, 2, 3, 4, 5, 6] | 0.104859|
|1      0 |           7 | [0, 1, 2, 3, 4, 5, 6] | 0.105014|
|2      0  |          5 |       [1, 2, 3, 4, 6] | 0.105133|
|3      0  |          6  |   [0, 1, 2, 3, 4, 5] | 0.105610|
|4      0   |         6  |   [0, 1, 2, 3, 4, 6]|  0.105637|

```
Best Subset Regression MAE: 0.105
Best Subset Regression coefficients:
{'Intercept': 1.264,
 'area_mean': 0.0,
 'diagnosis': 0.148,
 'perimeter_mean': -3.346,
 'radius_mean': -4.975,
 'smoothness_mean': -1.92,
 'texture_mean': -1.927}
 ```

In the end we also tested some other libaries and compared the different shrinking and subset methods. We compared OLS, Ridgde, Lasso and Elastic net, all of them having a very small error on the test set. (Code in notebook)

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/rmse.png)

### Conclusions:

Looking closer at the variables that are more influential and the description of each one we can see why they perform so well together. In this case we have `diagnosis`, `perimeter_mean`, `smoothness_mean`, `concavity_se`, `symmetry_se`, `area_worst` and `concave points_worst` as the most influential features. Perimeter and area seem the most apropiate since we had seen in the scatterplot how strongly they where correlated. Smoothness refers to the shape of the anomaly. Normally anomalies do not resemble a perfect circle, so the smothness and concavity tell us about the imperfections in the encircling area. 

Now that we know all of this parameters, we can try to compose our own dataset with the mini-MIAS, filling the missing values with the original image, which could give us an idea in how to perfect the segmentation process that we had in mind.

NOTE: We tried but failed to fully comprehend and form a multivariate problem with our original dataset of images with the loss and accuracy as our pedicting variables, mainly because we did not understad which features could we represent to be able to do all of the experiments above. Most of the variables we had where categorical (big, small, names of optimizers, activation funtions) so instead we tried to do something that worked in our favor with different data.

## **Homework 4: Linear methods for Classification:**

> **Instructions:** Pick one of the examples of the chapter that use the data of the book and replicate it in Python. Then, apply the steps in your own data. 

The example we are going to use is the South African heart disease. Same as other homeworks, we download our [dataset](https://github.com/mayraberrones94/Aprendizaje/blob/main/Datasets/sahd.csv) into our [notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/HW4_SAHD_example.ipynb) and see the description of our variables.

```python
import pandas as pd
import numpy as np

dataset = pd.read_csv('/content/drive/MyDrive/Datasets/sahd.csv')
dataset = dataset.drop('row.names', axis=1)
```

Here we can see some of the variables. 

|Name of variable | Description |
|-----------------|-------------|
|sbp              |Systolic blood presure|
|tobacco | Cumulative tobacco (kg) |
|ldl | Low density lipoprotein cholesterol|
|adiposity | |
|famhist | Family history of heart disease |
|typea | Type-A behavior |
|obesity | |
|alcohol | Current alcohol consumption |
|age | Age at onset |
|chd | Response coronary heart disease (1 is positive and 0 is negative)|

This information was gathered [here](https://www.kaggle.com/c/SAheart/data).



First we see in Figure 4.12 in the book that they ploted a similar pairplot than we did in our last homework. We import the library seaborn and plot a pairplot with the `chd` variable as target value. 

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/h4-scatter_sahd1.png)

As we can see the variable `famhist` is not present in this plot because is the only one that has text instead of numbers. We then convert the words inside. Present now is 1, and absent is 0.

```python
dataset['famhist'] = (dataset['famhist'] == 'Present').astype(int)
```

Now for the logistic regression, we do something similar to what we did in homework 3, where we use the library of `statsmodels` to train our model. 

```python
import statsmodels.api as sm
from scipy import stats
#https://www.statsmodels.org/dev/discretemod.html

target = 'chd'

features = ['sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age']
X, y = dataset[features].values, dataset[target].values
X_data = sm.add_constant(X)
lr = sm.Logit(y, X_data).fit(disp=False)

result = zip(['(Intercept)'] + features, lr.params, lr.bse, lr.tvalues)
print('               Coefficient   Std. Error   Z Score')
print('-------------------------------------------------')
for term, coefficient, std_err, z_score in result:
    print(f'{term:>12}{coefficient:>14.3f}{std_err:>13.3f}{z_score:>10.3f}')
```
As a result of this we have the following table:

|            |Coefficient  | Std. Error  | Z Score|
|------------|--------------|-------------|----------|
 |(Intercept)|        -4.130 |       0.964|    -4.283|
 |        sbp |        0.006 |       0.006 |    1.023|
|     tobacco |        0.080 |       0.026 |    3.034|
 |        ldl  |       0.185 |       0.057 |    3.218|
 |    famhist |        0.939 |       0.225 |    4.177|
|     obesity |       -0.035 |       0.029 |   -1.187|
|     alcohol |        0.001 |       0.004 |    0.136|
  |       age |        0.043 |       0.010 |    4.181|

If we follow the same steps we did in the example from chapter 3, we know that any Z score value that is greater than two is approximatly significant at 5%, so we can discard the values of `sbp`, `obesity`, and `alcohol`.

Now if we repeat the experiment with only the features that have significance, then we get Table 4.3 from the book. (For the complete code, refer to the notebook)

|              | Coefficient  | Std. Error  | Z Score|
|--------------|--------------|-------------|--------|
| (Intercept)  |      -4.204  |      0.498  |  -8.436|
 |    tobacco  |       0.081  |      0.026  |   3.163|
|         ldl  |       0.168   |     0.054  |   3.093|
 |    famhist  |       0.924   |     0.223  |   4.141|
 |        age  |       0.044  |      0.010  |   4.520|
 
 ### Logistic Regression with L1 Regularization
 
 We use now the library of `sklearn` to use regularization. Regularization is a technique used to prevent overfitting problem. If we are using the L1 regularization is called Lasso regression, and L2 is the Ridge regresion (both explored in previos work). Here we are going to use the L1 regularization. The same as before, we need to standardize our data:
 
 ```python
 import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create a scaler object
sc = StandardScaler()

# Fit the scaler to the training data and transform
X_train_std = sc.fit_transform(X_train)

# Apply the scaler to the test data
X_test_std = sc.transform(X_test)
```

And then we call for the logistic regression with the L1 penalty.

```python
C = [10, 1, .1, .001]

for c in C:
    clf = LogisticRegression(penalty='l1', C=c, solver='liblinear')
    clf.fit(X_train, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train_std, y_train))
    print('Test accuracy:', clf.score(X_test_std, y_test))
    print('')
```

Here are the results:

| Alpha | Co 1 | Co 2 | Co 3 | Co 4 | Co 5| Co 6 | Co 7 | Train acc | Test Acc |
|-------|------|-------|-----|------|------|------|------|-----------|----------|
| 10 |  1.23147379e-04 | 9.41743090e-02 | 1.59695109e-01 | 8.12604169e-01|-3.44760156e-02 |-9.37890006e-04 | 4.50527046e-02| 0.6531 |  0.6559|
| 1 | -3.80861093e-03|  9.55747083e-02 | 1.53761739e-01 | 7.41565324e-01 |-4.91484204e-02| -7.23104371e-04 | 4.41939756e-02| 0.6531| 0.6559 |
| 0.1 |-1.32202221e-02 | 9.58820956e-02 | 1.16129088e-01|  1.90052755e-01| -6.86888787e-02 | 8.82939644e-05|  4.47487507e-02| 0.6449 |  0.7311 |
|0.001 | -0.00359871 | 0.      |    0.    |      0.      |    0.       |   0. | 0.     |  0.4227 |  0.3225 |

Notice that as C decreases the model coefficients become smaller, until at C=0.001 almost all the coefficients are zero.

### Logistic regression dataset breast cancer:

For the next step we wanted to use the same data we used in the last homework, because it has the same structure as the one we saw in the example. (We where curious about this one from homework 3, when we saw the difference between linear and logistic regression) Next, we try to replicate it with our image dataset.

Similar to the last homework, we read and discard the columns in the dataset that we are not going to use. Then we use the same code used here to model a logistic regression. (Full code in the [notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/HW4_Breastcancer.ipynb))

```python
from sklearn.model_selection import train_test_split 
target = 'diagnosis'

features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
       'area_worst', 'smoothness_worst', 'compactness_worst',
       'concavity_worst', 'concave points_worst', 'symmetry_worst',
       'fractal_dimension_worst']


X, y = dataset[features].values, dataset[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
classifier7 = LogisticRegression()
classifier7.fit(X_train, y_train)
```

Now as a result we have the following metrics:

```python
import sklearn.metrics as metrics
print(metrics.classification_report(y_test, y_pred7))
```
| | precision |   recall|  f1-score |  support|
|---|---------|----------|----------|---------|
|         0.0  |     0.97   |   0.99  |    0.98 |       70|
 |        1.0     |  0.98   |   0.95 |     0.96  |      41|

 |   accuracy     |           |       |    0.97   |    114|
 |  macro avg     |  0.97    |  0.97    |  0.97    |   114|
|weighted avg     |  0.97  |    0.97    |  0.97    |   114|

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/h4_matrix1.png)

Following the first example, we take the penalties and search for the best parameter and best accuracy of the final model:

```python
from sklearn.model_selection import GridSearchCV
parameters_lr = [{'penalty':['l1','l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
grid_search_lr = GridSearchCV(estimator = classifier7,
                           param_grid = parameters_lr,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_lr.fit(X_train, y_train)
best_accuracy_lr = grid_search_lr.best_score_
best_paramaeter_lr = grid_search_lr.best_params_  
print("Best Accuracy of LR: {:.2f} %".format(best_accuracy_lr.mean()*100))
print("Best Parameter of LR:", best_paramaeter_lr)
```

The result of this is:
```
Best Accuracy of LR: 97.36 %
Best Parameter of LR: {'C': 0.1, 'penalty': 'l2'}
```
We have then as the best parameter of alpha 0.1, the best penalty as `l2` and the best accuracy of the model as 97.36.

### Using the Minimias dataset:

Now that we have this, we wanted to try the datasets of the images. At first we tried with the images themselves, but because of a minor inconvinience with my RAM memory, we repeated the process we made in homework 2, where we turned our images into text. Again, we use the dataset of minimias, because is the smallest one of all the others I have. For the entire code see [notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/HW4_Mias_2.ipynb).

Following the steps from before, we have our logistic regression model:

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix

print('Log loss = {:.5f}'.format(log_loss(y_test, test_prob)))
print('AUC = {:.5f}'.format(roc_auc_score(y_test, test_prob)))
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, test_prob)))
print('\nUsing 0.5 as threshold:')
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, test_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, test_pred)))
print('Recall = {:.5f}'.format(recall_score(y_test, test_pred)))
print('F1 score = {:.5f}'.format(f1_score(y_test, test_pred)))

print('\nClassification Report')
print(classification_report(y_test, test_pred))
```

The result was:

```
Log loss = 2.59377
AUC = 0.83667
Average Precision = 0.70296

Using 0.5 as threshold:
Accuracy = 0.74510
Precision = 0.60465
Recall = 0.74286
F1 score = 0.66667
```
Classification Report:

|             | precision  |  recall  |f1-score  | support|
|-------------|------------|-----------|----------|--------|
|           0  |     0.85   |   0.75   |   0.79  |      67|
 |          1  |     0.60    |  0.74   |   0.67   |     35|
|    accuracy  |              |        |   0.75     |  102|
 |  macro avg   |    0.73   |   0.74   |   0.73    |   102|
|weighted avg   |    0.76    |  0.75   |   0.75  |     102|

Next we print the plots that will show us the Roc curve:

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/h4roc1.png)

And the confusion matrix of our data:

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/h4matrix2.png)

Finally, we can see which are the best parameters and the best accuracy of our model:

```python
from sklearn.model_selection import GridSearchCV
parameters_lr = [{'penalty':['l1','l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
grid_search_lr = GridSearchCV(estimator = model,
                           param_grid = parameters_lr,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_lr.fit(X_train, y_train)
best_accuracy_lr = grid_search_lr.best_score_
best_paramaeter_lr = grid_search_lr.best_params_  
print("Best Accuracy of LR: {:.2f} %".format(best_accuracy_lr.mean()*100))
print("Best Parameter of LR:", best_paramaeter_lr)
```

And the final result is:

```
Best Accuracy of LR: 83.05 %
Best Parameter of LR: {'C': 1, 'penalty': 'l1'}
```

Where we can se that the best accuracy is 83.05, the best alpha is 1 and the best penalty the `l1`.

### Conclusions:

For this work, we got an accuracy that matches the best CNN models we have for this dataset, which in turn are far more complex. I had seen several times that logistic regression is very similar to the behaviour that neural networks have, so it was a pleasent suprise to see that it has good accuracy results.
 For google colab I think is a bit computational expensive to turn all the images of my various datasets into text, but if I translate to my computer and my console, it would be worth it to see what is the result of this same experiment with the bigger datasets, because the time it takes to train this model is significantly smaller than what it takes to train a CNN. 
 
The thing we want to try later on is to explore the use of tensorflow with this type of classification, or if there is a more efficient way to use logistic regression to images, since we saw that it could also be used for segmentation.

### Extra: Remove the random state from the model, and see how much it changes with each iteration.

For this experiment, we took `AUC`,	`Average_Precision`, `Accuracy`, `Precision`, `Recall`,	`F1_score`, and	`Best_accuracy` with the Plotly library.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/total_accs.png)

`Accuracy` and `best_accuracy` change very little in each iteration. `Average precision`, `Precision` and `Recall` are the ones that fluctuate the most. Next goes `F1_score`, but that is to be expected, when it uses the precision and recall to calculate it.

And finally, we have the percentage of times that `L1` or `L2` regularization was chosen for the best accuracy. Next to it, the most selected alpha value as well.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/pie_plot.png)

## **Homework 5: Basis expansion and regularization:**
> **Instructions:** Fit splines into single features in your project data. Explore options to obtain the best fit.

In this work, we wanted to explore the option of using Wavelets. In the book they mention the possible uses for wavelet smoothing for splines, and that they are very popular in signal processing and compression. 

The explanation for wavelets from the book was a bit dense, and we wanted to explore a bit more in the subject of image processing and compression, and if we could use it for any of the problems we described since homework 1.   For this, we compiled some information that helped us understand better the subject before we applied it to our models.

First, here are some handy concepts that helped:

> **Wavelet:** 
"A Wavelet is a wave-like oscillation that is localized in time, an example is given below. Wavelets have two basic properties: scale and location. Scale (or dilation) defines how “stretched” or “squished” a wavelet is. This property is related to frequency as defined for waves. Location defines where the wavelet is positioned in time (or space)."
...
"The basic idea is to compute how much of a wavelet is in a signal for a particular scale and location. For those familiar with convolutions, that is exactly what this is. A signal is convolved with a set of wavelets at a variety of scales."

This concept brought us back to the idea that we were up to something good with our analysis since we first understood splines and wavelets as signal processing over a period of time. Since our data are images, we could not relate the two different types of data, but at the explanation that it has a similar behavior as convolutions, we wanted to search for something more specific. So we landed on the next concept:

> **Wavelet Image Scattering:**
"In a digital image, texture provides information about the spatial arrangement of color or pixel intensities. Particular spatial arrangements of color or pixel intensities correspond to different appearances and consistencies of the physical material being imaged."
...
"Wavelet scattering works by cascading the image through a series of wavelet transforms, nonlinearities, and averaging"

This article in particular was really helpful since it gave us the idea we needed to fully understand how could we use the concept of wavelets for our classification problem.

What could be the advantages of using wavelets:

- Wavelet scattering is basically the equivalent of a deep convolutional network. Seeing some of the examples that we investigated, is very similar to the method of pooling since it takes some of the features of the images to take forward.
- It has proved to yield representations that are stable against deformations and robust to noise.

Now, since we have seen many comments and proofs of what the wavelets can do, we wonder what makes them different from neural networks, specifically, convolutional neural networks.

We are then introduced to the Wavelet Scattering Network (WSN). This model in itself can't do classification, but its outputs can be used as inputs to another type of classifier. 

In MATHLAB there is already software that takes advantage of these types of transformations. The wavelet is implemented alongside a deep convolutional network,

The original idea to solve our classification problem was to use different architectures to be able to find out what fitted best our target dataset. This however proved to be a major setback, since we found that the public datasets were performing well, and our target dataset was having issues going outside of the random accuracy (from 0.4 to 0.5). 

The best architecture generally for our experimentation with the other data sets (Mini-MIAS and DDSM) was the Alexnet model. Below we have some important parts of the adaptation we made for the Alexnet architecture. 

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib
```

The most important libraries for us are the TensorFlow and Keras libraries. All the other imports can be seen in the [full code](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/redv4_alex.py). (In this case, we did not use a notebook, because we use our computer console to run these codes)

```python
model = Sequential()
model.add(Conv2D(16, (11, 11), input_shape=(Lng, Hg, 3),
        padding='same', kernel_regularizer=l2(INIT_LR)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
```
Here is a small excerpt of one of the 7 layers of convolution and pooling of our model. 

```python
print("[INFO] Tr {} epochs...".format(EPOCHS))
H = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), callbacks=[reduce_lr], 
                                    validation_steps = 1000,
                                    steps_per_epoch=1000, 
                                    epochs=EPOCHS)
```
For this model we use a train data generator that makes some changes to the image (rotations, height and widht shifts, etc). First we experimented on the Minimias data set, since is the one we have been using for most of our work.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/mias-alex.png)

The final accuracy of our model was 0.69 for the training set, and 0.62 for the test set. And in the loss function we have 0.57 in training to 0.62 in testing.


Now, we look for a way to implement the wavelet scattering to our model. In this [link](https://github.com/menon92/WaveletCNN/blob/master/Wavelet%20Convolutional%20Neural%20Networks.pdf) we found the code for an article about 
Wavelet scattering, where they implement it on a dummy dataset about cats and dogs. They use a very complex CNN architecture, and since we are using the Alexnet for this comparision, we took only the wavelet transform functions, modifing the inputs and outputs to tone down the complexity to match an architecture like Alexnet.

They use a the pywavelet library from [python](https://pywavelets.readthedocs.io/en/latest/), and make it so it has several layers of image decomposition before we integrate it to the CNN.

```python
# batch operation usng tensor slice
def WaveletTransformAxisY(batch_img):
    odd_img  = batch_img[:,0::2]
    even_img = batch_img[:,1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H

def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:,:,::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:,::-1,...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:,::-1,...]
    return dst_L, dst_H
```

The rest of the code can be seen [here](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/wavelet.py). Then we modify the Alexnet architecture to be able to recieve as an input the images that went through wavelet transformations.

In the wavelet function it takes into consideration the chanel of color rgb, so we need to make modifications in case we are using images in black and white. Another important thing to have in mind is the size of the image, because in this function you also have to keep in mind that each decomposition requires a different image size (smaller with each iteration). It was recomended to use an initial image size that allowed to have this resizing without running out of pixels. In our case, since we only have two decompositions, we could still begin with a small size (to avoid very large computational load at the beginin of our training).

In the end, the code still uses almost the same libraries so it was easy to understand which parts we wanted to modify. The result in accuracy can be seen in the next plot:

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/mias-wave.png)

The final score was for the training set we had 0.84 of accuracy and 0.35 of loss, and for the test set we had 0.73 of accuracy and 0.53 of loss. It shows a good improvement from just using the CNN. We also noted that we had a very "spiky" plot, which made us think of the optimization problems we have seen before, where sometimes the result can get stuck in a local optimal solution, and that maybe, the wavelet transforms helped our optimizer to get out of those locals.

Since we saw good results and we were feeling optimistic, we wanted to try with our objective dataset. Like we mentioned, one of our biggest setbacks was that this dataset did not perform well at all in any of the architectures we used. We have confirmation that the dataset is separated correctly, so we where looking for options to see how could we make our model perform better.

 In this case, we followed the same process as before, and started with the Alexnet CNN architecture to see the results, which are not good at all. 
 
 ![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/incan-alex-plot.png)

As expected, the accuracy and loss of both datasets gets stuck in the random point, which means that it is a 50/50 chance it would classify correctly in real life.We have a score of 0.56 on accuracy and 0.72 on loss for the training set, and in the test set we have a 0.51 in accuracy and 0.87 in loss. Then, we train our model with our modified Wavelet Convolutional Network.

 ![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/incan-wavelet.png)
 
 | Score | First experiment | Second Experiment|
 |-------|------------------|------------------|
 |Train_acc | 0.91| 0.92 |
 |Train_loss| 0.22| 0.19 |
 |Test_acc| 0.69| 0.70 |
 |Test_loss| 0.80| 0.96 |
 
 
As we can see, we now have better results, and the training and testing sets are out of the random zone. We ran the experiment twice just to make sure it was just a random result. 

 In the end we have a better performance in the training set than we have in the test set. In this case we know that the reason for this could be the training and test spliting. We are going to work now on the features and variables that we know can help our model to see if we can balance them out.

### Conclusions:
This work was a bit hard to understand at first, because the information in the book about wavelets was a bit dense, but once we realized what we could do, and see the experimental results turn out better than we expected gave us a boost to continue to explore the wavelet transform, so we can keep improving our classification model. A major bonus was that we are now in a good path to understand why our target data set was not working to begin with, and we can now begin to form more hypothesis as to what features we can modify to improve our results.

## **Homework 6: Kernel smoothing methods:**
> **Instructions:** Build some local regression model for your data and adjust the parameters. Remember to read all of Chapter 6 first to get as many ideas as possible.

For this work, we had very different ideas. Since we work with convolutional networks, kernels are something that is mentioned constantly. In our experience, kernels are used as a part of the convolutional process to extract features from the input images. The size of the kernel is one of the parameters we always left fixed in a matrix of 3x3 since our understanding was that this matrix was used to extract little chunks of the image to analyze (and we wanted a small computational load for our models).

In this week's work, we would like to explore the different uses we can give kernels, starting with the ones we know, and then getting into the ones they mention in the book, such as kernel density estimation.

As always, we begin by importing some of the main libraries we are going to be using:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns; sns.set()
```

The first use we found of kernels was to enhance our dataset and augment data. Before we had knowledge of a more simple and effective way to augment our data (without compromising quality or storage) we practiced with some kernels that are widely used for image transformations. For this experiment, we are using one of the images from our free dataset.

```python
image = cv2.imread('mdb001.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(image)
plt.savefig('breast1.png')
```
![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/breast1.png)

Here we see the original image. For some transformations, we can construct and develop our own kernels, which helps us determine what kind of kernel could be more beneficial for our images.

```python
kernel_sharpen = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")


img_sharp = cv2.filter2D(image, -1, kernel_sharpen)
img_laplace = cv2.filter2D(image, -1, laplacian)
img_sox = cv2.filter2D(image, -1, sobelX)
img_soy = cv2.filter2D(image, -1, sobelY)

```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/img_sharp.png)

For the sharpening transformation, there is a slight chance that we can appreciate it in the upper part of the image. If we compare it with the original, we can actually see some of the lines (the veins) are a little more pronounced.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/img-laplace.png)

For the Laplace transormation, we can not get much out of it, but that is how it is supposed to work for this type of image.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/img-sox.png)

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/img-soy.png)

Finally, the Sobel transformation we have something similar to what we discovered in the wavelet transformation python library from homework 5. They take into consideration different sides of the image, so they look like two different shadings.

There are other Gaussian and median blur kernels that we can see in the full [notebook](https://github.com/mayraberrones94/Aprendizaje/blob/main/Notebooks/HW6_KDE.ipynb). A very interesting one that we found in the library of OpenCV that we had not seen before is the dilation and erosion function. 

```python
image = cv2.imread('mdb001.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
r ,image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
# create kernel
kernel = np.ones((5,5), np.uint8)
fig, ax = plt.subplots(1, figsize=(16,12))
# original
ax = plt.subplot(232)
plt.imshow(image)
plt.title('original')
# erosion
e = cv2.erode(image, kernel)
ax = plt.subplot(234)
plt.imshow(e)
plt.title('erosion')
# dilation
d = cv2.dilate(image, kernel)
ax = plt.subplot(235)
plt.imshow(d)
plt.title('dilation')
# morphological gradient (dilation - erosion)
m = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
ax = plt.subplot(236)
plt.imshow(m)
plt.title('dilation - erosion')\

plt.savefig('dilation-erosion.png')
```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/dilation-erosion.png)

Both of the filters combined can help us when we reach the step of feature extraction, and we have to make our own ground truth data.

In the notebook, we also mention the library PIL, which was the one we used to augment our data in our master's thesis experimentation. We had to make some changes to the augmentation process, because we had some problems with some images since it was not consistent when we used it in all the datasets.

Now we wanted to explore a bit more of the experiments and explanations we saw in the book. Again, we had to think a little bit differently, since the examples shown in the book use a different type of dataset than the one we have. We took one image and transform it into a histogram. 

```python
from skimage import io
import matplotlib.pyplot as plt

image = io.imread('mdb001.png')
ax = plt.hist(image.ravel(), bins = 256)
plt.savefig('breast-ravel.png')
plt.show()

```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/breast-ravel.png)

For this first plot, we used a full `binsize` of 256. The bars are too close together, so we plotted some more histograms with decreasing `binsize`.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/breast-bins150.png)

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/breast-bins75.png)

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/breast-bins30.png)

As we can see, all of them have similar behavior, but the one with 75 and 30 bin sizes, we can recognize the pattern much better.

Reading further into this behavior, we looked into other python libraries. In this case, we found the sklearn library and seaborn. With seaborn, we could actually plot a KDE with the default bandwidth for the gaussian distribution.

Then, same as we did before, we compare the smoothness of the KDE with different bandwidth estimates. 

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/kdeplot-bandwith-estim.png)


Further than this, we wanted to use a similar dataset from the book, so we took the data from the breast cancer Wisconsin that we have been using in previous homework. 

Seaborn also has a function where we can use the standard Gaussian function, or alter the bandwidth of the plot. We took the mean features of our dataset and compared the bandwidth for normal Gaussian distribution, 0.05 and 0.1 bandwidth.


![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/bcw-normal-kde.png)

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/bcw-kde-0.05.png)

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/bcw-kde-0.1.png)

Although is fun to change the bandwidth to see the different aspects of our data, sklearn actually has some functions that would help us determine the best bandwidth for our data. 

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

model = KernelDensity()
model.fit(x_train)
log_dens = model.score_samples(x_test)

bandwidth = np.arange(0.05, 2, .05)
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(x_train)

kde = grid.best_estimator_
log_dens = kde.score_samples(x_test)
plt.plot(x_test)
plt.title('Optimal estimate with Gaussian kernel')
plt.savefig('optimal-band1.png')
plt.show()
print("optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))
```

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/optimal-band1.png)

In the end, we have this plot and have that the best bandwidth for this data is `1.95`.

### Conclusions:

For this work, we wanted to focus only on the images, and see what the kernel density estimation could bring us, especially since we saw this post about the MNIST data set, and how they generated their own images from the original data. This idea was especially appealing to us since we have a limited amount of data from our target dataset, but after several tries, we realized that the computational load was going to be too much. The experiments we saw also yielded very poor images as a result. We also tried to apply the best bandwidth estimator for the small image experiment we first have, but it took way too much time to compile, so we moved on to the next one.

In some articles, we also saw how kernel density estimation works poorly with images (https://arxiv.org/pdf/2110.12644.pdf).

In the end, it was very interesting to see what kernels actually do outside of convolutions, where we just took them as another parameter. 

## **Homework 7: Model assesment and selection**
> **Instructions:**  Apply both cross-validation and bootstrap to your project data to study how variable your results are when you switch the test set around.

Since our last work from homework 5, we have been trying to find something that would help us improve the results shown in the accuracy and loss functions of our model. 

We had a major discovery when we could finally take our target dataset from the randomness it seemed to be stuck in. The only problem with those results was that the difference between the training and test set was showing signs of overfitting.

> **Overfitting** This occurs when we have a model that fits exactly against the training data points, which would make the training accuracy appear near perfect, but when we test the model, we can see that it is not performing as well as the training.

What are some signals we need to be looking after when we want to avoid overfitting:

- Low error rates and high variance are good indicators of overfitting. 
- If the training data has a low error rate and the test data has a high error rate, it signals to overfit.
- When the model trains for too long on sample data or when the model is too complex, it can start to learn the “noise,” or irrelevant information.
- When the model memorizes the noise and fits too closely to the training set, the model becomes “overfitted,” and it is unable to generalize well to new data. 

There is a variety of things we can try to avoid falling into an overfitted model. One of them is the use of Dropout.

> **Dropout** Dropout is referring to the data or noise that is intentionally dropped from a neural network to improve the processing and time of the results.

This dropout parameter is already considered inside all of the CNN architectures that we have used so far. Our favorite way to explain it is like a traffic quota that needs to be filled between neuron connections. If there is not enough traffic from neuron to neuron 
(depending on the threshold we assign), then that connection gets deprecated, so in the end, only the most-used connections are left.

Dropout proved to be a good parameter to avoid overfitting even when our architecture was in its simpler form. So now that we know that we are using this parameter, we can start looking for others that can help us improve the accuracy of our model and avoid overfitting.




## Cross validation

We have established now some of the characteristics of an overfitted model. We are already using dropout to help alleviate some of these results. So now we move to cross-validation.

> **Cross-validation:** This is a very important concept in machine learning. It can help our model by reducing the size of the data and ensuring our model is robust.

Most of the time, the way we solved the overfitting problem on datasets was to change the size of the training and test sets, but this also can turn out to be problematic if we do not have enough data in our training set that helps our model generalize the image features correctly.

As we mentioned, one of the advantages of using cross-validation is that helps us reduce this problem of imbalanced data.

Normally in our training process, we will have three sets. Training, validation, and test sets. If we do not have enough data (as seen in the YERAL and MINI MIAS datasets) these kinds of divisions can be contra-productive. 

Introducing cross-validation to our process can help reduce the need to have a validation set because it is already embedded in the train and test sets.

An example of how does this works can be seen in the example below, where we have an example of a 5 fold cross-validation data split.


Seeing some of the libraries that use this type of cross-validation we find that the `Sklearn` library uses a type of stratified split when we use the `train_test_split` method. This means that our data is well distributed to our target variables (normal and anomalies).

Now, as we mentioned, `Sklearn` already helps with the stratified separation of the datasets, but there are also other methods for splitting data that allow us to solve different types of problems.

Some of the most common and popular methods are:

- `train_test_split` - creates single split into train and test set.
- `Kfold` - creates k-fold splits allowing cross validation
- `StratifiedKFold` - creates k-fold splits considering the distribution of the target variable
- `cross_val_score` - evaluates model's score through cross validation

We begin by importing the necessary libraries:

```python
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# import libraries for charting and manipulations with datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
```
The library of Sklearn has a little experiment to represent what happens to our dataset when we use the cross-validation folds. In the first plot, we can see a kind of ordered way of taking data.

![alt text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/kfolds.png)

We made some modifications to our original code for wavelet transformations, so we can apply these methods of sampling and see if we can improve the accuracy and loss we presented in homework 5. In this case we wanted to focus only on the target data set of YERAL. Depending on our results, we are going to continue the experimentation with the other datasets.

```python
def get_model_name(k):
    return 'model_'+str(k)+'.h5'

kf = KFold(n_splits = 5)
                         
skf = StratifiedKFold(n_splits = 3, random_state = 7, shuffle = True) 

VALIDATION_ACCURACY = []

VALIDAITON_LOSS = []
from numpy import array
#save_dir = '/Users/MayraBerrones/Documents/VisualCode/'
fold_var = 1
X = np.asarray(X)
y = np.asarray(y)

```


The first thing we establish is the two imports we are going to use `KFold` and `StratifiedKFold`.

```python
for train, test in kf.split(X):
    trainX, testX, trainY, testY = X[train], X[test], y[train], y[test]
	
	# CREATE NEW MODEL
    model = get_wavelet_cnn_model()
	# COMPILE NEW MODEL
    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=["acc"])

	
	# CREATE CALLBACKS
    checkpoint = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    callbacks_list = [checkpoint]
	# There can be other callbacks, but just showing one because it involves the model name
	# This saves the best model
	# FIT THE MODEL
    history = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), 
                                    callbacks=callbacks_list, 
                                    validation_steps = 100,
                                    steps_per_epoch=100, 
                                    epochs=EPOCHS)
```


For the `KFold` results, we struggled a little bit to adapt it to the code, because the examples we found were usually for low dimension data and other algorithms such as RandomForest and SVM. In the end, the model takes a bit more time to compile, and each iteration in the training model goes up in time elapsed. 

In this case, we did not let the model finish the three intended folds, because it was taking too long, and the results for the first and second fold where not good at all:

| Fold | Train acc | Train loss | Test acc | Test loss |
|------|-----------|------------|----------|-----------|
| 1 | 0.90 | 0.21 | 0.20 | 2.78|
|2 | 0.85 | 0.31 | 0.14 | 5.19 |

As we can see in the results, `KFolds` does not improve our training or loss results. Since the results took too much time and the results were not very positive, we wanted to move on to the bootstrap method. But first, we tried the stratified k folds, since in the beginning, we mentioned that the `train_test_split` function already helped us with that.

When we ran the dummy model of the `sklearn` library, we ended up with a division like so:


![alt_text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/shuffleplot.png)

As we can see, the splt is different than we only used KFolds.

These are the results of the 3 folds of `StratifiedKFold`:

![alt_text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/kfolds_plot.png)


| Fold | Train acc | Train loss | Test acc | Test loss |
|------|-----------|------------|----------|-----------|
| 1 | 0.85 | 0.31 | 0.58 | 1.60|
|2 | 0.87 | 0.29 | 0.70 | 0.97 |
|3 | 0.84 | 0.35 | 0.68 | 0.74 |

Here are the changes we had to make to the code in order to run it:

```python
for train, test in skf.split(X, np.zeros(shape=(X.shape[0], 1))):
    trainX, testX, trainY, testY = X[train], X[test], y[train], y[test]
	
```

Our code also gives us the presision and recall factors, and in all the folds, the dataset containg anomalies had the highest percentage. In here we do not really see a big improvement in the loss function of the test set, which was the problem that we wanted to solve from homework 5. So now we try the bootstrap.

## Bootstrap.

When we searched in the `sklearn` library for a method that contained bootstrap, we found out that it was discontinued, so we learned the process that takes place in the methodology of "bootstraping" and searched for the function closest to it.

> **Bootstrap:** Bootstrap Sampling is a method that involves drawing of sample data repeatedly with replacement from a data source to estimate a population parameter.

The `resample()` scikit-learn function can be used for this. It takes as arguments the data array, whether or not to sample with replacement, the size of the sample, and the seed.

So now we have the wavelet-CNN with the following code (the rest of the code can be found [here]()):

```python
B = 5
errors = []

VALIDATION_ACCURACY = []

VALIDAITON_LOSS = []
for i in range(B):
    X_bootstrap, y_bootstrap = resample(X, y)
    trainX, testX, trainY, testY = train_test_split(X_bootstrap, y_bootstrap, test_size=0.25, random_state=42)
    model = get_wavelet_cnn_model()
    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=["acc"])
    checkpoint = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    callbacks_list = [checkpoint]

    history = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), 
                                    callbacks=callbacks_list, 
                                    validation_steps = 1000,
                                    steps_per_epoch=1000, 
                                    epochs=EPOCHS)

    predictions = model.predict(testX, batch_size=BS)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
    model.save("modelboot_"+str(i)+".h5")
    model.load_weights("modelboot_"+str(i)+".h5")
```
 
 This are the results of using the bootstrap method:
 
 ![alt_text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/boot-allplot.png)
 
 | Iteration | Train acc | Train loss | Test acc | Test loss |
|------|-----------|------------|----------|-----------|
| 1 | 0.89 | 0.25 | 0.79 | 0.79|
|2 |0.86 | 0.33 | 0.67 | 0.81|
|3 | 0.87 | 0.28 | 0.76 | 0.59|
|4 | 0.87 | 0.30 | 0.78 | 0.61|
|5 | 0.86 | 0.31 | 0.64 | 0.81|

### Conclusions:

In general, the bootstrap method worked faster than the stratified Kfold, and the results are also better. In three of the five iterations, we saw that the loss function of both datasets was not so far apart, which is also a really good result.

Another thing we noticed when comparing the stratified Kfolds and the bootstrap results was that in the precision and recall the results were not so far apart from normal and abnormal images in the bootstrap method, which makes it the more stable option.

## **Homework 8: Model inference and averaging** 

> **Instructions:**  Do EM (expectation-maximization) with your data following the from-scratch steps of Causevic.


For this homework, like homework 7, we wanted to explore the concepts of the work a little bit more outside of the book. Here are some concepts that helped us understand the assignment better:

> What is the EM algorithm: This algorithm is called EM for expectation-maximization, and in this case, those are the primary steps that the algorithm is known for. The E-step uses the dataset to try to estimate or guess the values of the missing data.  Then on the M step, we use the complete data and update the parameters. These two steps repeat until we converge in a solution.

The main advantage of this algorithm is that simple to implement, and it is always guaranteed that the value of the likelihood will increase after each iteration. 

As for disadvantages we have that the time of convergence can be very slow and it only converges to local minima. 

Gaussian mixture models are an approach to density estimation where the parameters of the distributions are fit using the expectation-maximization algorithm. In this case, density estimation involves the selection of a probability distribution function to try to explain the joint probability of the data. 

For EM applications, clustering with a mixture model is one of the most popular ones. The Gaussian Mixture Model, or GMM for short, is a mixture model that uses a combination of Gaussian (Normal) probability distributions and requires the estimation of the mean and standard deviation parameters for each.

One of the applications that we saw for this algorithm that caught our attention was the use of the EM algorithm for image segmentation. Image segmentation is an image processing procedure to label pixels of similar kinds into the same cluster groups. The most often compared methodology was the k-nearest neighbors, so we do the same here.

First, we load up the libraries. In this work, we are going to be using mostly `sklearn`, `scipy`, and `numpy` to build the EM algorithm. This code was adapted from a [project](https://github.com/tmclouisluk/Expectation-maximization-Algorithm-on-Image-Segmentation/blob/master/EM.py) where we changed it to be able to use our images and use the libraries updated functions mentioned below, the full code can be seen [here]().

```python
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
from numpy.random import randint, random
import scipy.stats
import math
import cv2
import imageio

from sklearn import cluster
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from scipy import ndimage
%matplotlib inline
```

Now we need a function to receive the input images. In this function, we can see we have a downsample of the image, as well as a blurring transform that we are going to denoise.

```python
def read_img(filename, mode, size):
    if mode == 'RGB':
        img_3d = imageio.imread(filename, pilmode = 'RGB')
    elif mode == 'L':
        img_3d = imageio.imread(filename, pilmode = 'L')
    # Downsample the image
    small = cv2.resize(img_3d, (0, 0), fx = size[0], fy = size[1])
    # Blurring effect to denoise
    blur = cv2.blur(small, (4, 4))
    return blur

```

We first have the parameter for the E-step, using the multivariate_normal function of `scipy` to update the conditional probability of pixel `i` given class` j`. Here one of the main thigs we changed was to allow for singular matrix, since the example we had was for a different type of image, and this was a constant error that was raised in the code.

```python
def update_responsibility(img, means, cov, pis, k):
    responsibilities = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(img, allow_singular=True, mean=means[j], cov=cov[j]) for j in range(k)]).T
    norm = np.sum(responsibilities, axis = 1)
    norm = np.reshape(norm, (len(norm), 1))
    responsibilities = responsibilities / norm
    return responsibilities
```

Another change we had to make, was to establish how many iterations we were going to allow the model, since our first tryout, the model got stuck in the first image having over 400 iterations.

Before modeling, we take a look at how our images are looking.

```python
# Visualize demo images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,15))
ax_list = [ax1, ax2, ax3]
rgb_img_list = []
dim_img_list = []
i = 1
for filename, ax in zip(FILENAME_LIST, ax_list):
    rgb_img = read_img(filename = filename, mode = 'RGB', size = (0.5, 0.5))
    x, y, z = rgb_img.shape
    # Store dimension for each image
    dim_img_list.append((x,y,z))
    # Store img 
    rgb_img_list.append(rgb_img)
    ax.imshow(rgb_img)
    ax.set_title('[Image {}] Original Image'.format(i))
    ax.axes.set_xlabel('x-coordinate')
    ax.axes.set_ylabel('y-coordinate')
    i += 1

plt.tight_layout()
fig.savefig('orig_images.pdf')
```

![alt_text](https://github.com/mayraberrones94/Aprendizaje/blob/main/Images/img_original.png)

