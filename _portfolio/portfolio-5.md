---
title: "ICR - Identifying Age-Related Conditions"
excerpt: "The *ICR - Identifying Age-Related Conditions* project aims to develop a model to predict whether an individual has one of three specific medical conditions (Class 1) or none (Class 0). Using the *ICR* dataset with 56 anonymized health features, a Random Forest model in TensorFlow Decision Forests is employed to achieve accurate binary classification of age-related conditions."
collection: portfolio
---




The objective of this notebook is to create a model to predict whether a person has none (Class 0) or any of three medical conditions (Class 1).
We are going to try to solve the problem by building a Random Forest model using TensorFlow Decision Forests on our dataset ICR - Identifying Age-Related Conditions, which is described as follows:

## Dataset Description

- `train.csv`: The training set.
  - `Id`: Unique identifier for each observation.
  - `AB-GL`: Fifty-six anonymized health characteristics. All are numeric except for `EJ`, which is categorical.
  - `Class`: A binary target: 1 indicates the subject has been diagnosed with one of the three conditions, 0 indicates they have not.

- `test.csv`: The test set. Your goal is to predict the probability that a subject in this set belongs to each of the two classes.

- `greeks.csv`: Supplemental metadata, only available for the training set.
  - `Alpha`: Identifies the type of age-related condition, if present.
  - `A`: No age-related condition. Corresponds to class 0.
  - `B`, `D`, `G`: The three age-related conditions. Correspond to class 1.
  - `Beta`, `Gamma`, `Delta`: Three experimental characteristics.
  - `Epsilon`: The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected.


## Import the libraries


```python
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
```

    /opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/__init__.py:98: UserWarning: unable to load libtensorflow_io_plugins.so: unable to open file: libtensorflow_io_plugins.so, from paths: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so']
    caused by: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so: undefined symbol: _ZN3tsl6StatusC1EN10tensorflow5error4CodeESt17basic_string_viewIcSt11char_traitsIcEENS_14SourceLocationE']
      warnings.warn(f"unable to load libtensorflow_io_plugins.so: {e}")
    /opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/__init__.py:104: UserWarning: file system plugins are not loaded: unable to open file: libtensorflow_io.so, from paths: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZTVN10tensorflow13GcsFileSystemE']
      warnings.warn(f"file system plugins are not loaded: {e}")



```python
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)
```

    TensorFlow v2.12.0
    TensorFlow Decision Forests v1.3.0


## Load the Dataset


```python
train_data = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
train_data.shape
```




    (617, 58)




```python
train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>AB</th>
      <th>AF</th>
      <th>AH</th>
      <th>AM</th>
      <th>AR</th>
      <th>AX</th>
      <th>AY</th>
      <th>AZ</th>
      <th>BC</th>
      <th>...</th>
      <th>FL</th>
      <th>FR</th>
      <th>FS</th>
      <th>GB</th>
      <th>GE</th>
      <th>GF</th>
      <th>GH</th>
      <th>GI</th>
      <th>GL</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000ff2bfdfe9</td>
      <td>0.209377</td>
      <td>3109.03329</td>
      <td>85.200147</td>
      <td>22.394407</td>
      <td>8.138688</td>
      <td>0.699861</td>
      <td>0.025578</td>
      <td>9.812214</td>
      <td>5.555634</td>
      <td>...</td>
      <td>7.298162</td>
      <td>1.73855</td>
      <td>0.094822</td>
      <td>11.339138</td>
      <td>72.611063</td>
      <td>2003.810319</td>
      <td>22.136229</td>
      <td>69.834944</td>
      <td>0.120343</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>007255e47698</td>
      <td>0.145282</td>
      <td>978.76416</td>
      <td>85.200147</td>
      <td>36.968889</td>
      <td>8.138688</td>
      <td>3.632190</td>
      <td>0.025578</td>
      <td>13.517790</td>
      <td>1.229900</td>
      <td>...</td>
      <td>0.173229</td>
      <td>0.49706</td>
      <td>0.568932</td>
      <td>9.292698</td>
      <td>72.611063</td>
      <td>27981.562750</td>
      <td>29.135430</td>
      <td>32.131996</td>
      <td>21.978000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>013f2bd269f5</td>
      <td>0.470030</td>
      <td>2635.10654</td>
      <td>85.200147</td>
      <td>32.360553</td>
      <td>8.138688</td>
      <td>6.732840</td>
      <td>0.025578</td>
      <td>12.824570</td>
      <td>1.229900</td>
      <td>...</td>
      <td>7.709560</td>
      <td>0.97556</td>
      <td>1.198821</td>
      <td>37.077772</td>
      <td>88.609437</td>
      <td>13676.957810</td>
      <td>28.022851</td>
      <td>35.192676</td>
      <td>0.196941</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>043ac50845d5</td>
      <td>0.252107</td>
      <td>3819.65177</td>
      <td>120.201618</td>
      <td>77.112203</td>
      <td>8.138688</td>
      <td>3.685344</td>
      <td>0.025578</td>
      <td>11.053708</td>
      <td>1.229900</td>
      <td>...</td>
      <td>6.122162</td>
      <td>0.49706</td>
      <td>0.284466</td>
      <td>18.529584</td>
      <td>82.416803</td>
      <td>2094.262452</td>
      <td>39.948656</td>
      <td>90.493248</td>
      <td>0.155829</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>044fb8a146ec</td>
      <td>0.380297</td>
      <td>3733.04844</td>
      <td>85.200147</td>
      <td>14.103738</td>
      <td>8.138688</td>
      <td>3.942255</td>
      <td>0.054810</td>
      <td>3.396778</td>
      <td>102.151980</td>
      <td>...</td>
      <td>8.153058</td>
      <td>48.50134</td>
      <td>0.121914</td>
      <td>16.408728</td>
      <td>146.109943</td>
      <td>8524.370502</td>
      <td>45.381316</td>
      <td>36.262628</td>
      <td>0.096614</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>612</th>
      <td>fd3dafe738fd</td>
      <td>0.149555</td>
      <td>3130.05946</td>
      <td>123.763599</td>
      <td>9.513984</td>
      <td>13.020852</td>
      <td>3.499305</td>
      <td>0.077343</td>
      <td>8.545512</td>
      <td>2.804172</td>
      <td>...</td>
      <td>0.173229</td>
      <td>1.26092</td>
      <td>0.067730</td>
      <td>8.967128</td>
      <td>217.148554</td>
      <td>8095.932828</td>
      <td>24.640462</td>
      <td>69.191944</td>
      <td>21.978000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>613</th>
      <td>fd895603f071</td>
      <td>0.435846</td>
      <td>5462.03438</td>
      <td>85.200147</td>
      <td>46.551007</td>
      <td>15.973224</td>
      <td>5.979825</td>
      <td>0.025882</td>
      <td>12.622906</td>
      <td>3.777550</td>
      <td>...</td>
      <td>10.223150</td>
      <td>1.24236</td>
      <td>0.426699</td>
      <td>35.896418</td>
      <td>496.994214</td>
      <td>3085.308063</td>
      <td>29.648928</td>
      <td>124.808872</td>
      <td>0.145340</td>
      <td>0</td>
    </tr>
    <tr>
      <th>614</th>
      <td>fd8ef6377f76</td>
      <td>0.427300</td>
      <td>2459.10720</td>
      <td>130.138587</td>
      <td>55.355778</td>
      <td>10.005552</td>
      <td>8.070549</td>
      <td>0.025578</td>
      <td>15.408390</td>
      <td>1.229900</td>
      <td>...</td>
      <td>0.173229</td>
      <td>0.49706</td>
      <td>0.067730</td>
      <td>19.962092</td>
      <td>128.896894</td>
      <td>6474.652866</td>
      <td>26.166072</td>
      <td>119.559420</td>
      <td>21.978000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>615</th>
      <td>fe1942975e40</td>
      <td>0.363205</td>
      <td>1263.53524</td>
      <td>85.200147</td>
      <td>23.685856</td>
      <td>8.138688</td>
      <td>7.981959</td>
      <td>0.025578</td>
      <td>7.524588</td>
      <td>1.229900</td>
      <td>...</td>
      <td>9.256996</td>
      <td>0.78764</td>
      <td>0.670527</td>
      <td>24.594488</td>
      <td>72.611063</td>
      <td>1965.343176</td>
      <td>25.116750</td>
      <td>37.155112</td>
      <td>0.184622</td>
      <td>0</td>
    </tr>
    <tr>
      <th>616</th>
      <td>ffcca4ded3bb</td>
      <td>0.482849</td>
      <td>2672.53426</td>
      <td>546.663930</td>
      <td>112.006102</td>
      <td>8.138688</td>
      <td>3.198099</td>
      <td>0.116928</td>
      <td>3.396778</td>
      <td>7.948668</td>
      <td>...</td>
      <td>0.173229</td>
      <td>1.14492</td>
      <td>0.149006</td>
      <td>13.673940</td>
      <td>72.611063</td>
      <td>6850.484442</td>
      <td>45.745974</td>
      <td>114.842372</td>
      <td>21.978000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>617 rows × 58 columns</p>
</div>



The data is composed of 58 columns and 617 entries. `Class` is the label column indicating if a person has one or more of any of the three medical conditions (i.e,`Class 1`), or none of the three medical conditions (i.e,`Class 0`). Given the features of the dataset, the goal of our model is to predict the value of Class for any person.

## Dataset exploration


```python
train_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AB</th>
      <th>AF</th>
      <th>AH</th>
      <th>AM</th>
      <th>AR</th>
      <th>AX</th>
      <th>AY</th>
      <th>AZ</th>
      <th>BC</th>
      <th>BD</th>
      <th>...</th>
      <th>FL</th>
      <th>FR</th>
      <th>FS</th>
      <th>GB</th>
      <th>GE</th>
      <th>GF</th>
      <th>GH</th>
      <th>GI</th>
      <th>GL</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>...</td>
      <td>616.000000</td>
      <td>617.000000</td>
      <td>615.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>617.000000</td>
      <td>616.000000</td>
      <td>617.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.477149</td>
      <td>3502.013221</td>
      <td>118.624513</td>
      <td>38.968552</td>
      <td>10.128242</td>
      <td>5.545576</td>
      <td>0.060320</td>
      <td>10.566447</td>
      <td>8.053012</td>
      <td>5350.388655</td>
      <td>...</td>
      <td>5.433199</td>
      <td>3.533905</td>
      <td>0.421501</td>
      <td>20.724856</td>
      <td>131.714987</td>
      <td>14679.595398</td>
      <td>31.489716</td>
      <td>50.584437</td>
      <td>8.530961</td>
      <td>0.175041</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.468388</td>
      <td>2300.322717</td>
      <td>127.838950</td>
      <td>69.728226</td>
      <td>10.518877</td>
      <td>2.551696</td>
      <td>0.416817</td>
      <td>4.350645</td>
      <td>65.166943</td>
      <td>3021.326641</td>
      <td>...</td>
      <td>11.496257</td>
      <td>50.181948</td>
      <td>1.305365</td>
      <td>9.991907</td>
      <td>144.181524</td>
      <td>19352.959387</td>
      <td>9.864239</td>
      <td>36.266251</td>
      <td>10.327010</td>
      <td>0.380310</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.081187</td>
      <td>192.593280</td>
      <td>85.200147</td>
      <td>3.177522</td>
      <td>8.138688</td>
      <td>0.699861</td>
      <td>0.025578</td>
      <td>3.396778</td>
      <td>1.229900</td>
      <td>1693.624320</td>
      <td>...</td>
      <td>0.173229</td>
      <td>0.497060</td>
      <td>0.067730</td>
      <td>4.102182</td>
      <td>72.611063</td>
      <td>13.038894</td>
      <td>9.432735</td>
      <td>0.897628</td>
      <td>0.001129</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.252107</td>
      <td>2197.345480</td>
      <td>85.200147</td>
      <td>12.270314</td>
      <td>8.138688</td>
      <td>4.128294</td>
      <td>0.025578</td>
      <td>8.129580</td>
      <td>1.229900</td>
      <td>4155.702870</td>
      <td>...</td>
      <td>0.173229</td>
      <td>0.497060</td>
      <td>0.067730</td>
      <td>14.036718</td>
      <td>72.611063</td>
      <td>2798.992584</td>
      <td>25.034888</td>
      <td>23.011684</td>
      <td>0.124392</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.354659</td>
      <td>3120.318960</td>
      <td>85.200147</td>
      <td>20.533110</td>
      <td>8.138688</td>
      <td>5.031912</td>
      <td>0.025578</td>
      <td>10.461320</td>
      <td>1.229900</td>
      <td>4997.960730</td>
      <td>...</td>
      <td>3.028141</td>
      <td>1.131000</td>
      <td>0.250601</td>
      <td>18.771436</td>
      <td>72.611063</td>
      <td>7838.273610</td>
      <td>30.608946</td>
      <td>41.007968</td>
      <td>0.337827</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.559763</td>
      <td>4361.637390</td>
      <td>113.739540</td>
      <td>39.139886</td>
      <td>8.138688</td>
      <td>6.431634</td>
      <td>0.036845</td>
      <td>12.969516</td>
      <td>5.081244</td>
      <td>6035.885700</td>
      <td>...</td>
      <td>6.238814</td>
      <td>1.512060</td>
      <td>0.535067</td>
      <td>25.608406</td>
      <td>127.591671</td>
      <td>19035.709240</td>
      <td>36.863947</td>
      <td>67.931664</td>
      <td>21.978000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.161666</td>
      <td>28688.187660</td>
      <td>1910.123198</td>
      <td>630.518230</td>
      <td>178.943634</td>
      <td>38.270880</td>
      <td>10.315851</td>
      <td>38.971568</td>
      <td>1463.693448</td>
      <td>53060.599240</td>
      <td>...</td>
      <td>137.932739</td>
      <td>1244.227020</td>
      <td>31.365763</td>
      <td>135.781294</td>
      <td>1497.351958</td>
      <td>143790.071200</td>
      <td>81.210825</td>
      <td>191.194764</td>
      <td>21.978000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 56 columns</p>
</div>



### Checking the balance in the classes.
We count the occurrences of each class in our dataset.


```python
train_data['Class'].value_counts().plot(kind='pie', labels = ['neg', 'pos'], colors=['green', 'red'], autopct='%1.1f%%')
```




    <Axes: ylabel='Class'>




    
![png](../../images/ICR-identifying-age-related-conditions_files/ICR-identifying-age-related-conditions_11_1.png)
    



```python
# Calculate the number of samples for each label.
neg, pos = np.bincount(train_data['Class']) 

#Total samples 
total = neg + pos

#Percent of each class
per_neg = round((neg/total)*100,1)
per_pos = round((pos/total)*100,1)

print(f'Class 0: {per_neg} %')
print(f'Class 1: {per_pos} %')
```

    Class 0: 82.5 %
    Class 1: 17.5 %


**Important**: Based on the pie chart and the percentage distribution of each class in the dataset, it is evident that the dataset suffers from a significant class imbalance. The fraction of positive (`1`) samples is considerably smaller compared to the negative (`0`) samples.

## Analyzing the numerical data.
First, we will list all the numerical columns names.


```python
# Store all the numerical column names into a list
NUM_FEATURE_COLUMNS =  [i for i in train_data.columns if i not in ["Id", "EJ", "Class"]]
```

Let us now plot the first 6 numerical columns and their values using bar charts.


```python
figure, axis = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.25, wspace=0.3)

for i, column_name in enumerate(NUM_FEATURE_COLUMNS[:6]):
    row = i//2
    col = i % 2
    bp = sns.barplot(ax=axis[row, col], x=train_data['Id'], y=train_data[column_name])
    bp.set(xticklabels=[])
    bp.set_xticklabels(bp.get_xticklabels(), rotation=90, size = 7)
    axis[row, col].set_title(column_name)
plt.show()
```


    
![png](../../images/ICR-identifying-age-related-conditions_files/ICR-identifying-age-related-conditions_17_0.png)
    


We can observe that there are spikes in the values of certain features only for certain patients, which could indicate outliers in the data.

We will also create a list of feature columns that will be used for training. We will drop Id from the list since it is not needed.


```python
FEATURE_COLUMNS = [i for i in train_data.columns if i not in ["Id"]]
```

Now let us split the dataset into training and testing datasets:

### KFold validation

Given the limited amount of data available, utilizing K-Fold cross-validation to train our model is a viable option to obtain effective results in the model's scoring.

We will split the dataset into 5 consecutive folds. Each fold is then used once as a validation set while the 4 (5-1) remaining folds form the training set.


```python
# Creates a GroupKFold with 5 splits
kf = KFold(n_splits=5)
```

### Out of Fold (OOF)

During K-Fold cross-validation, the predictions made on the test set of each fold are referred to as Out-of-Fold (OOF) predictions. We will assess the model's performance by aggregating the predictions made across all K (5 in this example) folds.

In our training loop, we will create a pandas dataframe named "oof" to store the predictions of the validation set during each fold.


```python
# Create list of ids for the creation of oof dataframe.
ID_LIST = train_data.index

# Create a dataframe of required size with zero values.
oof = pd.DataFrame(data=np.zeros((len(ID_LIST),1)), index=ID_LIST)

# Create an empty dictionary to store the models trained for each fold.
models = {}

# Create empty dict to save metircs for the models trained for each fold.
accuracy = {}
cross_entropy = {}

# Save the name of the label column to a variable.
label = "Class"
```

## Strategies to handle the dataset imbalance

As mentioned earlier, positive samples account for only about 17% of our entire dataset. This means that our dataset is heavily imbalanced, which can lead to poor predictions by our model.

To address this class imbalance, we will use a strategy called "class weighting."

### Class weighting

Since the postive(`1`) Class labels are only a small fraction of the dataset, we would want the classifier to heavily weight those examples. You can do this by passing Keras weights for each class through a parameter. This will cause the model to "pay more attention" to examples from an under-represented class. 


```python
# Calculate the weight for each label.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
```

    Weight for class 0: 0.61
    Weight for class 1: 2.86


## Train Random Forest model

As mentioned at the beginning of this notebook, we will create a Random Forest model to make our predictions.

Because of the smaller size of the dataset, it is likely that the model will overfit during training. Numerous parameters, primarily `max_depth` and `num_trees` can be changed to fine-tune the model and prevent overfitting.

The attribute `max_depth` indicates the maximum depth of the tree. To avoid overfitting, we can try to reduce the depth of the tree from it's default value, which is `16`. Another way to tackle overfitting is to increase the number of individual decision trees. To do this, we have to increase the value of the parameter `num_trees` from its default value(`300`). In our model we are using `max_depth = 6`, and `num_trees = 550`.

We will train a model for each fold and after training we will store the model and metrics. Here, we have chosen `accuracy` and `binary_crossentropy` as the metrics.


```python
# Loop through each fold
for i, (train_index, valid_index) in enumerate(kf.split(X=train_data)):
        print('##### Fold',i+1)

        # Fetch values corresponding to the index 
        train_df = train_data.iloc[train_index]
        valid_df = train_data.iloc[valid_index]
        valid_ids = valid_df.index.values
        
        # Select only feature columns for training.
        train_df = train_df[FEATURE_COLUMNS]
        valid_df = valid_df[FEATURE_COLUMNS]
        
        # There's one more step required before we can train the model. 
        # We need to convert the datatset from Pandas format (pd.DataFrame)
        # into TensorFlow Datasets format (tf.data.Dataset).
        # TensorFlow Datasets is a high performance data loading library 
        # which is helpful when training neural networks with accelerators like GPUs and TPUs.
        # Note: Some column names contains white spaces at the end of their name, 
        # which is non-comaptible with SavedModels save format. 
        # By default, `pd_dataframe_to_tf_dataset` function will convert 
        # this column names into a compatible format. 
        # So you can safely ignore the warnings related to this.
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
        valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_df, label=label)

        # Define the model and metrics
        rf = tfdf.keras.RandomForestModel(max_depth=6, num_trees=550)
        rf.compile(metrics=["accuracy", "binary_crossentropy"]) 
        
        # Train the model
        # We will train the model using a one-liner.
        # Note: you may see a warning about Autograph. 
        # You can safely ignore this, it will be fixed in the next release.
        # Previously calculated class weights is used to handle imbalance.
        rf.fit(x=train_ds, class_weight=class_weight)
        
        # Store the model
        models[f"fold_{i+1}"] = rf
        
        
        # Predict OOF value for validation data
        predict = rf.predict(x=valid_ds)
        
        # Store the predictions in oof dataframe
        oof.loc[valid_ids, 0] = predict.flatten() 
        
        # Evaluate and store the metrics in respective dicts
        evaluation = rf.evaluate(x=valid_ds,return_dict=True)
        accuracy[f"fold_{i+1}"] = evaluation["accuracy"]
        cross_entropy[f"fold_{i+1}"]= evaluation["binary_crossentropy"]
```

    ##### Fold 1
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Use /tmp/tmpxnlb0nl4 as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:05.455621. Found 493 examples.
    Training model...
    Model trained in 0:00:00.251381
    Compiling model...


    [INFO 23-07-04 03:16:32.6396 UTC kernel.cc:1242] Loading model from path /tmp/tmpxnlb0nl4/model/ with prefix 4e4c9c1323104590
    [INFO 23-07-04 03:16:32.6704 UTC decision_forest.cc:660] Model loaded with 550 root(s), 17894 node(s), and 56 input feature(s).
    [INFO 23-07-04 03:16:32.6704 UTC abstract_model.cc:1311] Engine "RandomForestOptPred" built
    [INFO 23-07-04 03:16:32.6704 UTC kernel.cc:1074] Use fast generic engine


    WARNING: AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7f7e0cc3f2e0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: could not get source code
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    Model compiled.
    1/1 [==============================] - 0s 115ms/step
    1/1 [==============================] - 1s 532ms/step - loss: 0.0000e+00 - accuracy: 0.9677 - binary_crossentropy: 0.2300
    ##### Fold 2
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Use /tmp/tmpq86voi85 as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.700582. Found 493 examples.
    Training model...
    Model trained in 0:00:00.207249
    Compiling model...


    [INFO 23-07-04 03:16:36.4415 UTC kernel.cc:1242] Loading model from path /tmp/tmpq86voi85/model/ with prefix 4dd7e09dbc404ce4
    [INFO 23-07-04 03:16:36.4685 UTC decision_forest.cc:660] Model loaded with 550 root(s), 18114 node(s), and 56 input feature(s).
    [INFO 23-07-04 03:16:36.4687 UTC kernel.cc:1074] Use fast generic engine


    Model compiled.
    1/1 [==============================] - 0s 107ms/step
    1/1 [==============================] - 0s 271ms/step - loss: 0.0000e+00 - accuracy: 0.8790 - binary_crossentropy: 0.2759
    ##### Fold 3
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Use /tmp/tmpo0j3mfxd as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.700287. Found 494 examples.
    Training model...
    Model trained in 0:00:00.203281
    Compiling model...


    [INFO 23-07-04 03:16:38.2617 UTC kernel.cc:1242] Loading model from path /tmp/tmpo0j3mfxd/model/ with prefix 460c04beb9104133
    [INFO 23-07-04 03:16:38.2886 UTC decision_forest.cc:660] Model loaded with 550 root(s), 17492 node(s), and 56 input feature(s).
    [INFO 23-07-04 03:16:38.2887 UTC kernel.cc:1074] Use fast generic engine


    Model compiled.
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 261ms/step - loss: 0.0000e+00 - accuracy: 0.9106 - binary_crossentropy: 0.2910
    ##### Fold 4
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Use /tmp/tmpqjs80nwq as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.703633. Found 494 examples.
    Training model...
    Model trained in 0:00:00.209231
    Compiling model...


    [INFO 23-07-04 03:16:40.0760 UTC kernel.cc:1242] Loading model from path /tmp/tmpqjs80nwq/model/ with prefix 65585f08e96f4eed
    [INFO 23-07-04 03:16:40.1051 UTC decision_forest.cc:660] Model loaded with 550 root(s), 17910 node(s), and 56 input feature(s).
    [INFO 23-07-04 03:16:40.1051 UTC kernel.cc:1074] Use fast generic engine


    Model compiled.
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 260ms/step - loss: 0.0000e+00 - accuracy: 0.9268 - binary_crossentropy: 0.2827
    ##### Fold 5
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Warning: Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    Use /tmp/tmpz3hbp5oe as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:01.310046. Found 494 examples.
    Training model...
    Model trained in 0:00:00.200500
    Compiling model...


    [INFO 23-07-04 03:16:42.4841 UTC kernel.cc:1242] Loading model from path /tmp/tmpz3hbp5oe/model/ with prefix 303fe1451e87465d
    [INFO 23-07-04 03:16:42.5131 UTC decision_forest.cc:660] Model loaded with 550 root(s), 17734 node(s), and 56 input feature(s).
    [INFO 23-07-04 03:16:42.5132 UTC kernel.cc:1074] Use fast generic engine


    Model compiled.
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 266ms/step - loss: 0.0000e+00 - accuracy: 0.9512 - binary_crossentropy: 0.2435


## Visualize the model
Let us pick one model from the `models` dict and select a tree for display.


```python
tfdf.model_plotter.plot_model_in_colab(models['fold_1'], tree_idx=0, max_depth=3)
```





<script src="https://d3js.org/d3.v6.min.js"></script>
<div id="tree_plot_876e8c467b014ec2b39a71eea05db603"></div>
<script>
/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *  Plotting of decision trees generated by TF-DF.
 *
 *  A tree is a recursive structure of node objects.
 *  A node contains one or more of the following components:
 *
 *    - A value: Representing the output of the node. If the node is not a leaf,
 *      the value is only present for analysis i.e. it is not used for
 *      predictions.
 *
 *    - A condition : For non-leaf nodes, the condition (also known as split)
 *      defines a binary test to branch to the positive or negative child.
 *
 *    - An explanation: Generally a plot showing the relation between the label
 *      and the condition to give insights about the effect of the condition.
 *
 *    - Two children : For non-leaf nodes, the children nodes. The first
 *      children (i.e. "node.children[0]") is the negative children (drawn in
 *      red). The second children is the positive one (drawn in green).
 *
 */

/**
 * Plots a single decision tree into a DOM element.
 * @param {!options} options Dictionary of configurations.
 * @param {!tree} raw_tree Recursive tree structure.
 * @param {string} canvas_id Id of the output dom element.
 */
function display_tree(options, raw_tree, canvas_id) {
  console.log(options);

  // Determine the node placement.
  const tree_struct = d3.tree().nodeSize(
      [options.node_y_offset, options.node_x_offset])(d3.hierarchy(raw_tree));

  // Boundaries of the node placement.
  let x_min = Infinity;
  let x_max = -x_min;
  let y_min = Infinity;
  let y_max = -x_min;

  tree_struct.each(d => {
    if (d.x > x_max) x_max = d.x;
    if (d.x < x_min) x_min = d.x;
    if (d.y > y_max) y_max = d.y;
    if (d.y < y_min) y_min = d.y;
  });

  // Size of the plot.
  const width = y_max - y_min + options.node_x_size + options.margin * 2;
  const height = x_max - x_min + options.node_y_size + options.margin * 2 +
      options.node_y_offset - options.node_y_size;

  const plot = d3.select(canvas_id);

  // Tool tip
  options.tooltip = plot.append('div')
                        .attr('width', 100)
                        .attr('height', 100)
                        .style('padding', '4px')
                        .style('background', '#fff')
                        .style('box-shadow', '4px 4px 0px rgba(0,0,0,0.1)')
                        .style('border', '1px solid black')
                        .style('font-family', 'sans-serif')
                        .style('font-size', options.font_size)
                        .style('position', 'absolute')
                        .style('z-index', '10')
                        .attr('pointer-events', 'none')
                        .style('display', 'none');

  // Create canvas
  const svg = plot.append('svg').attr('width', width).attr('height', height);
  const graph =
      svg.style('overflow', 'visible')
          .append('g')
          .attr('font-family', 'sans-serif')
          .attr('font-size', options.font_size)
          .attr(
              'transform',
              () => `translate(${options.margin},${
                  - x_min + options.node_y_offset / 2 + options.margin})`);

  // Plot bounding box.
  if (options.show_plot_bounding_box) {
    svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'none')
        .attr('stroke-width', 1.0)
        .attr('stroke', 'black');
  }

  // Draw the edges.
  display_edges(options, graph, tree_struct);

  // Draw the nodes.
  display_nodes(options, graph, tree_struct);
}

/**
 * Draw the nodes of the tree.
 * @param {!options} options Dictionary of configurations.
 * @param {!graph} graph D3 search handle containing the graph.
 * @param {!tree_struct} tree_struct Structure of the tree (node placement,
 *     data, etc.).
 */
function display_nodes(options, graph, tree_struct) {
  const nodes = graph.append('g')
                    .selectAll('g')
                    .data(tree_struct.descendants())
                    .join('g')
                    .attr('transform', d => `translate(${d.y},${d.x})`);

  nodes.append('rect')
      .attr('x', 0.5)
      .attr('y', 0.5)
      .attr('width', options.node_x_size)
      .attr('height', options.node_y_size)
      .attr('stroke', 'lightgrey')
      .attr('stroke-width', 1)
      .attr('fill', 'white')
      .attr('y', -options.node_y_size / 2);

  // Brackets on the right of condition nodes without children.
  non_leaf_node_without_children =
      nodes.filter(node => node.data.condition != null && node.children == null)
          .append('g')
          .attr('transform', `translate(${options.node_x_size},0)`);

  non_leaf_node_without_children.append('path')
      .attr('d', 'M0,0 C 10,0 0,10 10,10')
      .attr('fill', 'none')
      .attr('stroke-width', 1.0)
      .attr('stroke', '#F00');

  non_leaf_node_without_children.append('path')
      .attr('d', 'M0,0 C 10,0 0,-10 10,-10')
      .attr('fill', 'none')
      .attr('stroke-width', 1.0)
      .attr('stroke', '#0F0');

  const node_content = nodes.append('g').attr(
      'transform',
      `translate(0,${options.node_padding - options.node_y_size / 2})`);

  node_content.append(node => create_node_element(options, node));
}

/**
 * Creates the D3 content for a single node.
 * @param {!options} options Dictionary of configurations.
 * @param {!node} node Node to draw.
 * @return {!d3} D3 content.
 */
function create_node_element(options, node) {
  // Output accumulator.
  let output = {
    // Content to draw.
    content: d3.create('svg:g'),
    // Vertical offset to the next element to draw.
    vertical_offset: 0
  };

  // Conditions.
  if (node.data.condition != null) {
    display_condition(options, node.data.condition, output);
  }

  // Values.
  if (node.data.value != null) {
    display_value(options, node.data.value, output);
  }

  // Explanations.
  if (node.data.explanation != null) {
    display_explanation(options, node.data.explanation, output);
  }

  return output.content.node();
}


/**
 * Adds a single line of text inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {string} text Text to display.
 * @param {!output} output Output display accumulator.
 */
function display_node_text(options, text, output) {
  output.content.append('text')
      .attr('x', options.node_padding)
      .attr('y', output.vertical_offset)
      .attr('alignment-baseline', 'hanging')
      .text(text);
  output.vertical_offset += 10;
}

/**
 * Adds a single line of text inside of a node with a tooltip.
 * @param {!options} options Dictionary of configurations.
 * @param {string} text Text to display.
 * @param {string} tooltip Text in the Tooltip.
 * @param {!output} output Output display accumulator.
 */
function display_node_text_with_tooltip(options, text, tooltip, output) {
  const item = output.content.append('text')
                   .attr('x', options.node_padding)
                   .attr('alignment-baseline', 'hanging')
                   .text(text);

  add_tooltip(options, item, () => tooltip);
  output.vertical_offset += 10;
}

/**
 * Adds a tooltip to a dom element.
 * @param {!options} options Dictionary of configurations.
 * @param {!dom} target Dom element to equip with a tooltip.
 * @param {!func} get_content Generates the html content of the tooltip.
 */
function add_tooltip(options, target, get_content) {
  function show(d) {
    options.tooltip.style('display', 'block');
    options.tooltip.html(get_content());
  }

  function hide(d) {
    options.tooltip.style('display', 'none');
  }

  function move(d) {
    options.tooltip.style('display', 'block');
    options.tooltip.style('left', (d.pageX + 5) + 'px');
    options.tooltip.style('top', d.pageY + 'px');
  }

  target.on('mouseover', show);
  target.on('mouseout', hide);
  target.on('mousemove', move);
}

/**
 * Adds a condition inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!condition} condition Condition to display.
 * @param {!output} output Output display accumulator.
 */
function display_condition(options, condition, output) {
  threshold_format = d3.format('r');

  if (condition.type === 'IS_MISSING') {
    display_node_text(options, `${condition.attribute} is missing`, output);
    return;
  }

  if (condition.type === 'IS_TRUE') {
    display_node_text(options, `${condition.attribute} is true`, output);
    return;
  }

  if (condition.type === 'NUMERICAL_IS_HIGHER_THAN') {
    format = d3.format('r');
    display_node_text(
        options,
        `${condition.attribute} >= ${threshold_format(condition.threshold)}`,
        output);
    return;
  }

  if (condition.type === 'CATEGORICAL_IS_IN') {
    display_node_text_with_tooltip(
        options, `${condition.attribute} in [...]`,
        `${condition.attribute} in [${condition.mask}]`, output);
    return;
  }

  if (condition.type === 'CATEGORICAL_SET_CONTAINS') {
    display_node_text_with_tooltip(
        options, `${condition.attribute} intersect [...]`,
        `${condition.attribute} intersect [${condition.mask}]`, output);
    return;
  }

  if (condition.type === 'NUMERICAL_SPARSE_OBLIQUE') {
    display_node_text_with_tooltip(
        options, `Sparse oblique split...`,
        `[${condition.attributes}]*[${condition.weights}]>=${
            threshold_format(condition.threshold)}`,
        output);
    return;
  }

  display_node_text(
      options, `Non supported condition ${condition.type}`, output);
}

/**
 * Adds a value inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!value} value Value to display.
 * @param {!output} output Output display accumulator.
 */
function display_value(options, value, output) {
  if (value.type === 'PROBABILITY') {
    const left_margin = 0;
    const right_margin = 50;
    const plot_width = options.node_x_size - options.node_padding * 2 -
        left_margin - right_margin;

    let cusum = Array.from(d3.cumsum(value.distribution));
    cusum.unshift(0);
    const distribution_plot = output.content.append('g').attr(
        'transform', `translate(0,${output.vertical_offset + 0.5})`);

    distribution_plot.selectAll('rect')
        .data(value.distribution)
        .join('rect')
        .attr('height', 10)
        .attr(
            'x',
            (d, i) =>
                (cusum[i] * plot_width + left_margin + options.node_padding))
        .attr('width', (d, i) => d * plot_width)
        .style('fill', (d, i) => d3.schemeSet1[i]);

    const num_examples =
        output.content.append('g')
            .attr('transform', `translate(0,${output.vertical_offset})`)
            .append('text')
            .attr('x', options.node_x_size - options.node_padding)
            .attr('alignment-baseline', 'hanging')
            .attr('text-anchor', 'end')
            .text(`(${value.num_examples})`);

    const distribution_details = d3.create('ul');
    distribution_details.selectAll('li')
        .data(value.distribution)
        .join('li')
        .append('span')
        .text(
            (d, i) =>
                'class ' + i + ': ' + d3.format('.3%')(value.distribution[i]));

    add_tooltip(options, distribution_plot, () => distribution_details.html());
    add_tooltip(options, num_examples, () => 'Number of examples');

    output.vertical_offset += 10;
    return;
  }

  if (value.type === 'REGRESSION') {
    display_node_text(
        options,
        'value: ' + d3.format('r')(value.value) + ` (` +
            d3.format('.6')(value.num_examples) + `)`,
        output);
    return;
  }

  display_node_text(options, `Non supported value ${value.type}`, output);
}

/**
 * Adds an explanation inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!explanation} explanation Explanation to display.
 * @param {!output} output Output display accumulator.
 */
function display_explanation(options, explanation, output) {
  // Margin before the explanation.
  output.vertical_offset += 10;

  display_node_text(
      options, `Non supported explanation ${explanation.type}`, output);
}


/**
 * Draw the edges of the tree.
 * @param {!options} options Dictionary of configurations.
 * @param {!graph} graph D3 search handle containing the graph.
 * @param {!tree_struct} tree_struct Structure of the tree (node placement,
 *     data, etc.).
 */
function display_edges(options, graph, tree_struct) {
  // Draw an edge between a parent and a child node with a bezier.
  function draw_single_edge(d) {
    return 'M' + (d.source.y + options.node_x_size) + ',' + d.source.x + ' C' +
        (d.source.y + options.node_x_size + options.edge_rounding) + ',' +
        d.source.x + ' ' + (d.target.y - options.edge_rounding) + ',' +
        d.target.x + ' ' + d.target.y + ',' + d.target.x;
  }

  graph.append('g')
      .attr('fill', 'none')
      .attr('stroke-width', 1.2)
      .selectAll('path')
      .data(tree_struct.links())
      .join('path')
      .attr('d', draw_single_edge)
      .attr(
          'stroke', d => (d.target === d.source.children[0]) ? '#0F0' : '#F00');
}

display_tree({"margin": 10, "node_x_size": 160, "node_y_size": 28, "node_x_offset": 180, "node_y_offset": 33, "font_size": 10, "edge_rounding": 20, "node_padding": 2, "show_plot_bounding_box": false}, {"value": {"type": "PROBABILITY", "distribution": [0.5227602930870747, 0.4772397069129253], "num_examples": 478.8338456749916}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "FR", "threshold": 2.7061350345611572}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.014929499065309356, 0.9850705009346906], "num_examples": 40.59683209657669}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "CC", "threshold": 0.6739449501037598}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.050373132637029645, 0.9496268673629703], "num_examples": 12.032016575336456}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 1.0], "num_examples": 28.564815521240234}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.569804063557375, 0.43019593644262494], "num_examples": 438.2370135784149}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "DI", "threshold": 268.67132568359375}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.06605504369032043, 0.9339449563096796], "num_examples": 45.877675116062164}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "EL", "threshold": 85.41004180908203}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 1.0], "num_examples": 31.421297073364258}}, {"value": {"type": "PROBABILITY", "distribution": [0.20962732333445935, 0.7903726766655407], "num_examples": 14.456378042697906}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.628706277951115, 0.371293722048885], "num_examples": 392.35933846235275}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "FL", "threshold": 10.433052062988281}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.13424047737713196, 0.865759522622868], "num_examples": 85.7842373251915}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "EG", "threshold": 1536.67919921875}}, {"value": {"type": "PROBABILITY", "distribution": [0.767065105619419, 0.23293489438058096], "num_examples": 306.57510113716125}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "DN", "threshold": 25.685535430908203}}]}]}]}, "#tree_plot_876e8c467b014ec2b39a71eea05db603")
</script>




## Evaluate the model on the Out of bag (OOB) data and the validation dataset
When training our model, we separate the data into training and validation (`valid_ds`) sets.

We can also use Out of bag (OOB) score to validate our RandomForestModel. To train a Random Forest Model, a set of random samples from training set are choosen by the algorithm and the rest of the samples are used to finetune the model. The subset of data that is not chosen is known as Out of bag data (OOB). OOB score is computed on the OOB data.

The training logs show the binary_crossentropy evaluated on the out of bag dataset according to the number of trees in the model. Let us plot this for the models of each fold.

Note: Smaller values are better for this hyperparameter.


```python
figure, axis = plt.subplots(3, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, fold_no in enumerate(models.keys()):
    row = i//2
    col = i % 2
    logs = models[fold_no].make_inspector().training_logs()
    axis[row, col].plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    axis[row, col].set_title(f"Fold {i+1}")
    axis[row, col].set_xlabel('Number of trees')
    axis[row, col].set_ylabel('Loss (out-of-bag)')

axis[2][1].set_visible(False)
plt.show()
```


    
![png](../../images/ICR-identifying-age-related-conditions_files/ICR-identifying-age-related-conditions_33_0.png)
    



```python
figure, axis = plt.subplots(3, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, fold_no in enumerate(models.keys()):
    row = i//2
    col = i % 2
    logs = models[fold_no].make_inspector().training_logs()
    axis[row, col].plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    axis[row, col].set_title(f"Fold {i+1}")
    axis[row, col].set_xlabel('Number of trees')
    axis[row, col].set_ylabel('Accuracy (out-of-bag)')

axis[2][1].set_visible(False)
plt.show()
```


    
![png](../../images/ICR-identifying-age-related-conditions_files/ICR-identifying-age-related-conditions_34_0.png)
    


In the initial plots, we can observe how the loss decreases and approaches zero as the number of trees increases. In the subsequent plots, we see how the accuracy changes as the number of trees increases. By examining these plots, we can see that even with an increasing number of trees, our scores for both loss and accuracy would not improve significantly. That is why we decided to stick with 550 trees with a depth of 6. 

In the first set of graphs, we can observe a gradual decrease in the loss value as the number of trees increases. These loss values tend towards zero, indicating an improvement in the model's ability to fit the data.

In the following graphs, we see how the accuracy value changes as the number of trees increases. It is noticeable that as the number of trees increases, the accuracy also increases, indicating better performance of the model.

However, by examining these graphs, we can see that increasing the number of trees does not necessarily lead to a significant improvement in the loss and accuracy scores. Beyond a certain point, the benefit of adding more trees becomes marginal and does not justify the increase in complexity and computational cost.

That is why the decision was made to stick with `550` trees and a depth of `6` since it was considered that these values offer a suitable balance between performance and computational efficiency, without significantly compromising the achieved loss and accuracy scores.

We can also see some general stats on the OOB dataset:


```python
for _model in models:
    inspector = models[_model].make_inspector()
    print("*"*80)
    print(_model, inspector.evaluation())
```

    ********************************************************************************
    fold_1 Evaluation(num_examples=493, accuracy=0.8657528687111298, loss=0.39946834217178023, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)
    ********************************************************************************
    fold_2 Evaluation(num_examples=493, accuracy=0.9065865170858219, loss=0.3661344547861831, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)
    ********************************************************************************
    fold_3 Evaluation(num_examples=494, accuracy=0.860904312570069, loss=0.3798962228316164, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)
    ********************************************************************************
    fold_4 Evaluation(num_examples=494, accuracy=0.8821551504777397, loss=0.38469492520783366, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)
    ********************************************************************************
    fold_5 Evaluation(num_examples=494, accuracy=0.8646970145843453, loss=0.378995601555129, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)


Now, let us check the evaluation metrics for each fold and its average value.


```python
average_loss = 0
average_acc = 0

for _model in  models:
    average_loss += cross_entropy[_model]
    average_acc += accuracy[_model]
    print(f"{_model}: acc: {accuracy[_model]:.4f} loss: {cross_entropy[_model]:.4f}")

print("\n")
print("*"*80)
print(f"Average accuracy: {average_acc/5:.4f}  Average loss: {average_loss/5:.4f}")
print("*"*80)
```

    fold_1: acc: 0.9677 loss: 0.2300
    fold_2: acc: 0.8790 loss: 0.2759
    fold_3: acc: 0.9106 loss: 0.2910
    fold_4: acc: 0.9268 loss: 0.2827
    fold_5: acc: 0.9512 loss: 0.2435
    
    
    ********************************************************************************
    Average accuracy: 0.9271  Average loss: 0.2646
    ********************************************************************************


The model performs very well despite the small amount of data available. By adjusting the number of trees and their depth, we were able to achieve an average precision of nearly 93% and a loss of approximately 26%. These results are highly encouraging and demonstrate the effectiveness of the model despite the limitations in the amount of data used.








