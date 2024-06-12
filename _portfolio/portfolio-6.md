---
title: "Breast Cancer"
excerpt: "In this notebook, we present the project of training a naive Bayesian classifier to predict patients potentially affected by breast cancer using scikit-learn."
collection: portfolio
---




In this notebook, we present the project of training a naive Bayesian classifier to predict patients potentially affected by breast cancer, given the following parameters:

1. `Sample ID Code`
2. `Tumor Thickness` 1–10
3. `Uniformity of Cell Size` 1–10
4. `Uniformity of Cell Shape` 1–10
5. `Marginal Adhesion` 1–10
6. `Epithelial Cell Size` 1–10
7. `Bare Nuclei` 1–10
8. `Bland Chromatin` 1–10
9. `Normal Nucleoli` 1–10
10. `Mitosis of Cells` 1–10

## Importamos librerias


En esta ocasión, nos apoyaremos en la biblioteca scikit-learn, que cuenta con varios clasificadores y una herramienta muy sencilla para entrenar nuestros modelos.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
```

    /tmp/ipykernel_7412/1843597498.py:2: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
            
      import pandas as pd


## 📚 Load Data 


```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
column_names = ['id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
data = pd.read_csv(url, names=column_names)

```


```python
data.head()
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
      <th>id</th>
      <th>clump_thickness</th>
      <th>uniformity_of_cell_size</th>
      <th>uniformity_of_cell_shape</th>
      <th>marginal_adhesion</th>
      <th>single_epithelial_cell_size</th>
      <th>bare_nuclei</th>
      <th>bland_chromatin</th>
      <th>normal_nucleoli</th>
      <th>mitoses</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



We are not interested in id column


```python
data = data.drop(['id'], axis=1)
```


```python
num_cols = 3
num_rows = (len(data.columns) + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4))

for i, col in enumerate(data.columns):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].hist(data[data.columns[i]].dropna(), bins=50)
    axs[row, col].set_title(data.columns[i])

# Eliminar los gráficos vacíos
for i in range(len(data.columns), num_rows*num_cols):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()
```


    
![png](../../images/cancer_seno_files/cancer_seno_10_0.png)
    


## 🚜 Pre-Processing


```python
data = data.replace('?', np.nan)
data.isnull().sum()
```




    clump_thickness                 0
    uniformity_of_cell_size         0
    uniformity_of_cell_shape        0
    marginal_adhesion               0
    single_epithelial_cell_size     0
    bare_nuclei                    16
    bland_chromatin                 0
    normal_nucleoli                 0
    mitoses                         0
    class                           0
    dtype: int64



In this new block of cells, we are informed that there are records where the "bare_nuclei" variable does not have an assigned value. To handle this situation, we use the sklearn library and its "SimpleImputer" module. This module allows us to fill in the missing values with the mean of the values in that specific column by using the parameter "strategy=mean".



```python
# Manejar los datos faltantes
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)
```

## ✂️ Split data


```python
# Dividir el conjunto de datos
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=0)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)
```


```python
train_data.shape, valid_data.shape, test_data.shape
```




    ((419, 10), (140, 10), (140, 10))



We get the labes


```python
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_valid = valid_data[:, :-1]
y_valid = valid_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]
```

## Classifiers

Regarding classifiers, we will use four:

- **GaussianNB**: This is a Naive Bayes classifier that assumes the data for each feature follows a Gaussian (or normal) distribution. It is especially useful in cases where the features are continuous.

- **BernoulliNB**: This is another Naive Bayes classifier used when all your features are binary (i.e., take values of 0 or 1). This classifier assumes that all features are independent of each other.

- **RandomForestClassifier**: This is an ensemble classifier that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes of the individual trees for prediction. In your case, you are building a forest with 100 trees (n_estimators=100) and setting a seed for the random number generator to ensure the reproducibility of your results (random_state=0).

- **MultinomialNB**: This is another Naive Bayes classifier used when the features are categorical variables or continuous variables that have been converted into counts. This classifier also assumes that all features are independent of each other.



```python
clasificadores = [GaussianNB(), BernoulliNB(), RandomForestClassifier(n_estimators=100, random_state=0), MultinomialNB()]
clasificadores_nombres = ['GaussianNB', 'BernoulliNB', 'RandomForest', 'MultinomialNB']
```


```python
acuaracy_valid = []
acuaracy_test = []
for i, clf in enumerate(clasificadores):
    clf.fit(X_train, y_train)
    valid_pred = clf.predict(X_valid)
    y_test_pred = clf.predict(X_test)
    acuaracy_valid.append(accuracy_score(y_valid, valid_pred))
    acuaracy_test.append(accuracy_score(y_test, y_test_pred))
    print(f"Accuracy en el conjunto de validacion para {clasificadores_nombres[i]}: {accuracy_score(y_valid, valid_pred)}")
    print("-"*100)
    print(f"Accuracy en el conjunto de test para {clasificadores_nombres[i]}: {accuracy_score(y_test, y_test_pred)}")
    print("="*100)
```

    Accuracy en el conjunto de validacion para GaussianNB: 0.9214285714285714
    ----------------------------------------------------------------------------------------------------
    Accuracy en el conjunto de test para GaussianNB: 0.9714285714285714
    ====================================================================================================
    Accuracy en el conjunto de validacion para BernoulliNB: 0.5714285714285714
    ----------------------------------------------------------------------------------------------------
    Accuracy en el conjunto de test para BernoulliNB: 0.7142857142857143
    ====================================================================================================
    Accuracy en el conjunto de validacion para RandomForest: 0.9357142857142857
    ----------------------------------------------------------------------------------------------------
    Accuracy en el conjunto de test para RandomForest: 0.9785714285714285
    ====================================================================================================
    Accuracy en el conjunto de validacion para MultinomialNB: 0.8785714285714286
    ----------------------------------------------------------------------------------------------------
    Accuracy en el conjunto de test para MultinomialNB: 0.9
    ====================================================================================================



```python
# Crear el gráfico de barras
x = range(len(clasificadores_nombres))
plt.bar(x, acuaracy_valid, width=0.4, label='Validation', color='b', align='center')
plt.bar(x, acuaracy_test, width=0.4, label='Test', color='r', align='edge')

# Añadir etiquetas
plt.xlabel('Clasificadores')
plt.ylabel('Accuracy')
plt.title('Accuracy de los clasificadores en test y validation')
plt.xticks(x, clasificadores_nombres)
plt.legend()

# Mostrar el gráfico
plt.show()
```


    
![png](../../images/cancer_seno_files/cancer_seno_23_0.png)
    


The results show the accuracy of the four classifiers on the validation and test sets. Accuracy is a measure of how many predictions the model got right relative to all the predictions it made.

- **GaussianNB**: This classifier has high accuracy on both the validation and test sets, indicating it is performing well. This could be because the features follow a Gaussian distribution, which is a key assumption of this classifier.

- **BernoulliNB**: This classifier has significantly lower accuracy compared to the other classifiers. This could be because the features are not binary, which is a key assumption of this classifier. If the features are not binary, this classifier may not perform well.

- **RandomForestClassifier**: This classifier has the highest accuracy on both sets, indicating it is performing very well. Random forests are robust ensemble methods that tend to perform well on a variety of datasets.

- **MultinomialNB**: This classifier has reasonably high accuracy, but not as high as GaussianNB or RandomForest. This could be because the features are not counts or frequencies of events, which is a key assumption of this classifier.