---
title: "Spam Predictor"
excerpt: "In this notebook, naive Bayesian classifiers are implemented using two different probability distributions: Bernoulli and Categorical, to classify emails as spam or not spam."
collection: portfolio
---



In this notebook, naive Bayesian classifiers are implemented using two different probability distributions: Bernoulli and Categorical, to classify emails as spam or not spam.

## Libraries required


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
```

    /tmp/ipykernel_6533/945356947.py:1: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
            
      import pandas as pd


## Load Data

**Naming the columns of the dataset:**

We use $w_i$ to refer to the $i$ - th word, while the column $\text{spam}$ indicates whether the email is considered as spam or not.


```python
url = 'http://turing.iimas.unam.mx/~gibranfp/cursos/aprendizaje_automatizado/data/spam.csv'
columns = [f'w{i}' for i in range(2000)] + ['spam']
data = pd.read_csv(url,names=columns, delimiter=' ')
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
      <th>w0</th>
      <th>w1</th>
      <th>w2</th>
      <th>w3</th>
      <th>w4</th>
      <th>w5</th>
      <th>w6</th>
      <th>w7</th>
      <th>w8</th>
      <th>w9</th>
      <th>...</th>
      <th>w1991</th>
      <th>w1992</th>
      <th>w1993</th>
      <th>w1994</th>
      <th>w1995</th>
      <th>w1996</th>
      <th>w1997</th>
      <th>w1998</th>
      <th>w1999</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2001 columns</p>
</div>



## Percentage of emails labeled as spam and not spam


```python
# Numero total de correos
n_correos = data.shape[0]
# Numero de correos spam
n_spam = data['spam'].sum()
# Numero de correos no spam
n_no_spam = n_correos - n_spam

print(f'Numero total de correos: {n_correos}')
print(f'Porcentaje de correos spam: {n_spam/n_correos*100:.2f}%')
print(f'Porcentaje de correos no spam: {n_no_spam/n_correos*100:.2f}%')
```

    Numero total de correos: 5172
    Porcentaje de correos spam: 29.00%
    Porcentaje de correos no spam: 71.00%


## Data Splitting:

We split the data randomly into 60% for training, 20% for validation, and 20% for testing purposes.


```python
# Convertir el dataframe a un array de numpy
data = data.to_numpy()
# Mezclar los datos aleatoriamente
np.random.seed(0)
np.random.shuffle(data)
# Calcular los índices de división
train_end = int(0.6 * len(data))
valid_end = int(0.8 * len(data))

# Dividir los datos en entrenamiento, validación y prueba
train_data = data[:train_end] # 60%
valid_data = data[train_end:valid_end] # 20%
test_data = data[valid_end:] # 20%

# Separar los datos en palabras y etiquetas
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

X_valid = valid_data[:, :-1]
y_valid = valid_data[:, -1]

X_test = test_data[:, :-1]
y_test = test_data[:, -1]
```

## Bernoulli Naive Bayes Classifier


```python
def bernoulli(x, q):
    return q**x * (1-q)**(1-x)
```

In the training data, we have a total of 3103 samples, each with 2000 attributes. According to Bayes' theorem, we have the following relationship: 

$$P(spam \| email) = \frac{P(email \| spam) \cdot P(spam)}{P(email)}$$

Here, an email can be viewed as an array of words $$email = [w_1, w_2, ..., w_{2000}]$$. 

Given our naive assumption that these attributes are independent given the class, we have: 
$$P(email \| spam) = \prod_{i=1}^{2000} p(w_i \| spam)$$ 

For each word given the class, we assume a Bernoulli distribution: $$P(w_i=x \| spam) = q_i^x (1-q_i)^{1-x} \quad \text{for } x \in \{0,1\}$$ 

Using the maximum likelihood estimator, we find each $$ q_{i}$$ for each word: 

$$q_{i} = \frac{1}{n}\sum_{j=1}^{n} w_{ij}$$


```python
# Definición de la clase BernoulliNB
class BernoulliNB:
    # Método para entrenar el modelo
    def fit(self, X, y):
        # Obtenemos las clases únicas de los datos de salida
        self.classes = np.unique(y)
        # Obtenemos el número de clases únicas
        self.n_classes = self.classes.size
        # Obtenemos el número de palabras (características)
        self.n_words = X.shape[1]
        # Obtenemos el número total de ejemplos
        n_total = X.shape[0]
        
        # Inicializamos las probabilidades de las palabras y las clases
        self.qw = np.zeros((self.n_classes, self.n_words))
        self.pc = np.zeros(self.n_classes)
        # Para cada clase, calculamos las probabilidades de las palabras y las clases
        for i, c in enumerate(self.classes):
            # Obtenemos los ejemplos de la clase actual
            X_c = X[y == c]
            # Obtenemos el número de ejemplos de la clase actual
            nc = X_c.shape[0]

            # Contamos las palabras en los ejemplos de la clase actual
            counts = np.sum(X_c, axis=0)
            # Calculamos las probabilidades de las palabras para la clase actual
            self.qw[i, :] = counts / nc
            # Calculamos la probabilidad de la clase actual
            self.pc[i] = nc / n_total   
    
    # Método para predecir las probabilidades de las clases para los datos de entrada
    def predict_proba(self, X):
        # Obtenemos el número de ejemplos
        n = X.shape[0]
        
        # Inicializamos las probabilidades de las clases
        prop = np.zeros((n, self.n_classes))
        # Para cada clase, calculamos las probabilidades de las clases
        for i, c in enumerate(self.classes):
            # Calculamos las probabilidades de las clases para los datos de entrada
            prop[:, i] = np.prod(bernoulli(X, self.qw[i, :]), axis=1) * self.pc[i]
        # Devolvemos las probabilidades de las clases
        return prop

    # Método para predecir las clases para los datos de entrada
    def predict(self, X):
        # Predecimos las clases para los datos de entrada
        return np.argmax(self.predict_proba(X), axis=1)
```


```python
naiveBayesB = BernoulliNB()
naiveBayesB.fit(X_train, y_train)
# Predicciones con el conjunto de validación
y_predB_valid = naiveBayesB.predict(X_valid)
y_predB_test = naiveBayesB.predict(X_test)

```


```python
print(f'Accuracy en el conjunto test: {accuracy_score(y_test, y_predB_test)}')
print(f'Accuracy en el conjunto validación: {accuracy_score(y_valid, y_predB_valid)}')
```

    Accuracy en el conjunto test: 0.7101449275362319
    Accuracy en el conjunto validación: 0.7176015473887815


## Categorical Bayesian Classifier

In contrast to the previous Bayesian classifier, we now employ a categorical distribution, which is a generalization of the Bernoulli distribution. Additionally, in this case, we use the maximum a posteriori likelihood estimator to calculate the parameters. Each parameter is computed as follows:

$$\frac{{\text{{feature\_counts}} + \alpha - 1}}{{\text{{class\_counts}} + \alpha \cdot \text{{n\_features}} - \text{{n\_classes}}}}$$

This estimator takes into account both the feature counts and the class counts, and utilizes a smoothing parameter $\alpha$.


```python
# Definición de la clase CategoricalNB
class CategoricalNB:
    # Método constructor con parámetro de suavizado alpha
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Parámetro de suavizado

    # Método para entrenar el modelo
    def fit(self, X, y):
        # Obtenemos las clases únicas de los datos de salida
        self.classes = np.unique(y)
        # Obtenemos el número de clases únicas
        self.n_classes = self.classes.size
        # Obtenemos el número de características
        self.n_features = X.shape[1]
        # Obtenemos el número total de ejemplos
        n_total = X.shape[0]
        
        # Inicializamos los conteos de las características y las clases
        self.feature_counts = np.zeros((self.n_classes, self.n_features))
        self.class_counts = np.zeros(self.n_classes)
        
        # Para cada clase, calculamos los conteos de las características y las clases
        for i, c in enumerate(self.classes):
            # Obtenemos los ejemplos de la clase actual
            X_c = X[y == c]
            # Obtenemos el número de ejemplos de la clase actual
            nc = X_c.shape[0]

            # Contamos las características en los ejemplos de la clase actual
            self.feature_counts[i, :] = np.sum(X_c, axis=0)
            # Contamos los ejemplos de la clase actual
            self.class_counts[i] = nc

        # Aplicamos suavizado de Laplace a las probabilidades de las características y las clases
        self.feature_probs = (self.feature_counts + self.alpha-1) / (self.class_counts[:, None] + self.alpha * self.n_features-self.n_classes)
        self.class_probs = self.class_counts / n_total

    # Método para predecir las probabilidades de las clases para los datos de entrada
    def predict_proba(self, X):
        # Obtenemos el número de ejemplos
        n = X.shape[0]
        
        # Inicializamos las probabilidades de las clases
        prop = np.zeros((n, self.n_classes))
        # Para cada clase, calculamos las probabilidades de las clases
        for i, c in enumerate(self.classes):
            # Calculamos las probabilidades de las clases para los datos de entrada
            prop[:, i] = np.prod(np.power(self.feature_probs[i, :], X) * np.power(1 - self.feature_probs[i, :], 1 - X), axis=1) * self.class_probs[i]
        # Devolvemos las probabilidades de las clases
        return prop

    # Método para predecir las clases para los datos de entrada
    def predict(self, X):
        # Predecimos las clases para los datos de entrada
        return np.argmax(self.predict_proba(X), axis=1)
```


```python
catM = CategoricalNB()
catM.fit(X_train, y_train)
# Predicciones con el conjunto de validación
y_predC_valid = catM.predict(X_valid)
# Predicciones con el conjunto de test
y_predC_test = catM.predict(X_test)

```


```python
print(f'Accuracy en el conjunto test: {accuracy_score(y_test, y_predC_test)}')
print(f'Accuracy en el conjunto validación: {accuracy_score(y_valid, y_predC_valid)}')
```

    Accuracy en el conjunto test: 0.9458937198067633
    Accuracy en el conjunto validación: 0.9448742746615088


The accuracy results indicate that the CategoricalNB classifier performs significantly better than the BernoulliNB classifier both on the test and validation sets.

- The BernoulliNB classifier has an accuracy of approximately 0.71 on both sets. This means it correctly predicts 71% of cases.

- In contrast, the CategoricalNB classifier achieves an accuracy of approximately 0.945 on both sets. This indicates it correctly predicts 94.5% of cases.

The difference in accuracy could stem from various factors, including the data's nature, the distribution of features, and each classifier's ability to model the relationship between features and the target variable. In this case, it appears that the CategoricalNB classifier is better suited for these specific data.

- The BernoulliNB classifier treats each word as a binary feature indicating its presence or absence in the email.

- On the other hand, the CategoricalNB classifier treats each word as a categorical feature indicating its frequency in the email.

Given that each email contains approximately 2000 words, word frequency likely provides more information than just presence or absence. This could explain why the CategoricalNB classifier outperforms the BernoulliNB classifier significantly in this scenario.


