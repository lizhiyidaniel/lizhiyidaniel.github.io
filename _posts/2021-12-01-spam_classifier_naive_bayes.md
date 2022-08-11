---
layout: post
title: "Use Naive Bayes to classify spam messages"
subtitle: "Use Multinomial Naive Bayes model to classify spam messages and achieved 0.99 precision, recall and f1 score"
background: '/img/bg-about.jpg'
---

# View the original Github Repo: 
Naive Bayes is a popular rstatistical technique for e-mail filtering. It uses bag-of-words features (use countvectorizer to create the features from the text) to identify the email spam. <br>
It uses a Bayes' theorem to calculate a probability that an email is spam or not and it usually has low false positive spam detection rates, meaning that it rarely misclassify an legitimate email as spam. However, as its name suggest, it assumes strong and naive assumption about the independence of the words, which is rarely the case for words in the context. 
[Click here](https://github.com/lizhiyidaniel/spam_classifier_naive_bayes)
# Import the library


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# Explore the datasets

The datasets has been collected for SMS Spam research and it has two columns. One columns has the text and another shows the label(ham or spam)


```python
spam_df = pd.read_csv("emails.csv")
```


```python
spam_df.head()
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
      <th>text</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Subject: naturally irresistible your corporate...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Subject: the stock trading gunslinger  fanny i...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Subject: unbelievable new homes made easy  im ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Subject: 4 color printing special  request add...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Subject: do not have money , get software cds ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
spam_df.tail()
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
      <th>text</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5723</th>
      <td>Subject: re : research and development charges...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5724</th>
      <td>Subject: re : receipts from visit  jim ,  than...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5725</th>
      <td>Subject: re : enron case study update  wow ! a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5726</th>
      <td>Subject: re : interest  david ,  please , call...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5727</th>
      <td>Subject: news : aurora 5 . 2 update  aurora ve...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
spam_df.describe()
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
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5728.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.238827</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.426404</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
spam_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5728 entries, 0 to 5727
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   text    5728 non-null   object
     1   spam    5728 non-null   int64 
    dtypes: int64(1), object(1)
    memory usage: 89.6+ KB



```python
spam_df["spam"].value_counts(normalize=True)
```




    0    0.761173
    1    0.238827
    Name: spam, dtype: float64



# Visualize the datasets


```python
spam_df["length"] = spam_df["text"].apply(len)
spam_df.head()
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
      <th>text</th>
      <th>spam</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Subject: naturally irresistible your corporate...</td>
      <td>1</td>
      <td>1484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Subject: the stock trading gunslinger  fanny i...</td>
      <td>1</td>
      <td>598</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Subject: unbelievable new homes made easy  im ...</td>
      <td>1</td>
      <td>448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Subject: 4 color printing special  request add...</td>
      <td>1</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Subject: do not have money , get software cds ...</td>
      <td>1</td>
      <td>235</td>
    </tr>
  </tbody>
</table>
</div>




```python
spam_df["length"].plot(bins=100, kind="hist")
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](/img/spam/output_12_1.png)
    



```python
spam_df.length.describe()
```




    count     5728.000000
    mean      1556.768680
    std       2042.649812
    min         13.000000
    25%        508.750000
    50%        979.000000
    75%       1894.250000
    max      43952.000000
    Name: length, dtype: float64



# Clean the datasets


```python
#remove punctutation
import string
```


```python
#remove stopwords
from nltk.corpus import stopwords
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(spam_df["text"], spam_df["spam"], test_size=0.2)
```


```python
def clean_message(message):
    message = [char for char in message if char not in string.punctuation]
    message = "".join(message)
    message = [word for word in message.split() if word.lower() not in stopwords.words("english")]
    message = " ".join(message) 
    return message

```


```python
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()#min_df=0.01, max_features=300, stop_words="english")
```


```python
CV.fit(X_train.apply(clean_message))
```




    CountVectorizer()




```python
X_train = CV.transform(X_train)
X_test = CV.transform(X_test)
```

# Training the models


```python
from sklearn.naive_bayes import MultinomialNB
```


```python
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
```




    MultinomialNB()



# Evaluate the model


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
#predict the training set
y_predict_train = NB_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
```




    <AxesSubplot:>




    
![png](/img/spam/output_28_1.png)
    



```python
#predict the test set
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
```




    <AxesSubplot:>




    
![png](/img/spam/output_29_1.png)
    



```python
print(classification_report(y_test, y_predict_test))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       878
               1       0.99      0.99      0.99       268
    
        accuracy                           0.99      1146
       macro avg       0.99      0.99      0.99      1146
    weighted avg       0.99      0.99      0.99      1146
    

