

```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
```


```python
#Load the dataset

AH_data = pd.read_csv("/Users/Rohit/Desktop/nesarc_pds.csv")
data_clean = AH_data.dropna()
```


```python
data_clean.dtypes
```




    ETHRACE2A          int64
    ETOTLCA2          object
    IDNUM              int64
    PSU                int64
    STRATUM            int64
    WEIGHT           float64
    CDAY               int64
    CMON               int64
    CYEAR              int64
    REGION             int64
    CENDIV             int64
    CCS                int64
    FIPSTATE           int64
    BUILDTYP           int64
    NUMPERS            int64
    NUMPER18           int64
    NUMREL             int64
    NUMREL18           int64
    CHLD0              int64
    CHLD1_4            int64
    CHLD5_12           int64
    CHLD13_15          int64
    CHLD16_17          int64
    CHLD0_17           int64
    SPOUSE             int64
    FATHERIH           int64
    MOTHERIH           int64
    ADULTCH            int64
    OTHREL             int64
    NONREL             int64
                      ...   
    DEPPDDX2           int64
    OBCOMDX2           int64
    PARADX2            int64
    SCHIZDX2           int64
    HISTDX2            int64
    ALCABDEP12DX       int64
    ALCABDEPP12DX      int64
    TAB12MDX           int64
    TABP12MDX          int64
    TABLIFEDX          int64
    STIM12ABDEP        int64
    STIMP12ABDEP       int64
    PAN12ABDEP         int64
    PANP12ABDEP        int64
    SED12ABDEP         int64
    SEDP12ABDEP        int64
    TRAN12ABDEP        int64
    TRANP12ABDEP       int64
    COC12ABDEP         int64
    COCP12ABDEP        int64
    SOL12ABDEP         int64
    SOLP12ABDEP        int64
    HAL12ABDEP         int64
    HALP12ABDEP        int64
    MAR12ABDEP         int64
    MARP12ABDEP        int64
    HER12ABDEP         int64
    HERP12ABDEP        int64
    OTHB12ABDEP        int64
    OTHBP12ABDEP       int64
    dtype: object




```python
data_clean.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ETHRACE2A</th>
      <th>IDNUM</th>
      <th>PSU</th>
      <th>STRATUM</th>
      <th>WEIGHT</th>
      <th>CDAY</th>
      <th>CMON</th>
      <th>CYEAR</th>
      <th>REGION</th>
      <th>CENDIV</th>
      <th>...</th>
      <th>SOL12ABDEP</th>
      <th>SOLP12ABDEP</th>
      <th>HAL12ABDEP</th>
      <th>HALP12ABDEP</th>
      <th>MAR12ABDEP</th>
      <th>MARP12ABDEP</th>
      <th>HER12ABDEP</th>
      <th>HERP12ABDEP</th>
      <th>OTHB12ABDEP</th>
      <th>OTHBP12ABDEP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>...</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
      <td>43093.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.087764</td>
      <td>21547.000000</td>
      <td>27312.909544</td>
      <td>2726.858747</td>
      <td>4823.981575</td>
      <td>16.146195</td>
      <td>8.589632</td>
      <td>2001.141578</td>
      <td>2.636321</td>
      <td>5.142993</td>
      <td>...</td>
      <td>0.000255</td>
      <td>0.003922</td>
      <td>0.001532</td>
      <td>0.017776</td>
      <td>0.018634</td>
      <td>0.095027</td>
      <td>0.000348</td>
      <td>0.004618</td>
      <td>0.000093</td>
      <td>0.001230</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.560799</td>
      <td>12440.021912</td>
      <td>16019.733641</td>
      <td>1595.979984</td>
      <td>3485.046966</td>
      <td>8.801055</td>
      <td>3.051984</td>
      <td>0.348620</td>
      <td>1.031667</td>
      <td>2.511825</td>
      <td>...</td>
      <td>0.018655</td>
      <td>0.079789</td>
      <td>0.050501</td>
      <td>0.169523</td>
      <td>0.186201</td>
      <td>0.383204</td>
      <td>0.030082</td>
      <td>0.106426</td>
      <td>0.015233</td>
      <td>0.047429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1001.000000</td>
      <td>101.000000</td>
      <td>398.037382</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2001.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>10774.000000</td>
      <td>12044.000000</td>
      <td>1209.000000</td>
      <td>2240.897957</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>2001.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>21547.000000</td>
      <td>27018.000000</td>
      <td>2701.000000</td>
      <td>3723.955061</td>
      <td>16.000000</td>
      <td>9.000000</td>
      <td>2001.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>32320.000000</td>
      <td>40019.000000</td>
      <td>4004.000000</td>
      <td>7013.033942</td>
      <td>24.000000</td>
      <td>10.000000</td>
      <td>2001.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>43093.000000</td>
      <td>56017.000000</td>
      <td>5605.000000</td>
      <td>57902.204790</td>
      <td>31.000000</td>
      <td>12.000000</td>
      <td>2002.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 665 columns</p>
</div>




```python
#Split into training and testing sets

predictors = data_clean[['NUMREL','FATHERIH','MOTHERIH','ADULTCH','OTHREL','NONREL','SEX']]

#Target; Alcohol over
targets = data_clean.S2AQ1

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4)
```


```python
pred_train.shape
```




    (25855, 7)




```python
pred_test.shape
```




    (17238, 7)




```python
tar_train.shape
```




    (25855,)




```python
tar_test.shape
```




    (17238,)




```python
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)
```


```python
sklearn.metrics.confusion_matrix(tar_test,predictions)
```




    array([[13817,    57],
           [ 3326,    38]])




```python
sklearn.metrics.accuracy_score(tar_test, predictions)
```




    0.80374753451676528




```python
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)
```

    [ 0.24300737  0.04682037  0.04976432  0.0298187   0.04422369  0.13318322
      0.45318233]



```python
trees=range(25)
accuracy=np.zeros(25)
```


```python
for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
```


```python
plt.plot(trees, accuracy)
```




    [<matplotlib.lines.Line2D at 0x1664c8940>]




```python

```
