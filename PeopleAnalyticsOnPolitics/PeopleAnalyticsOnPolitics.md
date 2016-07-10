

```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
```


```python
"""
Data Management
"""
data = pd.read_csv("/Users/Rohit/Desktop/outlook.csv")

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

# Data Management
data_clean = data.dropna()
```


```python
#W1_A2: How much have you thought about the upcoming election for president?
#W1_A10: How often, if ever, do you discuss politics with your family or friends?
#W1_A11: How many days in the past week did you watch national news programs on television or on the Internet?
#W1_A12: Do you approve or disapprove of the way Barack Obama is handling his job as President?
#W1_B2: How much can people like you affect what the government does?

cluster=data_clean[['W1_A2', 'W1_A10', 'W1_A11', 'W1_A12', 'W1_B1']]
cluster.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>W1_A2</th>
      <th>W1_A10</th>
      <th>W1_A11</th>
      <th>W1_A12</th>
      <th>W1_B1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2292.000000</td>
      <td>2292.000000</td>
      <td>2292.000000</td>
      <td>2292.000000</td>
      <td>2292.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.130017</td>
      <td>2.678010</td>
      <td>4.192845</td>
      <td>1.239965</td>
      <td>3.599913</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.210902</td>
      <td>1.227214</td>
      <td>2.624862</td>
      <td>0.578239</td>
      <td>1.143200</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
#clustervar['W1_A1']=preprocessing.scale(clustervar['W1_A1'].astype('float64'))
clustervar['W1_A2']=preprocessing.scale(clustervar['W1_A2'].astype('float64'))
clustervar['W1_A10']=preprocessing.scale(clustervar['W1_A10'].astype('float64'))
clustervar['W1_A11']=preprocessing.scale(clustervar['W1_A11'].astype('float64'))
clustervar['W1_A12']=preprocessing.scale(clustervar['W1_A12'].astype('float64'))
clustervar['W1_B1']=preprocessing.scale(clustervar['W1_B1'].astype('float64'))
```


```python
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

```


```python
"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
%matplotlib inline
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
```




    <matplotlib.text.Text at 0x11b151c88>




![png](output_5_1.png)



```python
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()
```


![png](output_6_0.png)



```python
"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()
```




    2    618
    1    513
    0    473
    Name: cluster, dtype: int64




```python
# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
```

    Clustering variable means by cluster
                   index     W1_A2    W1_A10    W1_A11    W1_A12     W1_B1
    cluster                                                               
    0        1224.693446  1.151703 -0.819682 -0.804982  0.250493  0.547970
    1        1142.483431 -0.195955 -0.301577 -0.339406 -0.560074 -0.707371
    2        1095.655340 -0.710197  0.830843  0.847091  0.209088  0.132024



```python
# validate clusters in training data by examining cluster differences in W1_A1 using ANOVA
# first have to merge W1_A1 with clustering variables and cluster assignment data 
gpa_data=data_clean['W1_A1']
# split GPA data into train and test sets
gpa_train, gpa_test = train_test_split(gpa_data, test_size=.3, random_state=123)
gpa_train1=pd.DataFrame(gpa_train)
gpa_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(gpa_train1, merged_train, on='index')
sub1 = merged_train_all[['W1_A1', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

gpamod = smf.ols(formula='W1_A1 ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  W1_A1   R-squared:                       0.398
    Model:                            OLS   Adj. R-squared:                  0.397
    Method:                 Least Squares   F-statistic:                     529.7
    Date:                Sun, 10 Jul 2016   Prob (F-statistic):          2.80e-177
    Time:                        11:01:20   Log-Likelihood:                -2182.8
    No. Observations:                1604   AIC:                             4372.
    Df Residuals:                    1601   BIC:                             4388.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [95.0% Conf. Int.]
    -----------------------------------------------------------------------------------
    Intercept           3.7421      0.043     86.171      0.000         3.657     3.827
    C(cluster)[T.1]    -1.0696      0.060    -17.765      0.000        -1.188    -0.951
    C(cluster)[T.2]    -1.8780      0.058    -32.548      0.000        -1.991    -1.765
    ==============================================================================
    Omnibus:                       65.122   Durbin-Watson:                   1.976
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              143.973
    Skew:                          -0.237   Prob(JB):                     5.45e-32
    Kurtosis:                       4.389   Cond. No.                         3.95
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
print ('means for W1_A1 by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)
```

    means for W1_A1 by cluster
                W1_A1
    cluster          
    0        3.742072
    1        2.672515
    2        1.864078



```python
print ('standard deviations for W1_A1 by cluster')
m2= sub1.groupby('cluster').std()
print (m2)
```

    standard deviations for W1_A1 by cluster
                W1_A1
    cluster          
    0        1.011096
    1        1.020285
    2        0.817735



```python
mc1 = multi.MultiComparison(sub1['W1_A1'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
```

    Multiple Comparison of Means - Tukey HSD,FWER=0.05
    =============================================
    group1 group2 meandiff  lower   upper  reject
    ---------------------------------------------
      0      1    -1.0696  -1.2108 -0.9283  True 
      0      2     -1.878  -2.0134 -1.7426  True 
      1      2    -0.8084  -0.9408 -0.6761  True 
    ---------------------------------------------



```python

```
