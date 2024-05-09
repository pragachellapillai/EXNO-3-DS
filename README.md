# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:

Read the given Data.

STEP 2:

Clean the Data Set using Data Cleaning Process.

STEP 3:

Apply Feature Encoding for the feature in the data set.

STEP 4:

Apply Feature Transformation for the feature in the data set.

STEP 5:

Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by :PRAGAHARSHITHA NC
### Reg No : 212222110033

```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/b47db3ab-74b1-4254-9530-2e611d76a76d)



```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/63d394fe-e616-459e-ba61-1381b0ce6ce5)



```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/381ba264-a72c-4478-9e67-18100e9ac94e)



```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/690a88e3-5f85-4cab-802b-493faa9ff4b6)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/46524eb1-13e4-4c8a-9055-d7cdb595b55c)



```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/8d70f6e8-537a-431b-a3fb-b065d0e3f5fc)



```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/6722ce61-78f2-480d-9d07-1f4747770d60)



```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/cd214ae5-3b7e-484d-a30d-8148dea68585)



```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/86b662e9-591f-441c-85e6-caadbfeb45be)



```py
df.skew()
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/dddce7c3-a91d-451b-a950-dc2c5eadb50e)



```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/23c38287-1986-46e7-8554-9c7fc4f542a0)



```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/db111f6a-4a53-4244-87bf-85f529ec3193)



```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/3ac02c82-90da-40d4-8056-b11081bed3eb)



```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/a49aaa91-5795-4d53-a75d-a13f27694513)



```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/1fc129d9-fb29-4fee-a377-76070bf12884)



```py
df.skew()
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/d1324c34-376e-4eff-aa6b-ac33bea906a4)



```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/4acddb0d-8ac6-4f42-81c0-ba2708e22dee)


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/PSriVarshan/EXNO-3-DS/assets/114944059/d8838866-2c73-4e6f-bb78-719723589804)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/84b838d1-4121-4dc2-ae3d-d6bab32eb588)



```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/565a06a7-9374-4137-95a6-d7f85ba208e1)




```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/1e2b5120-f717-4826-9195-2f3e76014d52)



```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/c28e56e0-a1e2-4be9-b697-66929dd290e9)



```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/1527897e-3937-4385-888f-697b06b7d6ac)



```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/KANISHKAR2607/EXNO-3-DS/assets/118886772/20f70e0b-af2d-45ac-935e-2bee0c844ba7)




## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
