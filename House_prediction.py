#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import module
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data
df = pd.read_csv('../dataset/kc_house_data.csv')
df.head()

df.plot()

#to check if data have null or not
np.where(pd.isnull(df))

#drop column
column_to_drop = ['id','date','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
df.drop(column_to_drop,axis=1,inplace=True)
df.head()

#split independet variabel(x) and variabel dependen(y)
X= df.drop('price',axis=1)
y=df['price']
X.head()
y.head()

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#train model
model.predict(X_test)

#model score
model.score(X,y)

X.head()

#make new data for prediction(in this case i use my home data)
data_new={'bedrooms':2.0,'bathrooms':1.00,'sqft_living':800,'sqft_lot':4500,'floors':1.0,'waterfront':0,'view':0,'condition':7,
          'grade':7,'sqft_above':0,'sqft_basement':0,'yr_built':2016}
index=[1]
my_data=pd.DataFrame(data_new,index)

print(my_data)

#make prediction
my_data_price=model.predict(my_data)
rounded_price = np.round(my_data_price,2)
print(f" The predicted price for the given data is :{rounded_price}")

print(f" The predicted price for the given data is :{my_data_price}")

