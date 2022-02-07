#!/usr/bin/env python
# coding: utf-8

# In[81]:


# Q.01: Combine the global CO2 data and temprature data using pandas framework? Data is attached with the assignment?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

# read csv files:
dfGlobalCO2Data = pd.read_csv('MUHAMMAD FAYYAZ - global_co2.csv')
dfAnnualTempData = pd.read_csv('MUHAMMAD FAYYAZ - annual_temp.csv')

# merge data files:
# dfMergedData = pd.merge(dfGlobalCO2Data, dfAnnualTempData, on='Year', how='inner')
# print(dfMergedData)
dfGlobalCO2Data = dfGlobalCO2Data[dfGlobalCO2Data.Year >= 1950]
dfGlobalCO2Data = dfGlobalCO2Data.iloc[:,0:2]
# print(dfGlobalCO2Data)
dfAnnualTempData = dfAnnualTempData[dfAnnualTempData.Source == 'GCAG']
dfAnnualTempData = dfAnnualTempData[dfAnnualTempData.Year >= 1950]
dfAnnualTempData = dfAnnualTempData[dfAnnualTempData.Year <= 2010]
dfAnnualTempData.drop('Source',axis = 1,inplace=True)
dfAnnualTempData = dfAnnualTempData.reindex(index=dfAnnualTempData.index[::-1])
dfMergedData = pd.merge(dfGlobalCO2Data, dfAnnualTempData, on="Year")
print(dfMergedData)


# In[82]:


# Q.02: Utilize the Combined data file to create regression model for temprature and CO2 prediction?
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = dfMergedData.values[:,0]                                           
Y = dfMergedData.values[:,1:3]        

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size= 0.1) 
print("The first five sample")
print(train_x[:5])
print("The first five targets {}".format(train_y[:5]))
print("The number of samples in train set is {}".format(train_x.shape[0]))
print("The number of samples in test set is {}".format(test_x.shape[0]))
lr = LinearRegression()
lr.fit(train_x.reshape(-1,1),train_y) 


# In[83]:


# Q.03: Plot a 3D graph to show the model with referene to the data?
x_line = np.arange(1950,2010).reshape(-1,1)                
p = lr.predict(x_line).T 
fig1 = plt.figure()
fig1.set_size_inches(16, 9)
ax = fig1.add_subplot(111, projection='3d')
# plot the point on the figure
ax.scatter(xs=dfMergedData['Year'], ys=dfMergedData['Mean'], zs=dfMergedData['Total'])
ax.set_ylabel("Relative tempature");
ax.set_xlabel('Year');
ax.set_zlabel("Total Annual CO2 Emissions")


# In[84]:


# Q.04: Predict estimates from 2020 to 2050 based on the regression model?

x2_line = np.arange(1950,2050).reshape(-1,1)
p2 = lr.predict(x2_line).T

fig2 = plt.figure()
fig2.set_size_inches(16, 12)

plt.subplot(2, 1, 1)
plt.plot(dfMergedData['Year'],dfMergedData['Total'],label='Given')
plt.plot(x2_line,p2[0], label='Predicted')
plt.ylabel('Total Annual CO2 Emissions')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(dfMergedData['Year'],dfMergedData['Mean'],label='Given')
plt.plot(x2_line,p2[1], label='Predicted')
plt.xlabel('Year')
plt.ylabel("Relative temperature change")
plt.legend()
plt.show()


# In[ ]:




