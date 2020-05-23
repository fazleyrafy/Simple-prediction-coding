# Simple-prediction-coding

#importing all the necessary modules

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

#a list of prices of some areas to analyze in dataframe format using pandas package

df = pd.DataFrame({
    'area':[2600, 3000, 3200, 3600, 4000],
    'price':[550000,565000,610000,680000,725000]
})

#plot to visualize the relation

plt.scatter(df.area, df.price)

#since a linear relation is conspicuous, lets call the linear regression model to make a model for prediction

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

#this model now follows the expression, "predicted_price = prediction_coefficient * area_of_which_prediction_will_made + intercept or #expected mean value of listed prices"

reg.coef_
reg.intercept_

#using the resulted values of intercept and coef, predicted price can be shown with the above-mentioned expression or just by,

reg.predict([[3300]]) # assuming the area is 3300, whose price has to be predicted.

#this is the least recommended prediction model, though a very good one to get hold of other advanced and efficace models

