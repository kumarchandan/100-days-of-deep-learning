import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from grad_descent import train

df = pd.read_csv('https://raw.githubusercontent.com/instituteofai/ML-101/master/Datasets/Advertising_data.csv', index_col=0)

df.head(3)
'''
TV	radio	newspaper	sales
1	230.1	37.8	69.2	22.1
2	44.5	39.3	45.1	10.4
3	17.2	45.9	69.3	9.3


target - sales
features - tv, radio, newspaper

'''

cols = ['radio', 'sales']

# Train model with only feature 'radio'
df_data = df[cols].values

'''
df_data[:3]
array([[37.8, 22.1],
       [39.3, 10.4],
       [45.9,  9.3]])
'''
w, b = train(spendings=df_data[:, 0], sales=df_data[:, 1], w=0.0, b=0.0, alpha=0.001, epochs=15000)

x_new = 23.0
y_new = predict(x=x_new, w=w, b=b)
print(y_new)
