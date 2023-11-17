import csv 
import pandas as pd

# reads the data from csv file into dataframe
df = pd.read_csv('results/32neurons001lr.csv')

# delimits the dataframe to the roc column
df = df[['roc']]


# print(df.roc[0]) #0.7000000000001
# print(type(df.roc[0])) # numpy.float64




