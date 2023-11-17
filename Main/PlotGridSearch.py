
import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd

def create_ranges(data, num_ranges=100):
    # Find the min and max
    min_val = np.min(data)
    max_val = np.max(data)

    # Create 100 ranges 
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)

    # #data points
    counts, _ = np.histogram(data, bins=value_ranges)

    return counts

your_dataset = np.random.rand(1000) * 100  

counts_in_ranges = create_ranges(your_dataset)

print("Number of points in each range:", counts_in_ranges)


def create_ranges(data, num_ranges=100):
    min_val = np.min(data)
    max_val = np.max(data)
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)
    counts, _ = np.histogram(data, bins=value_ranges)
    return counts, value_ranges

# reads the data from csv file into dataframe
df = pd.read_csv('../results/32neurons001lr.csv')

# delimits the dataframe to the roc column
df = df[['roc']]

def plot_auc_count(datasets, labels, title):
    # Plotting
    for i, dataset in enumerate(datasets):
        counts_in_ranges, value_ranges = create_ranges(dataset)
        plt.plot(value_ranges[:-1], counts_in_ranges, linestyle='-', label=f'{labels[i]}')


    plt.title('{}.'.format(title))
    plt.xlabel('AUROC')
    plt.ylabel('Number of models')

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.xticks(tick_positions)

    plt.show()




# print(df.roc[0]) #0.7000000000001
# print(type(df.roc[0])) # numpy.float64




