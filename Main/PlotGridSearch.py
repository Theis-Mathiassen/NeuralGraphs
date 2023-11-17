
import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd

# reads the data from csv file into dataframe (df)
all_data = pd.read_csv('../results/Combined_Grid.csv')

# delimits the df to the roc column
ROC_df = all_data[['roc']]

# makes df containing specific optimizers
sgd_filter = all_data[all_data['optimizer'] == "SGD"]
adam_filter = all_data[all_data['optimizer'] == "adam"]
rmsprop_filter = all_data[all_data['optimizer'] == "RMSprop"]

# leaves only the roc column of each optimizer
sgd_ROC = sgd_filter[['roc']]
adam_ROC = adam_filter[['roc']]
rmsprop_ROC = rmsprop_filter[['roc']]

print(sgd_ROC[0:4])
print(adam_ROC[0:4])
print(rmsprop_ROC[0:4])

# print(df.roc[0]) #0.7000000000001
# print(type(df.roc[0])) # numpy.float64

# selects all rows with roc == 0.0
test_for_null = all_data.loc[all_data['roc'] == 0.0]

# creates csv with all rows that resulted in an ROC value = 0.0
test_for_null.to_csv('Main/test_for_null.csv', index=False)









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


def plot_auc_count(datasets, labels, title):
    # Plotting
    for i, dataset in enumerate(datasets):
        counts_in_ranges, value_ranges = create_ranges(dataset)
        plt.plot(value_ranges[:-1], counts_in_ranges, linestyle='-', label="{}".format(labels[i]))
        plt.legend(loc='upper left')


    plt.title('{}.'.format(title))
    plt.xlabel('AUROC')
    plt.ylabel('Number of models')

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.xticks(tick_positions)

    plt.show()

datasets = [sgd_ROC, adam_ROC, rmsprop_ROC]
labels = ["SGD", "ADAM", "RMSProp"]
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for all optimizers")








