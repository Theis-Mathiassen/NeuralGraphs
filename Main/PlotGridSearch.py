
import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd

# reads the data from csv file into dataframe (df)
all_data = pd.read_csv("results/Combined_Grid.csv")

# delimits the df to the roc column
ROC_df = all_data[['roc']]

def filter_data_from_csv(hyperparameter_category, hyperparameters_within_category):
    ROC_df = all_data[['roc']]
    # makes df containing specific optimizers
    filters = []
    rocs = []
    for i, hyperparameter in enumerate(hyperparameters_within_category):
        filter = all_data[all_data[hyperparameter_category] == hyperparameter]
        rocs.append(filter[['roc']])
 

    return rocs

#print(sgd_ROC[0:4])
#print(adam_ROC[0:4])
#print(rmsprop_ROC[0:4])

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

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.xticks(tick_positions)

    plt.show()


# Plot for optimizers
hyperparameters = ["SGD", "adam", "RMSprop"]
datasets= filter_data_from_csv("optimizer", hyperparameters)
print(datasets)
labels = ["SGD", "ADAM", "RMSProp"]
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for all optimizers")

# Plot for # of gcn layers
hyperparameters = [1, 2, 3, 9]
datasets= filter_data_from_csv("amount_of_layers", hyperparameters)
print(datasets)
labels = ["1", "2", "3", "9"]
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for all # layers")

# Plot for pooling algorithms
hyperparameters = ["mean", "sum"]
datasets= filter_data_from_csv("pooling_algorithm", hyperparameters)
print(datasets)
labels = ["Mean", "Sum"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for pooling algorithms")






