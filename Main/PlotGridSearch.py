
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


# ---------------------------------------------------------------- #
# Plotting of all hyperparameters
# ---------------------------------------------------------------- #

"""

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

# Plot for batch size
hyperparameters = [16, 32, 64, 150]
datasets= filter_data_from_csv("batch_size", hyperparameters)
print(datasets)
labels = ["16", "32", "64", "150"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for BS")


# Plot for learning rate
hyperparameters = [0.1, 0.01, 0.001]
datasets= filter_data_from_csv("learning_rate", hyperparameters)
print(datasets)
labels = ["0.1", "0.01", "0.001"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for LR")

# Plot for number of neurons
hyperparameters = [5, 32, 64, 128]
datasets= filter_data_from_csv("hidden_channels", hyperparameters)
print(datasets)
labels = ["5", "32", "64", "128"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for hidden channels")

# Plot for dropout rate
hyperparameters = [0.25, 0.5, 0.75]
datasets= filter_data_from_csv("dropout_rate", hyperparameters)
print(datasets)
labels = ["0.25", "0.5", "0.75"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for dropout rate")

# Plot for number of epochs
hyperparameters = [10, 50, 100, 200]
datasets= filter_data_from_csv("epochs", hyperparameters)
print(datasets)
labels = ["10", "50", "100", "200"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for epochs")

# Plot for number of activation function
hyperparameters = ["relu", "sigmoid", "tanh"]
datasets= filter_data_from_csv("activation_function", hyperparameters)
print(datasets)
labels = ["ReLU", "Sigmoid", "Tanh"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given ROC score within a range for AF")

"""





# Multiplot


def plot_auc_count2(ax, datasets, labels, title):
    # Plotting
    for i, dataset in enumerate(datasets):
        counts_in_ranges, value_ranges = create_ranges(dataset)
        ax.plot(value_ranges[:-1], counts_in_ranges, linestyle='-', label="{}".format(labels[i]))
        ax.legend(loc='upper left')

    ax.set_title('{}.'.format(title))
    ax.set_xlabel('AUROC')
    ax.set_ylabel('Number of models')

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(tick_positions)

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.suptitle("Distribution of AUROC Scores for Different Hyperparameters", fontsize=16)


# Plotting for optimizers
hyperparameters = ["SGD", "adam", "RMSprop"]
datasets = filter_data_from_csv("optimizer", hyperparameters)
labels = ["SGD", "ADAM", "RMSProp"]
plot_auc_count2(axs[0, 0], datasets, labels, "Optimizers")

# Plotting for # of gcn layers
hyperparameters = [1, 2, 3, 9]
datasets = filter_data_from_csv("amount_of_layers", hyperparameters)
labels = ["1", "2", "3", "9"]
plot_auc_count2(axs[0, 1], datasets, labels, "GCN Layers")

# Plotting for pooling algorithms
hyperparameters = ["mean", "sum"]
datasets = filter_data_from_csv("pooling_algorithm", hyperparameters)
labels = ["Mean", "Sum"]
plot_auc_count2(axs[0, 2], datasets, labels, "Pooling Algorithms")

# Plotting for batch size
hyperparameters = [16, 32, 64, 150]
datasets = filter_data_from_csv("batch_size", hyperparameters)
labels = ["16", "32", "64", "150"]
plot_auc_count2(axs[1, 0], datasets, labels, "Batch Size")

# Plotting for learning rate
hyperparameters = [0.1, 0.01, 0.001]
datasets = filter_data_from_csv("learning_rate", hyperparameters)
labels = ["0.1", "0.01", "0.001"]
plot_auc_count2(axs[1, 1], datasets, labels, "Learning Rate")

# Plotting for number of neurons
hyperparameters = [5, 32, 64, 128]
datasets = filter_data_from_csv("hidden_channels", hyperparameters)
labels = ["5", "32", "64", "128"]
plot_auc_count2(axs[1, 2], datasets, labels, "Hidden Channels")

# Plotting for dropout rate
hyperparameters = [0.25, 0.5, 0.75]
datasets = filter_data_from_csv("dropout_rate", hyperparameters)
labels = ["0.25", "0.5", "0.75"]
plot_auc_count2(axs[2, 0], datasets, labels, "Dropout Rate")

# Plotting for number of epochs
hyperparameters = [10, 50, 100, 200]
datasets = filter_data_from_csv("epochs", hyperparameters)
labels = ["10", "50", "100", "200"]
plot_auc_count2(axs[2, 1], datasets, labels, "Epochs")

# Plotting for number of activation function
hyperparameters = ["relu", "sigmoid", "tanh"]
datasets = filter_data_from_csv("activation_function", hyperparameters)
labels = ["ReLU", "Sigmoid", "Tanh"]
plot_auc_count2(axs[2, 2], datasets, labels, "Activation Functions")


# Adjust layout
#plt.tight_layout()
plt.show()