
import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd

# reads the data from csv file into dataframe (df)
all_data = pd.read_csv("results/Grid_LastLastFr.csv")

NUM_RANGES = 50
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





def create_ranges(data, num_ranges=100):
    min_val = np.min(data)
    max_val = np.max(data)
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)
    counts, _ = np.histogram(data, bins=value_ranges)
    return counts, value_ranges


def plot_auc_count(datasets, labels, title):
    # Plotting
    colors = [ "black", "springgreen", "red", "blue"]
    n = 4-len(datasets)
    c = colors[n:]
    counts = np.ndarray(shape = (len(datasets), NUM_RANGES))
    bins = np.ndarray(shape = (len(datasets), NUM_RANGES+1))
    also_bins = np.ndarray(shape = (len(datasets), NUM_RANGES))
    X = np.linspace(0, 1, NUM_RANGES)
    plt.figure(figsize = (12, 8))
    for i, dataset in enumerate(datasets):
        counts[i], bins[i] = np.histogram(dataset, bins=NUM_RANGES)
    for i in range(0, len(bins)):
        for j in range(0, len(bins[i])-1): 
            also_bins[i][j] = bins[i][j] 

    for i in range(0, len(also_bins)):
        plt.hist(also_bins[i], X, weights = counts[i], fill = True, label=labels[i], color=c[i], alpha = 0.2, edgecolor = c[i])
    plt.legend(loc='upper left')


    plt.title('{}.'.format(title))
    plt.xlabel('AUC-ROC')
    plt.ylabel('Number of configurations')

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.xticks(tick_positions)
    #plt.grid()

    #plt.show()

    plt.savefig("{}.".format(title))
# ---------------------------------------------------------------- #
# Plotting of all hyperparameters
# ---------------------------------------------------------------- #



# Plot for optimizers
hyperparameters = ["SGD", "adam", "rmsprop"]
datasets= filter_data_from_csv("optimizer", hyperparameters)
print(datasets)
labels = ["SGD", "ADAM", "RMSProp"]
plot_auc_count(datasets, labels, "Distribution of AUC-ROC Scores for Optimizers")

# Plot for # of gcn layers
hyperparameters = [1, 2, 3, 9]
datasets= filter_data_from_csv("amount_of_layers", hyperparameters)
print(datasets)
labels = ["1", "2", "3", "9"]
plot_auc_count(datasets, labels, "Distribution of AUC-ROC Scores for GCN Layers")

# Plot for pooling algorithms
hyperparameters = ["mean", "sum"]
datasets= filter_data_from_csv("pooling_algorithm", hyperparameters)
print(datasets)
labels = ["Mean", "Sum"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Distribution of AUC-ROC Scores for Pooling Algorithms")

# Plot for batch size
hyperparameters = [16, 32, 64, 150]
datasets= filter_data_from_csv("batch_size", hyperparameters)
print(datasets)
labels = ["16", "32", "64", "150"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given AUC-ROC score within a range for BS")


# Plot for learning rate
hyperparameters = [0.1, 0.01, 0.001]
datasets= filter_data_from_csv("learning_rate", hyperparameters)
print(datasets)
labels = ["0.1", "0.01", "0.001"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Distribution of AUC-ROC Scores for Learning Rate")

# Plot for number of neurons
hyperparameters = [5, 32, 64, 128]
datasets= filter_data_from_csv("hidden_channels", hyperparameters)
print(datasets)
labels = ["5", "32", "64", "128"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given AUC-ROC score within a range for hidden channels")

# Plot for dropout rate
hyperparameters = [0.25, 0.5, 0.75]
datasets= filter_data_from_csv("dropout_rate", hyperparameters)
print(datasets)
labels = ["0.25", "0.5", "0.75"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given AUC-ROC score within a range for dropout rate")

# Plot for number of epochs
hyperparameters = [10, 50, 100, 200]
datasets= filter_data_from_csv("epochs", hyperparameters)
print(datasets)
labels = ["10", "50", "100", "200"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Occurences of a given AUC-ROC score within a range for epochs")

# Plot for number of activation function
hyperparameters = ["relu", "sigmoid", "tanh"]
datasets= filter_data_from_csv("activation_function", hyperparameters)
print(datasets)
labels = ["ReLU", "Sigmoid", "Tanh"] # Should be same order as hyperparameters.
plot_auc_count(datasets, labels, "Distribution of AUC-ROC Scores for Activation Functions")




##################

plt.show()




import numpy as np
import matplotlib.pyplot as plt

NUM_RANGES = 50

def filter_data_from_csv(hyperparameter_category, hyperparameters_within_category):
    rocs = []
    for hyperparameter in hyperparameters_within_category:
        filter = all_data[all_data[hyperparameter_category] == hyperparameter]
        rocs.append(filter[['roc']])
    return rocs

def create_ranges(data, num_ranges=100):
    min_val = np.min(data)
    max_val = np.max(data)
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)
    counts, _ = np.histogram(data, bins=value_ranges)
    return counts, value_ranges

def plot_auc_count(datasets, labels, title, ax, num_ranges=50):
    colors = ["black", "springgreen", "red", "blue"]
    n = 4 - len(datasets)
    c = colors[n:]
    counts = np.ndarray(shape=(len(datasets), num_ranges))
    bins = np.ndarray(shape=(len(datasets), num_ranges + 1))
    also_bins = np.ndarray(shape=(len(datasets), num_ranges))
    X = np.linspace(0, 1, num_ranges)

    for i, dataset in enumerate(datasets):
        counts[i], bins[i] = np.histogram(dataset, bins=num_ranges)

    for i in range(0, len(bins)):
        for j in range(0, len(bins[i]) - 1):
            also_bins[i][j] = bins[i][j]

    for i in range(0, len(also_bins)):
        ax.hist(also_bins[i], X, weights=counts[i], fill=True, label=labels[i], color=c[i], alpha=0.2, edgecolor=c[i])

    ax.legend(loc='upper left')
    ax.set_title('{}.'.format(title))
    ax.set_xlabel('AUC-ROC')
    ax.set_ylabel('Number of configurations')

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(tick_positions)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.8, wspace=0.8)  # Adjust the spacing between subplots


#fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Plotting for each hyperparameter
hyperparameters_list = [
    ["optimizer", ["SGD", "adam", "rmsprop"]],
    ["amount_of_layers", [1, 2, 3, 9]],
    ["pooling_algorithm", ["mean", "sum"]],
    ["batch_size", [16, 32, 64, 150]],
    ["learning_rate", [0.1, 0.01, 0.001]],
    ["hidden_channels", [5, 32, 64, 128]],
    ["dropout_rate", [0.25, 0.5, 0.75]],
    ["epochs", [10, 50, 100, 200]],
    ["activation_function", ["relu", "sigmoid", "tanh"]]
]

titles = [
    "Optimizers",
    "GCN Layers",
    "Pooling Algorithms",
    "Batch Size",
    "Learning Rate",
    "Hidden Channels",
    "Dropout Rate",
    "Epochs",
    "Activation Functions"
]

for i, (hyperparameter_category, hyperparameters) in enumerate(hyperparameters_list):
    row = i // 3
    col = i % 3
    datasets = filter_data_from_csv(hyperparameter_category, hyperparameters)
    plot_auc_count(datasets, hyperparameters, titles[i], axes[row, col], num_ranges=NUM_RANGES)

plt.suptitle("Distribution of AUC-ROC Scores for Different Hyperparameters", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Add a margin below the title

# Adjust layout to prevent clipping of titles

# Show the combined plot
plt.show()




import numpy as np
import matplotlib.pyplot as plt

NUM_RANGES = 50

def filter_data_from_csv(hyperparameter_category, hyperparameters_within_category):
    rocs = []
    for hyperparameter in hyperparameters_within_category:
        filter = all_data[all_data[hyperparameter_category] == hyperparameter]
        rocs.append(filter[['roc']])
    return rocs

def create_ranges(data, num_ranges=100):
    min_val = np.min(data)
    max_val = np.max(data)
    value_ranges = np.linspace(min_val, max_val, num_ranges + 1)
    counts, _ = np.histogram(data, bins=value_ranges)
    return counts, value_ranges

def plot_auc_count(datasets, labels, title, ax, num_ranges=50):
    colors = ["black", "springgreen", "red", "blue"]
    n = 4 - len(datasets)
    c = colors[n:]
    bins = np.linspace(0, 1, num_ranges + 1)  # Use the same bin edges for all datasets
    X = np.linspace(0, 1, num_ranges)

    for i, dataset in enumerate(datasets):
        ax.hist(dataset, bins=bins, density=True, alpha=0.5, label=labels[i], color=c[i])

    ax.legend(loc='upper left')
    ax.set_title('{}.'.format(title))
    ax.set_xlabel('AUC-ROC')
    ax.set_ylabel('Density')

    tick_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(tick_positions)

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Plotting for each hyperparameter
hyperparameters_list = [
    ["optimizer", ["SGD", "adam", "rmsprop"]],
    ["amount_of_layers", [1, 2, 3, 9]],
    ["pooling_algorithm", ["mean", "sum"]],
    ["batch_size", [16, 32, 64, 150]],
    ["learning_rate", [0.1, 0.01, 0.001]],
    ["hidden_channels", [5, 32, 64, 128]],
    ["dropout_rate", [0.25, 0.5, 0.75]],
    ["epochs", [10, 50, 100, 200]],
    ["activation_function", ["relu", "sigmoid", "tanh"]]
]

titles = [
    "Distribution of AUC-ROC Scores for Optimizers",
    "Distribution of AUC-ROC Scores for GCN Layers",
    "Distribution of AUC-ROC Scores for Pooling Algorithms",
    "Occurences of a given AUC-ROC score within a range for BS",
    "Distribution of AUC-ROC Scores for Learning Rate",
    "Occurences of a given AUC-ROC score within a range for hidden channels",
    "Occurences of a given AUC-ROC score within a range for dropout rate",
    "Occurences of a given AUC-ROC score within a range for epochs",
    "Distribution of AUC-ROC Scores for Activation Functions"
]

for i, (hyperparameter_category, hyperparameters) in enumerate(hyperparameters_list):
    row = i // 3
    col = i % 3
    datasets = filter_data_from_csv(hyperparameter_category, hyperparameters)
    plot_auc_count(datasets, hyperparameters, titles[i], axes[row, col], num_ranges=NUM_RANGES)

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the combined plot
plt.show()
