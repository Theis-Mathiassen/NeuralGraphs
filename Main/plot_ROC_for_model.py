# Constants
EPOCHS = 150
LEARNING_RATE = 0.01
MANUAL_SEED = 12345
HIDDEN_NODE_COUNT = 64
DATASPLIT = 150
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 64


from sklearn.model_selection import ParameterGrid
from heatmap import heatMap
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from Classes import AllData, BaseModel, GCN, EvaluationMetricsData
from Train_Test import train, test
from Grid_Search import grid_search
from Bayesian_Search import bayesian_search
from Plot_func import HeatMap, GridBayesHist, GridBayesianComparison,HyperParamSearchPlot, plotROCAUC, MultiPlotter
from ReadCSV import GetHeatData,GetHistData,GetParamData
from Search_Model import search_model
from Train_Test import test_threshold

from tqdm import trange







# importing MUTAG
dataset = TUDataset(root='dataset/Mutag', name='MUTAG')

data_details = {
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes,
    "num_node_labels": dataset.num_node_labels,
    "num_edge_labels": dataset.num_edge_labels,
}

print(data_details)

#  checks if a GPU is available for use w/ pytorch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
print(device)

# Set seed: manual
torch.manual_seed(MANUAL_SEED)
dataset = dataset.shuffle()
# Allocate data for training and remainder for testing 
train_dataset = dataset[:DATASPLIT]
test_dataset = dataset[DATASPLIT:]

#parameter grid - set of hyper parameters and values for grid_search to iterate over
#0,737637211	120	0,095132456	18	187	1	SGD	sigmoid	sum


csv_data = []
with open('Main/data.csv') as f:
    lines = f.readlines()
    csv_data = lines[10000].split(';') #re.findall(r'\d+\.?\d*', lines[1])


param_grid = {
    'dropout_rate': [float(csv_data[0])],
    'hidden_channels': [int(csv_data[1])],
    'learning_rate': [float(csv_data[2])],
    'batch_size' : [int(csv_data[3])],
    'epochs' : [int(csv_data[4])],
    'amount_of_layers' : [int(csv_data[5])],
    'optimizer' : [csv_data[6]],        #String key   'SGD', 'adam', 'RMSprop'
    'activation_function' : [csv_data[7]], #'Relu', 'sigmoid', 'tanh'
    'pooling_algorithm' : [csv_data[8]]  #'mean', 'sum', 'max'
}

print(param_grid)

for params in ParameterGrid(param_grid):
    result, model = search_model(params,train_dataset,test_dataset, device)
    #MultiPlotter([result], [params], "Hello")

    eval_data = EvaluationMetricsData(result)

    print(eval_data)

    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(eval_data.fprs, eval_data.tprs, marker='.', label='Logistic')

    gmeans = []
    for i in range(len(eval_data.tprs)):
        gmeans.append(math.sqrt(eval_data.tprs[i] * (1-eval_data.fprs[i])))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (eval_data.thresholds[ix], gmeans[ix]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"], shuffle=False)
    
    threshold_test_result = test_threshold(model, test_loader, device=device, threshold=eval_data.thresholds[ix])

    print(f'Accuracy: {threshold_test_result.test_accuracy}')

    plt.scatter(eval_data.fprs[ix], eval_data.tprs[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()
