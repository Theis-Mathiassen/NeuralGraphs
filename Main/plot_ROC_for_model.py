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








#  checks if a GPU is available for use w/ pytorch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
print(device)



# importing MUTAG
dataset = TUDataset(root='dataset/Mutag', name='MUTAG')
# Set seed: manual
torch.manual_seed(MANUAL_SEED)
dataset = dataset.shuffle()
# Allocate data for training and remainder for testing 
train_dataset = dataset[:DATASPLIT]
test_dataset = dataset[DATASPLIT:]

param_strings = [
'0.25	128	0.01	150	50	3	SGD	relu	mean'.split('\t'),
'0.75	64	0.01	32	50	2	rmsprop	sigmoid	mean'.split('\t'),
'0.25	128	0.1	16	50	2	adam	sigmoid	sum'.split('\t'),
'0.25	32	0.001	150	50	9	SGD	relu	mean'.split('\t'),
'0.75	64	0.1	16	200	9	RMSprop	relu	mean'.split('\t'),
'0.75	64	0.1	16	100	1	RMSprop	relu	mean'.split('\t'),
'0.5	64	0.1	32	10	2	RMSprop	sigmoid	sum'.split('\t'),
'0.75	5	0.01	16	100	9	adam	tanh	sum'.split('\t'),
'0.75	32	0.01	150	10	9	adam	relu	mean'.split('\t'),
'0.25	32	0.01	16	200	2	rmsprop	tanh	sum'.split('\t'),
'0.5	32	0.01	16	200	3	rmsprop	tanh	sum'.split('\t'),
'0.75	32	0.01	16	200	3	rmsprop	tanh	sum'.split('\t'),
'0.5	32	0.01	32	200	9	rmsprop	tanh	sum'.split('\t'),
'0.75	128	0.1	16	200	1	rmsprop	tanh	sum'.split('\t'),
'0.75	32	0.1	150	10	1	adam	tanh	mean'.split('\t'),
'0.25	128	0.1	150	10	2	rmsprop	tanh	mean'.split('\t'),
'0.25	128	0.01	16	10	3	adam	sigmoid	sum'.split('\t'),
'0.25	128	0.01	150	100	1	SGD	tanh	mean'.split('\t'),
'0.75	128	0.01	150	100	1	SGD	tanh	mean'.split('\t'),
'0.75	5	0.01	16	200	2	SGD	relu	sum'.split('\t'),
'0.5	5	0.1	64	50	1	SGD	sigmoid	sum'.split('\t'),
'0.25	5	0.01	16	100	2	SGD	relu	sum'.split('\t'),
'0.25	32	0.1	16	200	3	adam	tanh	sum'.split('\t'),
'0.5	128	0.01	150	100	9	adam	relu	sum'.split('\t'),
'0.25	32	0.01	150	100	9	adam	relu	sum'.split('\t'),
'0.75	32	0.01	150	10	2	SGD	tanh	sum'.split('\t'),
'0.75	64	0.01	16	10	9	SGD	tanh	sum'.split('\t'),
                 ]
number = 0


for param_string in param_strings:

    #Choose 1 of the following two parameter grids
    param_grid = {
        'dropout_rate': [float(param_string[0])],
        'hidden_channels': [int(param_string[1])],
        'learning_rate': [float(param_string[2])],
        'batch_size' : [int(param_string[3])],
        'epochs' : [int(param_string[4])],
        'amount_of_layers' : [int(param_string[5])],
        'optimizer' : [param_string[6]],        #String key   'SGD', 'adam', 'RMSprop'
        'activation_function' : [param_string[7]], #'Relu', 'sigmoid', 'tanh'
        'pooling_algorithm' : [param_string[8]]  #'mean', 'sum', 'max'
    }



    for params in ParameterGrid(param_grid):
        result, model = search_model(params,train_dataset,test_dataset, device)

        eval_data = EvaluationMetricsData(result)
        
        plt.figure(figsize=(10,10))
        plt.plot([0,1], [0,1], linestyle='--', label='Random')
        plt.plot(eval_data.fprs, eval_data.tprs, marker='.', label='ROC')

        invert = True if eval_data.roc < 0.5 else False

        gmeans = []
        for i in range(len(eval_data.tprs)):
            gmeans.append(math.sqrt((eval_data.tprs[i])**2 + (1-eval_data.fprs[i])**2))
        
        ix = np.argmax(gmeans)
        
        test_loader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"], shuffle=False)
        #all_data_loader = DataLoader(dataset=dataset, batch_size=params["batch_size"], shuffle=False)
        
        threshold_test_result = test_threshold(model, test_loader, device=device, threshold=eval_data.thresholds[ix], invert=invert)

        
        plt.scatter(eval_data.fprs[ix], eval_data.tprs[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate', fontsize=26)
        plt.ylabel('True Positive Rate', fontsize=26)
        plt.legend(fontsize=26)
        plt.title('AUROC: %.3f' % eval_data.roc, fontsize=26)
        # show the plot
        #plt.show()
        plt.savefig(f'results/figures/roc_{eval_data.roc}_{number}.png', format='png', dpi=400)
        plt.close()
        print(eval_data.roc)
        with open(f'results/figures/roc_{eval_data.roc}_{number}.txt', 'w') as f:
            f.write(str(params) + "\n")
            f.write(str(eval_data) + "\n")
            f.write(f'TPR: {eval_data.tprs}, FPR: {eval_data.fprs}, Thresholds: {eval_data.thresholds}\n')
            f.write('Best Threshold=%f, G-Mean=%.3f\n' % (eval_data.thresholds[ix], gmeans[ix]))
            f.write(f'Accuracy, using new threshold: {threshold_test_result.test_accuracy}\n')
        number += 1

