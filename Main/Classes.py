#Imports
import torch
from torch.nn import Linear, ReLU, Sigmoid, Tanh
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from sklearn import metrics #Used For ROC-AUC
from collections import OrderedDict #Used for sequential input to layers
import csv
import os
import json

# constants
MANUAL_SEED = 12345

# graph convolutional network obj
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate, learning_rate, activation_function, amount_of_layers, pooling_algorithm, in_features=7, outfeatures = 2):
        super(GCN, self).__init__()
        torch.manual_seed(MANUAL_SEED)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.pooling_algorithm = pooling_algorithm
        
        # List of layers. Contains a name for the layer, the method to execute, and the function parameters
        self.layers = OrderedDict()
        
        # Input layer
        self.layers['conv1'] = (GCNConv(in_features, hidden_channels), 'x, edge_index -> x')
        
        # Hidden layers
        for i in range (amount_of_layers):
            # Pick activation function 
            if self.activation_function.lower() == "relu":
                self.layers['relu'+str(i+1)] = ReLU(inplace=True)
            elif self.activation_function.lower() == "sigmoid":
                self.layers['sigmoid'+str(i+1)] = Sigmoid()
            elif self.activation_function.lower() == "tanh":
                self.layers['tanh'+str(i+1)] = Tanh()
            else : raise Exception("Invalid activationFunction name: " + str(self.activation_function))
            
            #Add convolutional layer
            self.layers['conv'+str(i+2)] = (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x')
        
        #Add layers to the model, including function paramters 
        self.layers = Sequential('x, edge_index', self.layers)
        
        # Output layer
        self.lin = Linear(hidden_channels, outfeatures)

    # Foward propagation
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.layers(x, edge_index)

        # 2. Readout layer
        # Picks from pooling options
        if self.pooling_algorithm.lower() == 'mean' : x = global_mean_pool(x, batch)
        elif self.pooling_algorithm.lower() == 'sum' : x = global_add_pool(x, batch)
        elif self.pooling_algorithm.lower() == 'max' : x = global_max_pool(x, batch)
        else : raise Exception("Invalid pooling name: " + str(self.pooling_algorithm))
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)

        return x 

# model obj
class BaseModel():
    def __init__(self, model: GCN, loss, optim):
        self.model = model
        self.loss_function = loss
        self.optimizer = optim


# The data generated by training the model over 1 epoch
class TrainData():
    def __init__(self):
        self.train_losses = 0
        self.train_accuracies = 0
        self.train_labels = []
        self.train_scores = []
        self.train_probability_estimates = []
        
# The data generated by Testing the model once
class TestData():
    def __init__(self):
        self.test_losses = 0 
        self.test_accuracy = 0
        self.test_scores = []       #Model Guesses
        self.test_labels = []       #Truths
        self.test_probability_estimates = []

# Data covering different evaluation metrics based on input TestData
class EvaluationMetricsData():
    def __init__(self, TestData):
        self.accuracy = TestData.test_accuracy
        self.TP = 0
        self.TF = 0
        self.FP = 0
        self.FN = 0

        #Calculate True positive, true negative, false positive, false negative
        amountOfLabels = len(TestData.test_labels)

        for count in range(amountOfLabels): 
            if TestData.test_labels[count] == 1:       #If graph is true
                if TestData.test_labels[count] == TestData.test_scores[count]: 
                    self.TP += 1
                else:
                    self.FN += 1
            elif TestData.test_labels[count] == 0:     #If graph is false
                if TestData.test_labels[count] == TestData.test_scores[count]: 
                    self.TF += 1
                else:
                    self.FP += 1
                self.TP = 0
        self.TP = self.TP / amountOfLabels
        self.TF = self.TF / amountOfLabels
        self.FP = self.FP / amountOfLabels
        self.FN = self.FN / amountOfLabels


        # Check if both of them are zero. Then set result to 0
        
        if (self.TP == 0 and self.FN == 0): self.TPR = 0
        else: self.TPR = self.TP / (self.TP+self.FN)  #Sensitivity, Recall, true positive rate
        self.FPR = 1-self.TPR                   #false positive rate
        
        if (self.TP == 0 and self.FP == 0): self.PREC = 0
        else: self.PREC = self.TP / (self.TP + self.FP) #Precision
        
        if (self.PREC == 0 and self.TPR == 0) : self.f1 = 0
        else: self.f1 = 2 * self.PREC * self.TPR / (self.PREC + self.TPR)

        #AUC ROC and PR AUC
        
        #ROC fails if the TP and FP are zero
        if (self.TP == 0 and self.FP == 0): self.roc = 0
        else: self.roc = metrics.roc_auc_score(TestData.test_labels, TestData.test_probability_estimates)
        self.pr = metrics.average_precision_score(TestData.test_labels, TestData.test_scores)
                    

class StoredModel():
    def __init__(self):
        self.evalutation_metric = 0
        self.model = None
        self.params = None 
    
    #Function that updates parameters
    def update(self, evalutation_metric, model, params):
        self.evalutation_metric = evalutation_metric
        self.model = model
        self.params = params


# All data from a given set of hyperparameters.
# Data passed to plotting functions
class AllData():
    def __init__(self):
        self.train_accuracies = []
        self.train_losses = []
        self.train_labels = []
        self.train_scores = []
        self.train_probability_estimates = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_scores = []
        self.test_labels = []
        self.test_probability_estimates = []

    def insert_train_data (self, data: TrainData):
        self.train_accuracies.append(data.train_accuracies)
        self.train_losses.append(data.train_losses)
        self.train_labels.extend(data.train_labels)
        self.train_scores.extend(data.train_scores)
        self.train_probability_estimates.extend(data.train_probability_estimates)
        
    def insert_test_data (self, data: TestData):
        self.test_accuracies.append(data.test_accuracy)
        self.test_losses.append(data.test_losses)
        self.test_labels.extend(data.test_labels)
        self.test_scores.extend(data.test_scores)
        self.test_probability_estimates.extend(data.test_probability_estimates)

class CSVWriter():
    def __init__(self, name):
        # Create path if non-existent
        cwd = os.getcwd()
        path = os.path.join(cwd, 'results')
        os.makedirs(path, exist_ok=True)

        self.name = path + "/" + name + ".csv"
        
        with open(self.name, 'w') as csvfile:
            fieldnames = ['dropout_rate', 'hidden_channels', 'learning_rate', 'batch_size', 'epochs', 'amount_of_layers', 'optimizer', 'activation_function', 'pooling_algorithm', 'acc', 'f1', 'roc', 'pr', 'time']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writeheader()
            csvfile.close()

    def CSVWriteRow(self, params, eval : EvaluationMetricsData, time):
        self.writer.writerow({'dropout_rate': params['dropout_rate'], 'hidden_channels': params['hidden_channels'], 'learning_rate': params['learning_rate'], 'batch_size': params['batch_size'], 'epochs': params['epochs'], 'amount_of_layers': params['amount_of_layers'], 'optimizer': params['optimizer'], 'activation_function': params['activation_function'], 'pooling_algorithm': params['pooling_algorithm'], 'acc': eval.accuracy, 'f1': eval.f1, 'roc': eval.roc, 'pr': eval.pr, 'time' : time})

    def CSVOpen(self) : 
        self.csvfile = open(self.name, 'a', newline = '')
        fieldnames = ['dropout_rate', 'hidden_channels', 'learning_rate', 'batch_size', 'epochs', 'amount_of_layers', 'optimizer', 'activation_function', 'pooling_algorithm', 'acc', 'f1', 'roc', 'pr', 'time']
        self.writer = csv.DictWriter(self.csvfile, fieldnames = fieldnames)

    def CSVClose(self):
        self.csvfile.close()

class ModelData():
    def __init__(self, dropout_rate, hidden_channels, learning_rate, batch_size, epochs, amount_of_layers, optimizer, activation_function, pooling_algorithm, eval_name, eval_data):
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.amount_of_layers = amount_of_layers
        self.optimizer = optimizer
        self.activation_function = activation_function
        self.pooling_algorithm = pooling_algorithm
        self.eval_name = eval_name
        self.eval_data = eval_data
