import pandas as pd
import seaborn as sns
import numpy as np

def GetHeatData() :
    usecols = ["dropout_rate", "hidden_channels", "learning_rate", "batch_size","epochs","amount_of_layers","optimizer","activation_function","pooling_algorithm", "roc"]
    csv_data = pd.read_csv("results/Combined_Grid.csv", usecols = usecols)

    csv_data = csv_data.head(3456*4) #Currently necessary, but final version should not include this

    csv_data = pd.pivot_table(csv_data, values='roc', index=['dropout_rate', 'batch_size', 'hidden_channels', 'amount_of_layers'], columns=[ 'epochs', 'learning_rate', 'activation_function', 'optimizer', 'pooling_algorithm'])

    return csv_data

def GetParamData(parameter, score, cutoff) : 
    
    count = 0
    pindex = 0
    sindex = 0
    for param in ['dropout_rate','hidden_channels','learning_rate','batch_size','epochs','amount_of_layers','optimizer','activation_function','pooling_algorithm','acc','f1','roc','pr']:
        if parameter == param:
            pindex = count
        if score == param:
            sindex = count
        count += 1
    usecols = (pindex, sindex)

    grid_data = np.genfromtxt("results/Combined_Grid.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff)

    bayes_data = np.genfromtxt("results/bayes_data.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff)

    return grid_data, bayes_data

def GetHistData(score, cutoff = 0) : 
    count = 9
    index = 9
    for param in ['acc','f1','roc','pr']:
        if score == param:
            index = count
        count += 1

    usecols = (index)
    grid_data = np.genfromtxt("results/Combined_Grid.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff)
    bayes_data = np.genfromtxt("results/bayes_data.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff+20)
    bayes_data = bayes_data[20:]

    return grid_data, bayes_data