import pandas as pd
import seaborn as sns
import numpy as np

def GetHeatData() :
    usecols = ["dropout_rate", "hidden_channels", "learning_rate", "batch_size","epochs","amount_of_layers","optimizer","activation_function","pooling_algorithm", "roc"]
    csv_data = pd.read_csv("results/CombinedNew.csv", usecols = usecols)

    #csv_data = csv_data.tail(3456*10)
    csv_data = pd.pivot_table(csv_data, values=usecols[9], index=['dropout_rate', 'batch_size', 'hidden_channels', 'amount_of_layers'], columns=[ 'epochs', 'learning_rate', 'activation_function', 'optimizer', 'pooling_algorithm'])

    csv_data.fillna(0, inplace=True) # Fills "NaN" values with 0 instead
    csv_data = csv_data.loc[:, (csv_data != 0).any(axis=0)] # Remove all 0-entry columns. If this problem exists with rows, change axis to 1 instead

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

    grid_data = np.genfromtxt("results/CombinedNew.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff)

    bayes_data = np.genfromtxt("results/bayes_data.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff+20)
    bayes_data = bayes_data[20:]
    grid_data_s = np.empty(len(grid_data))
    grid_data_p = np.empty(len(grid_data))
    bayes_data_s = np.empty(len(bayes_data))
    bayes_data_p = np.empty(len(bayes_data))

    for i in range(0, len(grid_data)):
        grid_data_p[i] = grid_data[i][0]
        grid_data_s[i] = grid_data[i][1]
    for i in range(0, len(bayes_data)):
        bayes_data_p[i] = bayes_data[i][0]
        bayes_data_s[i] = bayes_data[i][1]
    

    return grid_data_p, grid_data_s, bayes_data_p, bayes_data_s

def GetHistData(score, cutoff = 0) : 
    count = 9
    index = 9
    for param in ['acc','f1','roc','pr']:
        if score == param:
            index = count
        count += 1

    usecols = (index)
    grid_data = np.genfromtxt("results/CombinedNew.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff)
    bayes_data = np.genfromtxt("results/bayes_data.csv", usecols=usecols, skip_header=True, delimiter=',', max_rows = cutoff+20)
    bayes_data = bayes_data[20:]

    return grid_data, bayes_data