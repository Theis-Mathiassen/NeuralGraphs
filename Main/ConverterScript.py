import json
import csv
 

def flatten_json(y):
    out = {}
 
    def flatten(x, name=''):
 
        # If the Nested key-value
        # pair is of dict type
        if type(x) is dict:
 
            for a in x:
                flatten(x[a], name + a + '_')
 
        # If the Nested key-value
        # pair is of list type
        elif type(x) is list:
 
            i = 0
 
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
 
    flatten(y)
    return out



# Opening JSON file and loading the data
# into the variable data
with open('./results/Bayes2500.log.json') as json_file:
    jsondata = json.load(json_file)
 
# now we will open a file for writing
data_file = open('./results/Bayes2500.csv', 'w', newline='')

csv_writer = csv.writer(data_file)


count = 0
for data in jsondata:
    data = flatten_json(data)
    if count == 0:
        #head = {"target": 0.7928571428571429, "params": {"activation_function": 1.2243403512352373, "amount_of_layers": 2.3525580347650914, "batch_size": 74.43190859573681, "dropout_rate": 0.6346312362615614, "epochs": 66.11180783654648, "hidden_channels": 5.0, "learning_rate", "optimizer", "pooling_algorithm", "datetime", "elapsed", "delta"}
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())
 
data_file.close()





