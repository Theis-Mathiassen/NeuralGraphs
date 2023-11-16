import torch
from Classes import GCN, BaseModel, EvaluationMetricsData
from Train_Test import train, test
from tqdm import trange
from torch_geometric.loader import DataLoader


# Basic search model that is used for testing parameters for each of the hyper parameter optimizers
# Inputs:
# params - library of parameters (str -> array)
# train_dataset: train dataset containing graphs
# test_dateset: test dataset containing graphs
# device: contains device to run progam on

def search_model(params, train_dataset, test_dataset, device) : 
        # Define the data loaders, this uses batch_size
        train_loader = DataLoader(dataset=train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"], shuffle=False)

        # Create model
        model = GCN(hidden_channels=params["hidden_channels"], dropout_rate=params["dropout_rate"], learning_rate=params["learning_rate"], 
                        activation_function=params["activation_function"], amount_of_layers=params["amount_of_layers"], pooling_algorithm=params["pooling_algorithm"])
        model.to(device)

        #Deciding which optimizer to use
        optimizer = None
        if(params["optimizer"].lower() == 'sgd'): optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate)
        elif(params["optimizer"].lower() == 'adam'): optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        elif(params["optimizer"].lower() == 'rmsprop'): optimizer = torch.optim.RMSprop(model.parameters(), lr=model.learning_rate)
        else : raise Exception("Invalid optimizer name: " + str(params["optimizer"]))

        loss_function = torch.nn.CrossEntropyLoss() # Why not a hyper param?

        baseModel = BaseModel(model, loss_function, optimizer) 


        for epoch in trange(0, params['epochs']):
            # TRAIN
            train(baseModel, train_loader, device)
        test_data = test(baseModel, test_loader, device)

        return test_data, model