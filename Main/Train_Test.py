from Classes import TrainData, TestData, BaseModel
import numpy as np
import torch

def train(model_env: BaseModel, data_loader, device):
    model_env.model.train()
    loss_ = 0
    correct = 0
    i = 0
    returnData = TrainData()

    for data in data_loader:  # Iterates the batches. We declared each batch to be of size 64
        data = data.to(device, non_blocking=True)
        # Calculate output, and get the maximum of those in order to obtain the predicted value
        out = model_env.model(data.x, data.edge_index, data.batch)
        cat = torch.argmax(out, dim=1)

        correct += int((cat == data.y).sum())  # Check against ground-truth labels.
        
        loss = model_env.loss_function(out, data.y)
        loss_ += loss.item()
        
        loss.backward()
        model_env.optimizer.step()
        model_env.optimizer.zero_grad()

        i+=1

        # Append actual and preddicted to respective array. Have to be converted to NumPy arrays in order to flatten them.
        # We flatten them as 1D arrays are required by SK in order to calculate and     plot ROC AUC
        #This is not going to change for each epoch, so computational power is wasted here...
        arrayLabel = np.array(data.y.to('cpu'))
        for value in arrayLabel.flatten():
            returnData.train_labels.append(value)

        arrayCat = np.array(cat.to('cpu'))
        for value in arrayCat.flatten():
            returnData.train_scores.append(value)
        # Turn output tensor into numpy array
        arrayPred = out.detach().cpu().numpy()
        for value in enumerate(arrayPred):
            returnData.train_probability_estimates.append(value[1][1])

    returnData.train_accuracies = (correct/len(data_loader.dataset))
    returnData.train_losses = (loss_/i)
    #tt.set_description("loss: %2f. accuracy %2f." % (loss, correct/len(train_loader.dataset)))
    return returnData


def test(model_env: BaseModel, data_loader, device):
    model_env.model.eval()
    correct = 0
    loss_ = 0
    i = 0
    returnData = TestData()

    for data in data_loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device, non_blocking=True)
        out = model_env.model(data.x, data.edge_index, data.batch)
        cat = torch.argmax(out, dim=1)

        correct += int((cat == data.y).sum())  # Check against ground-truth labels.

        loss = model_env.loss_function(out, data.y)
        loss_ += loss.item()

        i+=1

        arrayLabel = np.array(data.y.to('cpu'))
        for value in arrayLabel.flatten():
            returnData.test_labels.append(value)

        arrayCat = np.array(cat.to('cpu')) 
        for value in arrayCat.flatten():
            returnData.test_scores.append(value)

        arrayPred = out.detach().cpu().numpy()
        for value in enumerate(arrayPred):
            returnData.test_probability_estimates.append(value[1][1])

    returnData.test_accuracies = (correct/len(data_loader.dataset))
    returnData.test_losses = (loss_/ i)
    
    return returnData