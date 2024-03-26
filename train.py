from neuralnet import *
from util import plots

def train(model, x_train, y_train, x_valid, y_valid, config, plot=False):
    """
    Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    # Loading the train hyperparameters
    epochs = config['epochs']
    batch_size = config['batch_size']
    early_stop_epoch = config['early_stop_epoch']
    
    # Initialize variables for early stopping
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    # For plotting
    losses, accuracies = {'train': [], 'valid': []}, {'train': [], 'valid': []} 
    
    # Training the model
    for i in range(epochs):
        # Shuffle the data and split into batches
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        epoch_loss, epoch_correct = 0, 0 # Init Loss and Correct predictions for the epoch
        
        # Loop through each batch
        for j in range(0, len(x_train), batch_size):
            start, end = j, min(j + batch_size, len(x_train)) # Get the start and end indices for the batch

            batch_x, batch_y = x_train[indices[start:end]], y_train[indices[start:end]] # Get the batch
            
            # Forward and backward pass
            loss, num_correct = model.forward(batch_x, batch_y)
            epoch_loss += loss
            epoch_correct += num_correct
            model.backward()
        
        # Calculate the average loss and accuracy for the epoch
        avg_loss = epoch_loss / (len(x_train) / batch_size)
        avg_acc = epoch_correct / len(x_train)
        # Append the loss and accuracy to the list for plotting
        losses['train'].append(avg_loss)
        accuracies['train'].append(avg_acc)
        # Print the loss and accuracy for the epoch
        print("Epoch: ", i + 1, " Loss: ", avg_loss, " Accuracy: ", avg_acc)
        
        # Validation
        validation_loss, validation_corrects = model.forward(x_valid, y_valid)
        # Append the loss and accuracy to the list for plotting
        losses['valid'].append(validation_loss)
        accuracies['valid'].append(validation_corrects / len(x_valid))
        # Print the loss and accuracy for the epoch
        print("Validation Loss: ", validation_loss, " Validation Accuracy: ", validation_corrects / len(x_valid))
        
        # Early stopping
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= early_stop_epoch:
            print("Early stopping triggered")
            break             
           
    if plot:
        # Plot the losses and accuracies
        plots(losses['train'], accuracies['train'], losses['valid'], accuracies['valid'], i)

    return model

#This is the test method
def modelTest(model, X_test, y_test):
    """
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    loss, no_correct = model(X_test, y_test)
    
    return no_correct / X_test.shape[0], loss
    