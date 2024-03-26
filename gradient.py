import numpy as np
from neuralnet import Neuralnetwork
import copy

def check_grad(model, x_train, y_train):

    """
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-2 
    deepcopy = copy.deepcopy(model) # Store a copy of the model
    
    # Dictionary of weights to check
    weights = {
        'output bias': (-1, 0, 0),
        'hidden bias': (0, 0, 0),
        'output weight 1': (-1, 1, 0),
        'output weight 2': (-1, 2, 0),
        'hidden weight 1': (0, 1, 0),
        'hidden weight 2': (0, 2, 0),
    }
    
    # Loop through each weight we want to check
    for name, (layer, row, col) in weights.items():
        original = model.layers[layer].w[row][col] # Store original value
        # Compute gradient backprop
        model.forward(x_train, y_train)
        model.backward(gradReqd=False)
        grad_backprop = model.layers[layer].dw[row][col]
        
        # Compute loss with +epsilon
        model.layers[layer].w[row][col] = original + epsilon
        loss_plus, _ = model.forward(x_train, y_train)
        
        # Compute loss with -epsilon
        model.layers[layer].w[row][col] = original - epsilon
        loss_minus, _ = model.forward(x_train, y_train)
        
        # Compute gradient approximation
        grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
        
        difference = abs(abs(grad_approx) - abs(grad_backprop))
        
        # Print results
        print("Grad approx weight({}): ".format(name), abs(grad_approx))
        print("Grad backprop weight({}): ".format(name), abs(grad_backprop))
        print("Difference({}): ".format(name), difference)
        if difference < epsilon**2:
            print("{} gradient is correct".format(name))
        else:
            print("{} gradient is incorrect".format(name))
        
        # Reset model
        model = copy.deepcopy(deepcopy)
        print('------------------------------------------')
        
    
    
    
    
    


def checkGradient(x_train, y_train, config):
    subsetSize = 1  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
    
