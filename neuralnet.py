import numpy as np
import util

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.maximum(0, x)

    def output(self, x):
        """
        Implement softmax function here.
        Remember to take care of the overflow condition (i.e. how to avoid denominator becoming zero).
        """
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exponents = np.exp(shifted_x)
        return exponents / np.sum(exponents, axis=-1, keepdims=True)
        

    def grad_sigmoid(self, x):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def grad_tanh(self, x):
        """
        Compute the gradient for tanh here.
        """
        return 1 - np.tanh(x) ** 2

    def grad_ReLU(self, x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return np.where(x > 0, 1, 0)

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1  #Deliberately returning 1 for output layer case


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """
        # np.random.seed(42)

        # Randomly initialize weights (Includes bias term)
        self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    # output without activation
        self.z = None    # Output After Activation
        self.activation=activation

        self.dw_prev = None  # Save the previous weight update here for momentum
        self.dw = 0  # Save the gradient w.r.t w in this. w already includes bias term

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
         Compte the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = np.hstack((np.ones((x.shape[0], 1)), x)) # Add bias to input
        self.a = np.dot(self.x, self.w) # Calculate weighted input
        self.z = self.activation(self.a) # Calculate activation of weighted input
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, momentum, regularization_type, regularization_penalty, gradReqd=True):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputsx i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass

        When implementing softmax regression part, just focus on implementing the single-layer case first.
        """
        # Calculate dw and delta
        delta = deltaCur * self.activation.backward(self.a)
        self.dw = np.dot(self.x.T, delta)/self.x.shape[0]

        # Add regularization
        bias_buffer = np.zeros_like(self.w[0]) # Buffer for bias since we don't regularize it

        if regularization_type == "l2":
            reg_grad = 2 * self.w[1:] * regularization_penalty # w[1:] to not regularize bias
            reg_grad = np.vstack((bias_buffer, reg_grad)) # Add bias buffer back for calculation
            self.dw -= reg_grad # Subtract regularization gradient from dw as gradient is negative already
        elif regularization_type == "l1":
            reg_grad = regularization_penalty * np.sign(self.w[1:]) # w[1:] to not regularize bias
            reg_grad = np.vstack((bias_buffer, reg_grad)) # Add bias buffer back for calculation
            self.dw -= reg_grad # Subtract regularization gradient from dw as gradient is negative already


        
        
        # Add Momentum
        if momentum: # If momentum is true
            if self.dw_prev is None: # If dw_prev is not initialized
                self.dw_prev = np.zeros_like(self.dw)
            
            # Update dw with momentum
            self.dw = self.dw + momentum_gamma * self.dw_prev
            self.dw_prev = self.dw
        
        # Calculate new delta before updating weights
        new_delta = np.dot(delta, self.w[1:].T)
        
        # Update weights if gradReqd is true
        if gradReqd:
            self.w += learning_rate * self.dw
            
        return new_delta
            


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.momentum = config['momentum'] # Boolean for whether to use momentum
        self.momentum_gamma = config['momentum_gamma']

        # Check if regularization is required and set the type
        self.regularization_type = config['regularization_type']
        if self.regularization_type == "L2":
            self.regularization_penalty = config['L2_penalty']
        elif self.regularization_type == "L1":
            self.regularization_penalty = config['L1_penalty']
        else:
            self.regularization_penalty = 0

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1], Activation(config['activation'])))
            elif i  == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        # Save the input and targets
        self.x = x
        self.targets = targets
        
        # Loop through each layer and calculate the output of the layer
        for layer in self.layers:
            x = layer.forward(x)
        self.y = x # Save the output of the last layer in self.y
            
        # If targets are provided return the loss and number of correct predictions
        if targets is not None:
            loss = self.loss(x, targets)
            no_correct = util.calculateCorrect(x, targets)
            return loss/len(x), no_correct
        else:
            return x
        


    def loss(self, logits, targets):
        '''
        Compute the categorical cross-entropy loss and return it.
        '''   
        clipped_logits = np.clip(logits, 1e-10, 1 - 1e-10) # Used ChatGpt as we couldn't figure out why we kept getting nans
        return -np.sum(targets * np.log(clipped_logits)) 

    def backward(self, gradReqd=True):
        '''
        Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y
        # Loop through each layer and calculate the delta for the layer (backpropagate)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, self.momentum_gamma,\
                self.momentum, self.regularization_type, self.regularization_penalty,\
                        gradReqd=gradReqd)
            delta /= self.x.shape[0] # Normalizing the delta by batch size
