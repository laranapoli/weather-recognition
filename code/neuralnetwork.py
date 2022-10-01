import numpy as np

class ForwardPropagation:
    def __linear_forward(self, A, W, b):
        # Linear part of forward propagation: Z = W * A(prev) + b            
        Z = np.dot(W, A) + b
        
        cache = (A, W, b)
        
        return Z, cache

    def __relu(self, Z):    
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        
        return A, cache

    def __softmax(self, Z):
        # For multiclass problems' output layer.             
        exp_values = np.exp(Z) 
        A = exp_values / np.sum(np.exp(Z), axis=0, keepdims=True)  # axis=0 needed to sum over columns

        cache = Z

        return A, cache

    def __linear_activation_forward(self, A_prev, W, b, activation):
        # Outputs the activation function for each layer storing both linear and activation caches ((A, W, b), Z).
        if activation == "softmax":
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = self.__softmax(Z)
        
        elif activation == "relu":
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = self.__relu(Z)        

        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        # Forward propagation: Repeats RELU (L-1) times and the last one with SOFTMAX.
        # It will perform one iteration of forward propagation through all layers and output the predictions (AL).
        
        caches = []
        A = X
        L = len(parameters) // 2             # number of layers in the neural network
        
        # [LINEAR -> RELU]*(L-1)
        for l in range(1, L):
            A_prev = A 
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]

            A, cache = self.__linear_activation_forward(A_prev, W, b, activation='relu')
            caches.append(cache)
        
        # LINEAR -> SOFTMAX
        A_prev = A                     # from last iteration (layer)
        W = parameters["W" + str(L)]
        b = parameters["b" + str(L)]
        AL, cache = self.__linear_activation_forward(A_prev, W, b, activation='softmax')
        caches.append(cache)
            
        return AL, caches

    def compute_cost(self, AL, Y):
        # When using softmax as the activation function on the output layer, Categorical Cross-Entropy is the function generally applied."""
        softmax_outputs = AL.T

        y_pred_clipped = np.clip(softmax_outputs, 1e-7, 1-1e-7) # Prevents log of 0

        correct_confidences = y_pred_clipped[range(len(softmax_outputs)), Y]

        negative_log_likelihoods = -np.log(correct_confidences)

        cost = np.mean(negative_log_likelihoods)

        return cost

class BackwardPropagation:
    def __one_hot(self, Y):
        #One hot encodes Y, making each column an example.
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def __relu_backward(self, dA, cache):
        Z = cache                      # activation cache
        dZ = np.array(dA, copy=True)   # converting dz to a correct object.
        
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def __linear_backward(self, dZ, cache):
        # Compute derivatives (gradients) of the cost with respect to weights, biases and activation of previous layer - dW(l), db(l) and dA(l-1)         
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def __linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = self.__relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
            
        elif activation == "softmax":
            dA_prev, dW, db = self.__linear_backward(dA, linear_cache)
        
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        # Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SOFTMAX.        
        grads = {}
        L = len(caches)           # number of layers

        one_hot_Y = self.__one_hot(Y)
        dZL = AL - one_hot_Y
        
        # Lth layer (SOFTMAX -> LINEAR) gradients.
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(dZL, current_cache, activation='softmax')
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] =db_temp
        
        # Loop from l=L-2 to l=0 (ReLU)
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation='relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp       

        return grads

    def update_parameters(self, params, grads, learning_rate):
        # Update weights and biases
        parameters = params.copy()
        L = len(parameters) // 2        # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        return parameters


class MultiClassNeuralNetwork(ForwardPropagation, BackwardPropagation):
    def __init__(self, layer_dimensions):
        ForwardPropagation.__init__(self)
        BackwardPropagation.__init__(self)
        self.layer_dimensions = layer_dimensions
        

    def __initialize_parameters(self):    
        # Random initialization       
        parameters = {}
        layer_dims = self.layer_dimensions
        L = len(layer_dims) # number of layers in the network
        

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters


    def train_model(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):        
        costs = []                         # keep track of cost
        
        # Parameters initialization.
        parameters = self.__initialize_parameters()
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
            AL, caches = self.L_model_forward(X, parameters)
            
            # Compute cost.
            cost = self.compute_cost(AL, Y)

            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)
            
            # Update parameters.
            if i > 20000:
                lr = learning_rate*0.6
                parameters = self.update_parameters(parameters, grads, lr)
            elif i > 10000:
                lr = learning_rate*0.8
                parameters = self.update_parameters(parameters, grads, lr)
            else:
                parameters = self.update_parameters(parameters, grads, learning_rate)
            
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        
        return parameters, costs

        
    def predict(self, X, y, parameters, print_accuracy=False):
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)
        # From output units, 
        p = np.argmax(probas, 0).reshape(y.shape)
        accuracy = np.mean((p == y))

        if print_accuracy:
            print("Accuracy: " + str(accuracy))
            
        return p, accuracy

    def make_prediction(self, X, parameters):
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)
        # From output units
        p = np.argmax(probas, 0)

        return p