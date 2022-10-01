import numpy as np
from neuralnetwork import ForwardPropagation

class GradientCheck:
    def __init__(self, layer_dimensions, parameters, epsilon=1e-7):
        self.epsilon = epsilon
        self.params = parameters
        self.layer_dimensions = layer_dimensions

    def __dictionary_to_vector(self):
        """Transforms the shape of 'parameters' to a single vector."""
        parameters = self.params
        keys = []
        count = 0

        for key in parameters.keys():
            # flatten parameter
            new_vector = np.reshape(parameters[key], (-1, 1))
            keys = keys + [key] * new_vector.shape[0]
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, keys

    def __gradients_to_vector(self, gradients):
        """Reshape gradients dictionary into a single vector."""
        L = len(self.layer_dimensions)
        count = 0
        keys = []

        for l in range(1, L):
            keys.append("dW" + str(l))
            keys.append("db" + str(l))

        for key in keys:
            # flatten parameter
            new_vector = np.reshape(gradients[key], (-1, 1))
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta

    def __vector_to_dictionary(self, theta):
        """Unroll our vector into previous parameters dictionary's shape."""
        original_parameters = self.params
        parameters = {}
        initial_value = 0

        for k,v in original_parameters.items():
            shape = v.shape
            number_items = (shape[0] * shape[1])
            final_value = initial_value + number_items
                
            parameters[k] = theta[initial_value: final_value].reshape(shape)
            initial_value = final_value

        return parameters

    def gradient_check_n(self, gradients, X, Y, epsilon=1e-7, print_msg=False):
        """This method aims to calculate a numerical approximation of derivatives to check if backward propagation is correctly implemented."""
    
        # Set-up variables
        parameters_values, _ = self.__dictionary_to_vector()
                
        grad = self.__gradients_to_vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        fp = ForwardPropagation()
        
        # Compute gradapprox
        for i in range(num_parameters):
            theta_plus = np.copy(parameters_values) 
            theta_plus[i] += epsilon

            # Forward propagation:
            AL, caches = fp.L_model_forward(X, self.__vector_to_dictionary(theta_plus))
            # Compute cost.
            J_plus[i] = fp.compute_cost(AL, Y)
            
            theta_minus = np.copy(parameters_values) 
            theta_minus[i] -= epsilon 

            # Forward propagation:
            AL, caches = fp.L_model_forward(X, self.__vector_to_dictionary(theta_minus))
            # Compute cost.
            J_minus[i] = fp.compute_cost(AL, Y)
            
            # Compute gradapprox[i]
            gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

        # Compare gradapprox to backward propagation gradients by computing euclidean distance
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) +  np.linalg.norm(gradapprox)
        difference = numerator / denominator  
        
        # Print result
        if print_msg:
            if difference > 1e-2:
                print ("\033[93m" + "There is a mistake in the backward propagation! Difference = " + str(difference) + "\033[0m")
            elif difference > 1e-4:
                print ("\033[93m" + "You should feel uncomfortable. Difference = " + str(difference) + "\033[0m")
            elif difference > 1e-7:
                print ("\033[93m" + "The difference is acceptable for objectives with kinks (e.g. use of tanh nonlinearities and softmax). Difference = " + str(difference) + "\033[0m")
            else:
                print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

        return difference