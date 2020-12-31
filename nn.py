import math
import numpy as np
list_activations = ['relu', 'sigmoid', 'softplus']
output_activations = ['softmax', 'softplus']

class NeuralNetwork:
    def __init__(self, 
    model = None, 
    bias = None,
    random_weights_intensity = (-0.5, 0.5), 
    random_bias_intensity = (-0.1, 0.1),
    mutation_intensity = (-0.1, 0.1),
    input_shape = None, 
    hidden_shapes = [], 
    output_shape = None, 
    dropout_percentage=([0.2, 0.0]), 
    random_function=None, 
    crossover_probability = 0.5, 
    activations = [], 
    mutate_random_value = None, 
    continue_mutate_probability = 0.005, 
    continue_crossover_probability = 0.005):
        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        self.dropout_percentage = dropout_percentage
        self.continue_mutate_probability = continue_mutate_probability
        self.continue_crossover_probability = continue_crossover_probability
        self.random_function = random_function
        self.crossover_probability = crossover_probability
        self.random_weights_intensity = random_weights_intensity
        self.random_bias_intensity = random_bias_intensity
        self.mutation_intensity = mutation_intensity
        if self.random_function == None:
            self.random_function = lambda *args,**kwargs: np.random.uniform(*args, **kwargs)
        self.mutate_random_value = mutate_random_value
        if self.mutate_random_value == None:
            self.mutate_random_value = lambda: np.random.uniform(self.mutation_intensity[0], self.mutation_intensity[1])
        if model == None:
            model,activations,gen_bias = self.create_model()
        
        self.bias = bias
        if self.bias == None:
            self.bias = gen_bias
        self.model, self.activations = model,activations

    def copy(self):
        return NeuralNetwork(
            model = self.model, 
            bias = self.bias,
            random_bias_intensity = self.random_bias_intensity,
            random_weights_intensity = self.random_weights_intensity,
            mutation_intensity = self.mutation_intensity,
            activations=self.activations,
            input_shape = self.input_shape, 
            hidden_shapes = self.hidden_shapes, 
            output_shape = self.output_shape, 
            dropout_percentage = self.dropout_percentage, 
            random_function = self.random_function, 
            crossover_probability = self.crossover_probability,
            mutate_random_value = self.mutate_random_value,
            continue_mutate_probability = self.continue_mutate_probability,
            continue_crossover_probability = self.continue_crossover_probability)

    def mutate(self):
        def _mutate(model, mutate_random_value):
            z = math.floor(self.random_function(0, len(model)))
            i = math.floor(self.random_function(0, len(model[z])))
            j = math.floor(self.random_function(0, len(model[z][i])))
            model[z][i][j] += mutate_random_value()
            return model

        self.model = _mutate(self.model, self.mutate_random_value)
        while self.random_function(0,1) <= self.continue_mutate_probability:
            self.model = _mutate(self.model, self.mutate_random_value)
        return self

    def crossover(self, other):
        offspring = self.copy()
        model = offspring.model

        def _crossover(model, other_model, random_function, crossover_probability):
            res_model = np.copy(model)
            for i in range(len(model)):
                for j in range(model[i].shape[0] - 1):
                    for k in range(model[i].shape[1] - 1):
                        res_model[i][j, k] = np.random.choice([model[i][j, k], other_model[i][j, k]])
                for j in range(model[i].shape[1]):
                    res_model[i][0, j] = np.random.choice([model[i][0, j], other_model[i][0, j]])
            return res_model

        model = _crossover(model, other.model, self.random_function, self.crossover_probability)
        while self.random_function(0,1) <= self.continue_crossover_probability:
            model = _crossover(model, other.model, self.random_function, self.crossover_probability)

        offspring.model = model
        return offspring


    def predict(self, sensors, show_neural_activations = True):
        Y = sensors
        neural_activations = []
        for i,L in enumerate(self.model):
            X = Y
            dropout = np.random.binomial(1, 1 - self.dropout_percentage[i], size=L.shape)
            Lw = (L * dropout)
            Z = (np.dot(Lw, X) + self.bias[i])
            Y = getattr(self, self.activations[i])(Z)
            neural_activations.append((Lw,Y))

        if show_neural_activations:
            return (np.argmax(Y), neural_activations)
        return np.argmax(Y)

    def relu(self, values):
        values[values < 0] = 0
        return values

    def sigmoid(self, values):
        return np.array(list(map(
            lambda x: 1 / (1 + math.exp(-x)),
            values
        )))

    def softmax(self, values):
        e_x = np.exp(values - np.max(values))
        return e_x / e_x.sum()

    def softplus(self, values):
        return np.log(np.exp(values + 1))

    def create_model(self):
        model = []
        activations = []
        bias = []
        model.append(self.random_function(self.random_weights_intensity[0], self.random_weights_intensity[1], size=(self.input_shape[1], self.input_shape[0])))
        bias.append(self.random_function(self.random_bias_intensity[0], self.random_bias_intensity[1], size=(self.input_shape[1], )))
        activations.append(list_activations[math.floor(self.random_function(0,len(list_activations)))])
        for (i,o) in self.hidden_shapes:
            model.append(self.random_function(self.random_weights_intensity[0], self.random_weights_intensity[1], size=(o, i)))
            bias.append(self.random_function(self.random_bias_intensity[0], self.random_bias_intensity[1], size=(o,)))
            activations.append(list_activations[math.floor(self.random_function(0,len(list_activations)))])
        model.append(self.random_function(self.random_weights_intensity[0], self.random_weights_intensity[1], size=(self.output_shape[1], self.output_shape[0])))
        bias.append(self.random_function(self.random_bias_intensity[0], self.random_bias_intensity[1], size=(self.output_shape[1],)))
        activations.append(output_activations[math.floor(self.random_function(0,len(output_activations)))])
        return (model,activations,bias)