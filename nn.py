import math
import numpy as np
activations = ['relu', 'sigmoid']

class NeuralNetwork:
    def __init__(self, model = None, input_shape = None, hidden_shape = None, output_shape = None, dropout_percentage=0.02, mutate_ratio = 0.01):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.dropout_percentage = dropout_percentage
        self.mutate_ratio = mutate_ratio
        self.activations = ['relu', 'sigmoid', 'softmax']
        if model == None:
            model = self.create_model()
        self.model = model

    def copy(self):
        model = self.create_model()
        return NeuralNetwork(
            model=model,
            input_shape=self.input_shape,
            hidden_shape=self.hidden_shape,
            output_shape=self.output_shape,
            dropout_percentage=self.dropout_percentage)

    def mutate(self):
        for z,l in enumerate(self.model):
            for i,r in enumerate(l):
                for j in range(len(r)):
                    if (np.random.uniform(0,1,1)[0] < self.mutate_ratio):
                        self.model[z][i][j] = np.random.uniform(-1,1,1)[0]
        return self

    def crossover(self, other):
        i = round(np.random.uniform(0,2,1)[0])

        self.model[i] = other.model[i]
        self.activations[i] = other.activations[i]

        return self


    def predict(self, sensors):
        L1, L2, L3 = self.model
        A1, A2, A3 = self.activations

        X = np.array(sensors)
        Z1 = np.matmul(L1, X)
        O1 = getattr(self, A1)(Z1)
        Z2 = np.matmul(L2, O1)
        O2 = getattr(self, A2)(Z2)
        Z3 = np.matmul(L3, O2)
        Y = getattr(self, A3)(Z3)
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

    def create_model(self):
        L1 = np.random.uniform(-1, 1, size=(self.input_shape[1], self.input_shape[0]))
        L2 = np.random.uniform(-1, 1, size=(self.hidden_shape[1], self.hidden_shape[0]))
        L3 = np.random.uniform(-1, 1, size=(self.output_shape[1], self.output_shape[0]))
        A1 = round(np.random.uniform(0,len(activations)-1,1)[0])
        A2 = round(np.random.uniform(0,len(activations)-1,1)[0])
        self.activations[0] = activations[A1]
        self.activations[1] = activations[A2]
        model = [L1,L2,L3]

        return model