import random
import copy

def train(inputs, targets):
    '''Take a two dimensional list as the first input where each inner list
    corresponds to a single input. Each inner list item is either 0 or 1.
    The second input is a two dimensional list of training target data. Each targets item
    is a list containing either 0 or 1 and represents the target output value for a
    particular neuron.
    Return a two dimensional list of weights.'''

    #Number of inputs
    inputDim = len(inputs[0])
    #Number of neurons
    outputDim = len(targets[0])
    #Size of data
    if (len(inputs) == len(targets)):
        dataSize = len(inputs)
    else:
        print("Number of input entries is not equal to number of output entries")
        return

    weights = [];
    
    #initialize the weights list with default values
    for i in range(inputDim + 1):
        weights.append([]);
        for j in range(outputDim):
            weights[i].append(2*random.random() - 1);

    #variable to keep track of weights from the previous iteration
    weightsPrevious = 0

    iterCount = 1
    while (weightsPrevious != weights):
        print("iteration", iterCount)
        iterCount += 1
        weightsPrevious = copy.deepcopy(weights)
        for k in range(dataSize):
            for j in range(outputDim):
                #set activation to the weighted value obtained from the bias input
                activation = -1*weights[0][j]
                
                #compute the activation with the current weights
                for i in range(inputDim):
                    activation += inputs[k][i]*weights[i + 1][j]

                #Decide whether neuron fires or not
                if (activation > 0):
                    activation = 1
                else:
                    activation = 0

                print(activation)

                #Learning rate
                eta = 0.25

                #Update weights
                weights[0][j] += eta*(targets[k][j] - activation)*(-1)
                for i in range(inputDim):
                    weights[i + 1][j] += eta*(targets[k][j] - activation)*inputs[k][i]

    return weights

def predict(inputs, weights):
    '''Take a two dimensional list as the first input where each inner list
    corresponds to a single input. Each inner list item is either 0 or 1.
    The second input is a two dimensional list of weights obtained from
    the training dataset by running train() defined above.
    Return a two dimensional list of outputs. Each inner list item is either 0 or 1.'''

    #Number of inputs
    inputDim = len(inputs[0])
    #Number of neurons
    outputDim = len(weights[0])

    dataSize = len(inputs)

    outputs = []
    for k in range(dataSize):
            outputs.append([])
            for j in range(outputDim):
                #set activation to the weighted value obtained from the bias input
                activation = -1*weights[0][j]
                
                #compute the activation with the current weights
                for i in range(inputDim):
                    activation += inputs[k][i]*weights[i + 1][j]

                #Decide whether neuron fires or not
                if (activation > 0):
                    outputs[k].append(1)
                else:
                    outputs[k].append(0)

    return outputs

