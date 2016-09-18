import numpy as np

#LSTM: learningRate, forgetBias, hyperParam, batchSize

def random_search_params(numParams, searchSpace):
    random_solution = np.zeros(numParams)
    i = 0
    while i < numParams:
        random_solution[i] = np.random.random_sample()*searchSpace[i]
        i += 1
    return random_solution

#num_iter = 10
#searchSpace_lstm = [1, 10, 10, 1000]
#numParams_lstm = 4
#print(random_search_params(numParams_lstm, searchSpace_lstm))