import numpy as np
'''try to learn binary'''

def genData(numFeat):
    numData = np.power(2,numFeat) #128
    binary = lambda n: n>0 and [n&1]+binary(n>>1) or []
    binaryNumbers = np.zeros((numData, numFeat))
    for n in range(1, numData):
        zeros = np.zeros(numFeat)
        bin = binary(n)
        #print(n, bin)
        for i in range(len(bin)-1, -1, -1):
            zeros[i] = bin[i]
            binaryNumbers[n] = zeros[::-1]

    labels = np.zeros((numData, numData))
    for i in range(0, labels.shape[0]):
        labels[i, i]=1

    return binaryNumbers, labels

#bin, label = genData(7)
#print(label.shape)