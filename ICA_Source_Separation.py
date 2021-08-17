import numpy as np
import scipy.io.wavfile
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal
import librosa 
from IPython.display import Audio
import random

def getSoundMatrix(files):
    sound = np.zeros((76800,), dtype=int)
    sound = np.mat(sound)
    for i in range(len(files)):
        rate, data = scipy.io.wavfile.read(files[i])
        data = np.mat(data)
        sound = np.append(sound, data, axis=0)

    sound= np.delete(sound,0,axis = 0)
    return sound

def getEigenDecomposition(sound):
    covMatrix = np.cov(sound)
    eigenValues, eigenVector = np.linalg.eig(covMatrix)
    eigenValues = eigenValues.reshape(1,20)
    return eigenValues, eigenVector

def getWhiteningMatrix(eigenValues, eigenVector, rank = 4):
    whiteningMatrix = np.zeros(20)
    whiteningMatrix = np.mat(whiteningMatrix)
    whiteningMatrix = whiteningMatrix.reshape(1,20)
    for i in range(rank):
        temp = np.divide(eigenVector[i,:], np.sqrt(eigenValues[0,i]))
        temp = np.mat(temp)
        whiteningMatrix = np.append(whiteningMatrix, temp, axis=0)
    whiteningMatrix = np.delete(whiteningMatrix,0, axis = 0)
    return whiteningMatrix

def getICA(Z):
    # Initialise Parameters for the ICA Algorithm
    N = 76800
    learningRate = 10**-8
    W = np.random.rand(4,4)
    W = np.mat(W)
    print("Initial W is-------------")
    print(W)
    I = np.identity(4)
    I = np.mat(I)
    Y = W @ Z

    # ICA Algorithm Main update loop
    flag = False
    iterations = []
    differences = []
    i = 1
    while flag is False:
        FY = np.power(Y,3)
        GY = np.tanh(Y)
        deltaW = ((N * I) - (GY * np.transpose(FY))) @ W
        newW = W + (learningRate * deltaW)
        compare = np.sum(np.abs(newW - W))
        iterations.append(i)
        differences.append(compare)
        i = i + 1
        W = newW
        Y = W @ Z
        if compare <= 0.005:
            flag = True

    print("Final Converged W is-------------")
    print(W)
    print("Number of iterations is", len(iterations))
    return Y
    
def generateOutputSoundFile(Y):
    Y = Y *1000
    for i in range(Y.shape[0]):
        temp = Y[i]
        temp = np.array(temp)
        temp = temp.astype(np.int16)
        scipy.io.wavfile.write('outputSourceSound' + str(i+1) + '.wav' , 16000, temp.T)

    print("Source Separation Successful! Please find the source files in the same directory as the application.")

# Reading the files and applying ICA Algorithm for Source Separation

files = ['x_ica_1.wav','x_ica_2.wav','x_ica_3.wav','x_ica_4.wav','x_ica_5.wav',
        'x_ica_6.wav','x_ica_7.wav','x_ica_8.wav','x_ica_9.wav','x_ica_10.wav',
        'x_ica_11.wav','x_ica_12.wav','x_ica_13.wav','x_ica_14.wav','x_ica_15.wav',
        'x_ica_16.wav','x_ica_17.wav','x_ica_18.wav','x_ica_19.wav','x_ica_20.wav']

sound = getSoundMatrix(files)
eigenValues, eigenVector = getEigenDecomposition(sound)
print("Eigen Values are")
print(eigenValues)
print("From the EigenValues, we can conclude that there are 4 sound sources that are present in these sound files.")
whiteningMatrix = getWhiteningMatrix(eigenValues, eigenVector, rank = 4)
Z = whiteningMatrix @ sound
Y = getICA(Z)
print("Shape of Y Matrix", Y.shape)
generateOutputSoundFile(Y)
