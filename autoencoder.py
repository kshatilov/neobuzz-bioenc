import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

nChannels = 7
nPoints = 20
nSamples = 240
nTrain = 200
nTest = nSamples - nTrain
nPoses = 4 # how many classes you have
buzzDim = 4  


totalSamples = np.load("total_samples.npy")
totalSamples += np.random.normal(0, 0.01, size = totalSamples.shape) # add tabasco


trainIdx = np.random.choice(range(nSamples*nPoses), size = nTrain, replace = False)
xTrain = totalSamples[trainIdx]
testIdx = [i for i in range(nSamples*nPoses) if i not in trainIdx]
xTest = totalSamples[testIdx]



regRate = 1e-5


inputData = keras.Input(shape=(nPoints * nChannels,))


encoded = layers.Dense(int(nPoints * nChannels / 4),
                       activation='sigmoid',
                       activity_regularizer=regularizers.l1(regRate)
                      )(inputData)
encoded = layers.Dense(int(nPoints * nChannels / 8), 
                       activation='relu',
                       activity_regularizer=regularizers.l1(regRate)
                      )(encoded)
encoded = layers.Dense(buzzDim,
                       activation='relu',
                       activity_regularizer=regularizers.l1(regRate)
                      )(encoded)

decoded = layers.Dense(int(nPoints * nChannels / 8), 
                       activity_regularizer=regularizers.l1(regRate),
                       activation='relu')(encoded)
decoded = layers.Dense(int(nPoints * nChannels / 4), 
                       activity_regularizer=regularizers.l1(regRate),
                       activation='relu')(decoded)
decoded = layers.Dense(nPoints * nChannels, 
                       activation='sigmoid')(decoded)


autoencoder = keras.Model(inputData, decoded)

encoder = keras.Model(inputData, encoded)
encoded_input = keras.Input(shape=(buzzDim))


# create the decoder model for validating
decoder = autoencoder.layers[-3](encoded_input)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)


decoder = keras.Model(encoded_input, decoder)


autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(xTrain, xTrain,
                epochs=200,
                batch_size=16,
                shuffle=True,
                validation_data=(xTest, xTest))


encodedOut = encoder.predict(xTest)
decodedOut = decoder.predict(encodedOut)

encodedOut /= encodedOut.max(axis = 0) 
encodedOut *= 255 # to buzz vibration intensity

