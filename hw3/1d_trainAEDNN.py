import os

from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import Input, Dense, Reshape, Activation, Flatten, BatchNormalization
import a_model
import pickle
import sys
import numpy
import random
import os.path
import copy

def randomShuffle(a, b):
	p = range(len(a))
	random.shuffle(p)
	newA = []
	newB = []
	for index in p:
		newA.append(a[index])
		newB.append(b[index])
	return newA, newB

def flipAugment(trainMatrix, labelMatrix):
	trainMatrix = list(trainMatrix)
	labelMatrix = list(labelMatrix)
	num = len(trainMatrix)
	for i in range(num):
		trainMatrix.append(numpy.fliplr(trainMatrix[i]))
		labelMatrix.append(labelMatrix[i])
	trainMatrix = numpy.array(trainMatrix)
	labelMatrix = numpy.array(labelMatrix)
	return trainMatrix, labelMatrix

def flipAugment2(trainMatrix):
	trainMatrix = list(trainMatrix)
	num = len(trainMatrix)
	for i in range(num):
		trainMatrix.append(numpy.fliplr(trainMatrix[i]))
	trainMatrix = numpy.array(trainMatrix)
	return trainMatrix

def readLabelData(filename):
	with open(filename) as file:
		labelData = pickle.load(file)

	trainMatrix = []
	labelMatrix = []

	for label in range(len(labelData)):
		labelDataOfLabel = labelData[label]
		for i in range(len(labelDataOfLabel)):
			instance = labelDataOfLabel[i]
			instanceInXYC = numpy.ndarray((32, 32, 3))
			for c in range(3):
				for y in range(32):
					for x in range(32):
						instanceInXYC[y, x, c] = float(instance[c * 32 * 32 + y * 32 + x]) / 255
			trainMatrix.append(instanceInXYC)

			labelInstance = numpy.ndarray((10))
			labelInstance.fill(0)
			labelInstance[label] = 1
			labelMatrix.append(labelInstance)
	
	trainMatrix, labelMatrix = randomShuffle(trainMatrix, labelMatrix)
	return [numpy.array(trainMatrix), numpy.array(labelMatrix)]

def readUnlabelData(filename):
	with open(filename) as file:
		unlabelData = pickle.load(file)

	trainMatrix = []

	for i in range(len(unlabelData)):
		instance = unlabelData[i]
		instanceInXYC = numpy.ndarray((32, 32, 3))
		for c in range(3):
			for y in range(32):
				for x in range(32):
					instanceInXYC[y, x, c] = float(instance[c * 32 * 32 + y * 32 + x]) / 255
		trainMatrix.append(instanceInXYC)

	return numpy.array(trainMatrix)

def readUnlabelDataFromTest(filename):
	with open(filename) as file:
		unlabelData = pickle.load(file)

	trainMatrix = []

	for i in range(len(unlabelData['data'])):
		instance = unlabelData['data'][i]
		instanceInXYC = numpy.ndarray((32, 32, 3))
		for c in range(3):
			for y in range(32):
				for x in range(32):
					instanceInXYC[y, x, c] = float(instance[c * 32 * 32 + y * 32 + x]) / 255
		trainMatrix.append(instanceInXYC)

	return numpy.array(trainMatrix)

random.seed(0)

input_img = Input(shape=(32, 32, 3))
encoded = Flatten()(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(256, activation='relu')(encoded)

decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(32*32*3, activation='sigmoid')(encoded)
decoded = Reshape((32, 32, 3))(decoded)

ae = Model(input=input_img, output=decoded)
ae.compile(loss='binary_crossentropy', optimizer='adam')

dnnd = Dense(512, activation='relu')(encoded)
dnnd = Dense(10, activation='softmax')(dnnd)

aednn = Model(input=input_img, output=dnnd)
aednn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Parse Arguments
trainDataPath = sys.argv[1]
modelPath = sys.argv[2]

# Read Label Data
print "Read label data..."
output = readLabelData(trainDataPath + "/all_label.p")
labelTrainMatrix = output[0][0:4500]
labelLabelMatrix = output[1][0:4500]
valTestMatrix = output[0][4500:5000]
valLabelMatrix = output[1][4500:5000]

# Read Unlabel Data
print "Read unlabel data..."
unlabelTrainMatrix = labelTrainMatrix
#unlabelTrainMatrix = numpy.concatenate((unlabelTrainMatrix, readUnlabelDataFromTest(trainDataPath + "/test.p"), readUnlabelData(trainDataPath + "/all_unlabel.p")))

# Train AE
print "Training AE..."

unlabelTrainMatrix = flipAugment2(unlabelTrainMatrix)
output = copy.copy(unlabelTrainMatrix)
ae.fit(unlabelTrainMatrix, output, batch_size=16,
       nb_epoch=1, validation_data=(valTestMatrix, valTestMatrix), verbose=1, callbacks=[EarlyStopping(monitor='val_loss', mode='min')])


# Train


print "Training AE+DNN..."
labelTrainMatrix, labelLabelMatrix = flipAugment(labelTrainMatrix, labelLabelMatrix)
randomShuffle(labelTrainMatrix, labelLabelMatrix)
aednn.fit(labelTrainMatrix, labelLabelMatrix, batch_size=8, nb_epoch=25, validation_data=(valTestMatrix, valLabelMatrix), callbacks=[ModelCheckpoint(modelPath, monitor='val_loss', save_best_only=True), EarlyStopping(monitor='val_loss', patience=2, mode='min')])


