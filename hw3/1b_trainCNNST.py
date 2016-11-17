import os

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import a_model
import pickle
import sys
import numpy
import random
import os.path

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

# Read Model
print "Read model..."
firstTime = not os.path.isfile(modelPath)
if firstTime:
	model = a_model.get()
else:
	model = load_model(modelPath)

if not firstTime:
	# Read Unlabel Data
	print "Read unlabel data..."
	unlabelTrainMatrix = numpy.concatenate((readUnlabelDataFromTest(trainDataPath + "/test.p"), readUnlabelData(trainDataPath + "/all_unlabel.p")))

	# Self training
	print "Predicting unlabel..."
	score = model.predict(unlabelTrainMatrix, verbose=1)
	newUnlabelTrainMatrix = []
	labelTrainMatrix = list(labelTrainMatrix)
	labelLabelMatrix = list(labelLabelMatrix)
	for i in range(len(score)):
		for j in range(10):
			if score[i][j] > 0.9:
				labelTrainMatrix.append(unlabelTrainMatrix[i])

				labelInstance = numpy.ndarray((10))
				labelInstance.fill(0)
				labelInstance[j] = 1
				labelLabelMatrix.append(labelInstance)
				break
			elif j == 9:
				newUnlabelTrainMatrix.append(unlabelTrainMatrix[i])

	unlabelTrainMatrix = numpy.array(newUnlabelTrainMatrix)
	labelTrainMatrix = numpy.array(labelTrainMatrix)
	labelLabelMatrix = numpy.array(labelLabelMatrix)

# Train
print "Training..."
labelTrainMatrix, labelLabelMatrix = flipAugment(labelTrainMatrix, labelLabelMatrix)
randomShuffle(labelTrainMatrix, labelLabelMatrix)
model.fit(labelTrainMatrix, labelLabelMatrix, batch_size=8, nb_epoch=25, validation_data=(valTestMatrix, valLabelMatrix), callbacks=[ModelCheckpoint(modelPath, monitor='val_loss', save_best_only=True), EarlyStopping(monitor='val_loss', patience=2, mode='min')])


