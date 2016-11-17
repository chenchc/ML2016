import os

from keras.models import load_model
import a_model
import pickle
import sys
import numpy
import random
import csv

def readTestData(filename):
	with open(filename) as file:
		testData = pickle.load(file)

	testMatrix = []

	for i in range(len(testData['data'])):
		instance = testData['data'][i]
		instanceInXYC = numpy.ndarray((32, 32, 3))
		for c in range(3):
			for y in range(32):
				for x in range(32):
					instanceInXYC[y, x, c] = float(instance[c * 32 * 32 + y * 32 + x]) / 255
		testMatrix.append(instanceInXYC)
	
	return numpy.array(testMatrix)


# Parse Arguments
testDataPath = sys.argv[1]
modelPath = sys.argv[2]
outputPath = sys.argv[3]

# Read Data
testMatrix = readTestData(testDataPath + "/test.p")

# Test
model = load_model(modelPath)
predict = model.predict_classes(testMatrix, batch_size=8)
with open(outputPath, 'wb') as file:
	writer = csv.writer(file, delimiter=',')
	writer.writerow(['ID', 'class'])
	for i in range(len(predict)):
		writer.writerow([i, predict[i]])
