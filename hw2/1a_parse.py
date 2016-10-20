import sys
import csv
import math
from random import shuffle

FEATURE_COUNT = 57
LABEL_INDEX = 58

def parseFileIntoFeatureMatrix(filename):
    rawMatrix = []
    with open(filename, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            rawMatrix.append(row)
    
    featureMatrix = []
    for i in range(0, len(rawMatrix)):
        featureRow = []
        for j in range(1, 1 + FEATURE_COUNT):
            element = rawMatrix[i][j]
            element = float(element)

            featureRow.append(element)
        
        featureMatrix.append(featureRow)

    return featureMatrix

def parseFileIntoLabelMatrix(filename):
    rawMatrix = []
    with open(filename, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            rawMatrix.append(row)
    
    labelMatrix = []
    for i in range(0, len(rawMatrix)):
        element = rawMatrix[i][LABEL_INDEX]
        element = float(element)

        labelMatrix.append(element)

    return labelMatrix

def writeMatrix(filename, matrix):
    with open(filename, 'wb') as file:
        writer = csv.writer(file, delimiter = ' ')
        for row in matrix:
            if isinstance(row, float):
                writer.writerow([row])
            else:
                writer.writerow(row)

# Parse arguments         
filename_train = sys.argv[1]
filename_test = sys.argv[2]
filename_featureMatrix = sys.argv[3]
filename_labelMatrix = sys.argv[4]
filename_testingFeatureMatrix = sys.argv[5]

# Training data
featureMatrix = parseFileIntoFeatureMatrix(filename_train)
labelMatrix = parseFileIntoLabelMatrix(filename_train)

randomIndex = range(len(featureMatrix))
shuffle(randomIndex)
newFeatureMatrix = []
for index in randomIndex:
    newFeatureMatrix.append(featureMatrix[index])
writeMatrix(filename_featureMatrix, newFeatureMatrix)

newLabelMatrix = []
for index in randomIndex:
    newLabelMatrix.append(labelMatrix[index])
writeMatrix(filename_labelMatrix, newLabelMatrix)

# Testing data
testingFeatureMatrix = parseFileIntoFeatureMatrix(filename_test)
writeMatrix(filename_testingFeatureMatrix, testingFeatureMatrix)

