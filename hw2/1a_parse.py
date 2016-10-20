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

def featureSelection(featureMatrix):
    FEATURE_LIST = [5, 6, 7, 8, 9, 10, 12]
    DATE_LIST = [2, 3, 4, 5, 6, 7, 8]
    newFeatureMatrix = []
    for row in featureMatrix:
        newFeatureMatrix.append([])
        for i in range(len(row)):
            if i / FEATURE_COUNT in DATE_LIST and i % FEATURE_COUNT in FEATURE_LIST:
                newFeatureMatrix[len(newFeatureMatrix) - 1].append(row[i])

    return newFeatureMatrix
       
# Parse arguments         
filename_train = sys.argv[1]
filename_test = sys.argv[2]
filename_featureMatrix = sys.argv[3]
filename_labelMatrix = sys.argv[4]
filename_testingFeatureMatrix = sys.argv[5]

# Training data
featureMatrix = parseFileIntoFeatureMatrix(filename_train)
labelMatrix = parseFileIntoLabelMatrix(filename_train)

#featureMatrix = featureSelection(featureMatrix)

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
#testingFeatureMatrix = featureSelection(testingFeatureMatrix)
writeMatrix(filename_testingFeatureMatrix, testingFeatureMatrix)

