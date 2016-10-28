import sys
import csv
import math
from random import shuffle

FEATURE_COUNT = 57
LABEL_INDEX = 58

def mean(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]

    return total / len(numbers)

def stdev(numbers):
    meanValue = mean(numbers)

    total = 0
    for i in range(len(numbers)):
        total += pow(numbers[i] - meanValue, 2)

    return math.sqrt(total / len(numbers))

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

def truncate(featureMatrix, labelMatrix):
    newFeatureMatrix = []
    newLabelMatrix = []
    for i in range(1, len(featureMatrix)):
        if not featureMatrix[i] == featureMatrix[i - 1]:
            newFeatureMatrix.append(featureMatrix[i])
            newLabelMatrix.append(labelMatrix[i])
    del featureMatrix[:]
    featureMatrix.extend(newFeatureMatrix)
    del labelMatrix[:]
    labelMatrix.extend(newLabelMatrix)

def writeMatrix(filename, matrix):
    with open(filename, 'wb') as file:
        writer = csv.writer(file, delimiter = ' ')
        for row in matrix:
            if isinstance(row, float):
                writer.writerow([row])
            else:
                writer.writerow(row)

def featurePreprocess(featureMatrix):
    # Get feature mean and SD
    CAPITAL_RUN_LENGTH_AVERAGE_MEAN = mean([row[54] for row in featureMatrix])
    CAPITAL_RUN_LENGTH_AVERAGE_SD = stdev([row[54] for row in featureMatrix])
    CAPITAL_RUN_LENGTH_LONGEST_MEAN = mean([row[55] for row in featureMatrix])
    CAPITAL_RUN_LENGTH_LONGEST_SD = stdev([row[55] for row in featureMatrix])
    CAPITAL_RUN_LENGTH_TOTAL_MEAN = mean([row[56] for row in featureMatrix])
    CAPITAL_RUN_LENGTH_TOTAL_SD = stdev([row[56] for row in featureMatrix])

    # Feature preprocessing
    newFeatureMatrix = []
    for i in range(len(featureMatrix)):
        newFeatureRow = []

        # Copy non-preprocessing features
        NONPREPROCESSING_FEATURE_LIST = range(0, 54)
        for j in NONPREPROCESSING_FEATURE_LIST:
            newFeatureRow.append(featureMatrix[i][j])

        # Normalized features
        newFeatureRow.append((featureMatrix[i][54] - CAPITAL_RUN_LENGTH_AVERAGE_MEAN) / CAPITAL_RUN_LENGTH_AVERAGE_SD)
        newFeatureRow.append((featureMatrix[i][55] - CAPITAL_RUN_LENGTH_LONGEST_MEAN) / CAPITAL_RUN_LENGTH_LONGEST_SD)
        newFeatureRow.append((featureMatrix[i][56] - CAPITAL_RUN_LENGTH_TOTAL_MEAN) / CAPITAL_RUN_LENGTH_TOTAL_SD)

        # Feature expansion
        newFeatureRow.append(featureMatrix[i][54] / featureMatrix[i][55])
        newFeatureRow.append(featureMatrix[i][54] / featureMatrix[i][56])
        newFeatureRow.append(featureMatrix[i][55] / featureMatrix[i][56])
        
        newFeatureMatrix.append(newFeatureRow)

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

if len(featureMatrix) != 0:
    truncate(featureMatrix, labelMatrix)

    featureMatrix = featurePreprocess(featureMatrix)

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
if len(testingFeatureMatrix) != 0:
    testingFeatureMatrix = featurePreprocess(testingFeatureMatrix)
    writeMatrix(filename_testingFeatureMatrix, testingFeatureMatrix)

