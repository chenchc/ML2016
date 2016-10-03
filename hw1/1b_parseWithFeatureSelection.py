import sys
import csv
import math

TRAINDATA_ROW_BEGIN = 1
TRAINDATA_COL_BEGIN = 3
TRAINDATA_COL_COUNT = 24
TESTDATA_COL_BEGIN = 2
FEATURE_COUNT = 18
FEATURE_TIMESPAN = 9
FEATURE_PM25_INDEX = 9

def parseTrainFileIntoTimeSeries(filename):
    rawMatrix = []
    with open(filename, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            rawMatrix.append(row)

    timeSeries = []
    for i in range(1, len(rawMatrix), FEATURE_COUNT):
        # First row of a row set
        for j in range(TRAINDATA_COL_BEGIN, TRAINDATA_COL_BEGIN + TRAINDATA_COL_COUNT):
            element = rawMatrix[i][j]
            if element == 'NR':
                element = 0.0
            element = float(element)

            timeSeries.append([element])

        # Remaining rows of a row set
        for j in range(i + 1, i + FEATURE_COUNT):
            for k in range(TRAINDATA_COL_BEGIN, TRAINDATA_COL_BEGIN + TRAINDATA_COL_COUNT):
                element = rawMatrix[j][k]
                if element == 'NR':
                    element = 0.0
                element = float(element)

                timeSeriesIndex = len(timeSeries) - TRAINDATA_COL_COUNT + (k - TRAINDATA_COL_BEGIN)
                timeSeries[timeSeriesIndex].append(element)
    
    return timeSeries

def getFeatureMatrixGivenTimeSeries(timeSeries):
    featureMatrix = []
    for i in range(len(timeSeries) - FEATURE_TIMESPAN):
        featureRow = []
        for j in range(FEATURE_TIMESPAN):
            featureRow.extend(timeSeries[i + j])

        featureMatrix.append(featureRow)

    return featureMatrix

def getLabelMatrixGivenTimeSeries(timeSeries):
    labelMatrix = []
    for i in range(FEATURE_TIMESPAN, len(timeSeries)):
        labelMatrix.append(timeSeries[i][FEATURE_PM25_INDEX])

    return labelMatrix

def parseTestFileIntoTestingFeatureMatrix(filename):
    rawMatrix = []
    with open(filename, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            rawMatrix.append(row)
    
    testingFeatureMatrix = []
    for i in range(0, len(rawMatrix), FEATURE_COUNT):
        testingFeatureRow = []
        for j in range(TESTDATA_COL_BEGIN, TESTDATA_COL_BEGIN + FEATURE_TIMESPAN):
            for k in range(i, i + FEATURE_COUNT):
                element = rawMatrix[k][j]
                if element == 'NR':
                    element = 0.0
                element = float(element)

                testingFeatureRow.append(element)

        testingFeatureMatrix.append(testingFeatureRow)

    return testingFeatureMatrix

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
timeSeries = parseTrainFileIntoTimeSeries(filename_train)

featureMatrix = getFeatureMatrixGivenTimeSeries(timeSeries)
newFeatureMatrix = featureSelection(featureMatrix)
writeMatrix(filename_featureMatrix, newFeatureMatrix)

labelMatrix = getLabelMatrixGivenTimeSeries(timeSeries)
writeMatrix(filename_labelMatrix, labelMatrix)

# Testing data
testingFeatureMatrix = parseTestFileIntoTestingFeatureMatrix(filename_test)
newTestingFeatureMatrix = featureSelection(testingFeatureMatrix)
writeMatrix(filename_testingFeatureMatrix, newTestingFeatureMatrix)

