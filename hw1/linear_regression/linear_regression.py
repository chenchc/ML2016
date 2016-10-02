import sys
import csv

FEATURE_COUNT = 18
DATA_ROW_BEGIN = 1
DATA_COL_BEGIN = 3
DATA_COL_COUNT = 24
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
        for j in range(DATA_COL_BEGIN, DATA_COL_BEGIN + DATA_COL_COUNT):
            element = rawMatrix[i][j]
            if element == 'NR':
                element = 0.0
            element = float(element)

            timeSeries.append([element])

        # Remaining rows of a row set
        for j in range(i + 1, i + FEATURE_COUNT):
            for k in range(DATA_COL_BEGIN, DATA_COL_BEGIN + DATA_COL_COUNT):
                element = rawMatrix[j][k]
                if element == 'NR':
                    element = 0.0
                element = float(element)

                timeSeriesIndex = len(timeSeries) - DATA_COL_COUNT + (k - DATA_COL_BEGIN)
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
       
# Parse arguments         
filename_train = sys.argv[1]

# Train
timeSeries = parseTrainFileIntoTimeSeries(filename_train)
featureMatrix = getFeatureMatrixGivenTimeSeries(timeSeries)
labelMatrix = getLabelMatrixGivenTimeSeries(timeSeries)


