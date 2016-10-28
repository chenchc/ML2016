import sys
import csv
import math
from random import shuffle
import random

def parseFileIntoMatrix(filename):
    rawMatrix = []
    with open(filename, 'rb') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            rawMatrix.append(row)
    
    return rawMatrix

def writeMatrix(filename, matrix):
    with open(filename, 'wb') as file:
        writer = csv.writer(file, delimiter = ' ')
        for row in matrix:
            if isinstance(row, float):
                writer.writerow([row])
            else:
                writer.writerow(row)
       
# Parse arguments         
filename_featureMatrix = sys.argv[1]
filename_labelMatrix = sys.argv[2]
filename_output_featureMatrix = sys.argv[3]
filename_output_labelMatrix = sys.argv[4]

# Parse
featureMatrix = parseFileIntoMatrix(filename_featureMatrix)
labelMatrix = parseFileIntoMatrix(filename_labelMatrix)

newFeatureMatrix = []
newLabelMatrix = []
for i in range(len(featureMatrix)):
    select = random.randint(0, len(featureMatrix) - 1)
    newFeatureMatrix.append(featureMatrix[select])
    newLabelMatrix.append(labelMatrix[select])

writeMatrix(filename_output_featureMatrix, newFeatureMatrix)
writeMatrix(filename_output_labelMatrix, newLabelMatrix)
