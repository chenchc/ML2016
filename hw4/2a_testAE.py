import sys;
import string;
import re;
import math;
import numpy;
import gc;
import csv;
from keras.models import Sequential, load_model

LABEL = ["wordpress", "oracle", "svn", "apache", "excel", "matlab", "visual studio", "cocoa", "osx", "bash", "spring", "hibernate", "scala", "sharepoint", "ajax", "qt", "drupal", "linq", "haskell", "magento"];

def getRawLineVector(filename):
	file = open(filename, 'r');
	rawLineVector = [];
	for line in file:
		rawLineVector.append(line);
	return rawLineVector;

def pruneCodes(input):
	output = []
	for line in input:
		if line[0].isupper():
			output.append(line);
	return output;

def prunePunctuation(input):
	output = []
	for line in input:
		line = re.sub("[ \r\t\n]", " ", line);
		line = "".join(re.findall("[a-zA-Z ]*", line));
		line = string.lower(line);
		line = re.sub(" +", " ", line);
		output.append(line);
	return output;

def readWordVector(wordPath):
	wordVector = [];
	file = open(wordPath, 'r');
	for line in file:
		wordVector.append(re.sub("\n", "", line));
	return wordVector;

def makeDict(wordVector):
	output = {};
	for i in range(len(wordVector)):
		output[wordVector[i]] = i;
	return output;

def extractOccurenceFeatureMatrix(lineVector, wordVectorDict):
	output = numpy.zeros((len(lineVector), len(wordVectorDict)), dtype=numpy.int8);
	for line in range(len(lineVector)):
		#if "visual studio" in lineVector[line]:
		#	output[line][wordVectorDict["visual studio"]] = 1;
		for word in lineVector[line].split(' '):
			if word in wordVectorDict:
				i = wordVectorDict[word];
				output[line][i] = 1;
	return output;

def readIDF(idfPath):
	file = open(idfPath, 'r');
	idf = []
	for line in file:
		idf.append(float(line));
	return numpy.array(idf, ndmin=2, dtype=numpy.float32);

def extractTFIDFFeatureMatrix(lineVector, wordVectorDict, idf):
	tf = numpy.zeros((len(lineVector), len(wordVectorDict)), dtype=numpy.float32);
	for line in range(len(lineVector)):
		#if "visual studio" in lineVector[line]:
		#	tf[line][wordVectorDict["visual studio"]] += 1.0;
		for word in lineVector[line].split(' '):
			if word in wordVectorDict:
				i = wordVectorDict[word];
				tf[line][i] += 1.0;
	tfidf = tf;
	for row in range(idf.shape[0]):
		for col in range(tfidf.shape[1]):
			tfidf[row][col] *= idf[0][col];
	return tfidf;


# Parse arguments
testTitlePath = sys.argv[1];
checkIndexPath = sys.argv[2];
wordsPath = sys.argv[3];
idfPath = sys.argv[4];
modelPath = sys.argv[5];
submissionPath = sys.argv[6];

# Read test file
testLineVector = getRawLineVector(testTitlePath);
testLineVector = prunePunctuation(testLineVector);

# Read word vector
wordVector = readWordVector(wordsPath);
wordVectorDict = makeDict(wordVector);

# Extract feature matrix
occurenceFeatureMatrix = extractOccurenceFeatureMatrix(testLineVector, wordVectorDict);
idf = readIDF(idfPath);
tfidfFeatureMatrix = extractTFIDFFeatureMatrix(testLineVector, wordVectorDict, idf);
#featureMatrix = numpy.concatenate((occurenceFeatureMatrix, tfidfFeatureMatrix), 1);
#featureMatrix = numpy.reshape(occurenceFeatureMatrix, (occurenceFeatureMatrix.shape[0], occurenceFeatureMatrix.shape[1] + tfidfFeatureMatrix.shape[1]));
featureMatrix = tfidfFeatureMatrix;

# Read model & predict
model = load_model(modelPath);
predict = model.predict(featureMatrix, verbose=1);
predictClass = numpy.argmax(predict, axis=1);
print numpy.bincount(predictClass);

for i in range(predictClass.shape[0]):
	dirty = 0;
	for label in range(len(LABEL)):
		if LABEL[label] in testLineVector[i]:
			dirty += 1;
		if dirty != 1:
			continue;
	for label in range(len(LABEL)):
		if LABEL[label] in testLineVector[i]:
			predictClass[i] = label;
print numpy.bincount(predictClass);

# Read check index
infile = open(checkIndexPath, 'rb');
reader = csv.reader(infile, delimiter=',');
rawCheckIndex = [];
for row in reader:
	rawCheckIndex.append(row);

x = [];
y = [];
for i in range(len(rawCheckIndex) - 1):
	x.append(int(rawCheckIndex[i + 1][1]));
	y.append(int(rawCheckIndex[i + 1][2]));

outfile = open(submissionPath, 'wb');
writer = csv.writer(outfile, delimiter=',');
writer.writerow(['ID', 'Ans']);
for i in range(len(x)):
	if predictClass[x[i]] == predictClass[y[i]]:
		writer.writerow([i, 1]);
	else:
		writer.writerow([i, 0]);
		
