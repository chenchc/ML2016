import sys;
import string;
import re;
import math;
import numpy;

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

def getWordVector(lineVector):
	wordSet = set();
	for line in lineVector:
		wordSet.update(line.split(' '));
	return list(wordSet);

def writeVector(vector, path):
	file = open(path, 'w');
	for elem in vector:
		if isinstance(elem, basestring):
			file.write(elem + '\n');
		else:
			file.write(str(elem) + '\n');

def makeDict(wordVector):
	output = {};
	for i in range(len(wordVector)):
		output[wordVector[i]] = i;
	return output;

def extractOccurenceFeatureMatrix(lineVector, wordVectorDict):
	output = numpy.zeros((len(lineVector), len(wordVectorDict)));
	for line in range(len(lineVector)):
		for word in lineVector[line].split(' '):
			i = wordVectorDict[word]
			if i != None:
				output[line][i] = 1.0;
	return output;

def extractIDFOfWords(occurenceMatrix):
	globalTF = numpy.zeros((1, occurenceMatrix.shape[1]));
	for row in occurenceMatrix:
		globalTF += row;
	idf = numpy.zeros((1, occurenceMatrix.shape[1]));
	for col in range(occurenceMatrix.shape[1]):
		idf[0][col] = math.log(occurenceMatrix.shape[0] / globalTF[0][col]);
	return idf;

def extractTFIDFFeatureMatrix(lineVector, wordVectorDict, idf):
	tf = numpy.zeros((len(lineVector), len(wordVectorDict)));
	for line in range(len(lineVector)):
		for word in lineVector[line].split(' '):
			i = wordVectorDict[word];
			if i != None:
				tf[line][i] += 1.0;
	tfidf = tf;
	for row in range(idf.shape[0]):
		for col in range(tfidf.shape[1]):
			tfidf[row][col] *= idf[0][col];
	return tfidf;


# Parse arguments
trainPath = sys.argv[1];
wordsPath = sys.argv[2];
idfPath = sys.argv[3];
modelPath = sys.argv[4];

# Read file to get raw line vectors
lineVector = getRawLineVector(trainPath);

# A bunch of preprocessing
lineVector = pruneCodes(lineVector);
lineVector = prunePunctuation(lineVector);

# Get word vector
wordVector = getWordVector(lineVector);
writeVector(wordVector, wordsPath);
wordVectorDict = makeDict(wordVector);

# Extract feature matrix
occurenceFeatureMatrix = extractOccurenceFeatureMatrix(lineVector, wordVectorDict);
idf = extractIDFOfWords(occurenceFeatureMatrix);
writeVector(idf.tolist()[0], idfPath);
tfidfFeatureMatrix = extractTFIDFFeatureMatrix(lineVector, wordVectorDict, idf);
featureMatrix = numpy.concatenate((occurenceFeatureMatrix, tfidfFeatureMatrix), 1);


