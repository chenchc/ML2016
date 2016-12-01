import sys;
import string;
import re;
import math;
import numpy;
import gc;
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Activation, Dropout
from keras import regularizers
from keras.utils.np_utils import to_categorical

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

def getWordVector(lineVector):
	wordSet = set();
	for line in lineVector:
		wordSet.update(line.split(' '));
	return list(wordSet);

def pruneWordVector(wordVectorDict, testLineVector):
	wordSet = set();
	for line in testLineVector:
		for word in line.split(' '):
			if word in wordVectorDict:
				wordSet.add(word);
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
	output = numpy.zeros((len(lineVector), len(wordVectorDict)), dtype=numpy.int8);
	for line in range(len(lineVector)):
		if "visual studio" in lineVector[line]:
			output[line][wordVectorDict["visual studio"]] = 1;
		for word in lineVector[line].split(' '):
			if word in wordVectorDict:
				i = wordVectorDict[word];
				output[line][i] = 1;
	return output;

def extractIDFOfWords(occurenceMatrix):
	globalTF = numpy.zeros((1, occurenceMatrix.shape[1]), dtype=numpy.float32);
	for row in occurenceMatrix:
		globalTF += row;
	idf = numpy.zeros((1, occurenceMatrix.shape[1]), dtype=numpy.float32);
	for col in range(occurenceMatrix.shape[1]):
		idf[0][col] = math.log(occurenceMatrix.shape[0] / globalTF[0][col]);
	return idf;

def extractTFIDFFeatureMatrix(lineVector, wordVectorDict, idf):
	tf = numpy.zeros((len(lineVector), len(wordVectorDict)), dtype=numpy.float32);
	for line in range(len(lineVector)):
		if "visual studio" in lineVector[line]:
			tf[line][wordVectorDict["visual studio"]] += 1.0;
		for word in lineVector[line].split(' '):
			if word in wordVectorDict:
				i = wordVectorDict[word];
				tf[line][i] += 1.0;
	tfidf = tf;
	for row in range(idf.shape[0]):
		for col in range(tfidf.shape[1]):
			tfidf[row][col] *= idf[0][col];
	return tfidf;

def getLabeledData(featureMatrix, wordVectorDict):
	labeledFeatureMatrix = [];
	labelMatrix = [];
	for row in featureMatrix:
		for label in range(len(LABEL)):
			if row[wordVectorDict[LABEL[label]]] > 0.0:
				labeledFeatureMatrix.append(row);
				labelMatrix.append(label);
	return (numpy.array(labeledFeatureMatrix, dtype=numpy.float32), numpy.array(labelMatrix, dtype=numpy.int8));

# Parse arguments
trainPath = sys.argv[1];
testTitlePath = sys.argv[2];
wordsPath = sys.argv[3];
idfPath = sys.argv[4];
modelPath = sys.argv[5];

# Read file to get raw line vectors
lineVector = getRawLineVector(trainPath);

# A bunch of preprocessing
lineVector = pruneCodes(lineVector);
lineVector = prunePunctuation(lineVector);

# Read test file
testLineVector = getRawLineVector(testTitlePath);
testLineVector = prunePunctuation(testLineVector);

# Get word vector
wordVector = getWordVector(lineVector);
wordVectorDict = makeDict(wordVector);
wordVector = pruneWordVector(wordVectorDict, testLineVector);
wordVector.append("visual studio");
wordVectorDict = makeDict(wordVector);
writeVector(wordVector, wordsPath);


# Extract feature matrix
occurenceFeatureMatrix = extractOccurenceFeatureMatrix(lineVector, wordVectorDict);
idf = extractIDFOfWords(occurenceFeatureMatrix);
writeVector(idf.tolist()[0], idfPath);
tfidfFeatureMatrix = extractTFIDFFeatureMatrix(lineVector, wordVectorDict, idf);
#featureMatrix = numpy.concatenate((occurenceFeatureMatrix, tfidfFeatureMatrix), 1);
#featureMatrix = numpy.reshape(occurenceFeatureMatrix, (occurenceFeatureMatrix.shape[0], occurenceFeatureMatrix.shape[1] + tfidfFeatureMatrix.shape[1]));
featureMatrix = tfidfFeatureMatrix;

# Labeled data?
(labeledFeatureMatrix, labelMatrix) = getLabeledData(featureMatrix, wordVectorDict);

# AutoEncoder
input_img = Input(shape=(featureMatrix.shape[1],));
encoded = Dense(80, activation='relu')(input_img);
encoded = Dense(40, activation=None, activity_regularizer=regularizers.activity_l1(10e-5))(input_img);

decoded = Dense(80, activation='relu')(encoded);
decoded = Dense(featureMatrix.shape[1], activation='linear')(encoded);

classd = Dense(20, activation='softmax')(encoded)

ae = Model(input=input_img, output=decoded);
ae.compile(loss='mean_squared_error', optimizer='adam');

encoder = Model(input=input_img, output=encoded);
encoder.compile(loss='mean_squared_error', optimizer='adam');

aednn = Model(input=input_img, output=classd)
aednn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);

#ae.fit(featureMatrix, featureMatrix, batch_size=64, nb_epoch=50, validation_split=0.1, callbacks=[ModelCheckpoint(modelPath, monitor='val_loss', save_best_only=True), EarlyStopping(monitor='val_loss', patience=2, mode='min')]);

#numpy.set_printoptions(threshold='nan');
#result = encoder.predict(featureMatrix, batch_size=128, verbose=1);
#for i in range(100):
#	print numpy.argmax(result[i]);
#	print result[i];
#print numpy.bincount(numpy.argmax(result, axis=1));

aednn.fit(labeledFeatureMatrix, to_categorical(labelMatrix, nb_classes=20), batch_size=64, nb_epoch=50, validation_split=0.1, callbacks=[ModelCheckpoint(modelPath, monitor='val_loss', save_best_only=True), EarlyStopping(monitor='val_loss', patience=2, mode='min')]);
