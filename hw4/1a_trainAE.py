import sys;
import string;
import re;
import math;
import numpy;
import gc;
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Activation, Dropout, RepeatVector, Embedding, BatchNormalization
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
			if word in wordVectorDict and word not in LABEL and word not in ["visual", "studio"]:
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
		#if "visual studio" in lineVector[line]:
		#	output[line][wordVectorDict["visual studio"]] = 1;
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

def getLabeledData(featureMatrix, lineVector):
	labeledFeatureMatrix = [];
	labelMatrix = [];
	for row in range(len(featureMatrix)):
		dirty = 0;
		for label in range(len(LABEL)):
			if LABEL[label] in lineVector[row]:
				dirty += 1;
		if dirty != 1:
			continue;
		for label in range(len(LABEL)):
			if LABEL[label] in lineVector[row]:
				labeledFeatureMatrix.append(featureMatrix[row]);
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
#wordVector.append("visual studio");
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
(labeledFeatureMatrix, labelMatrix) = getLabeledData(featureMatrix, lineVector);
classWeight = 1.0 / numpy.bincount(labelMatrix);

# NN configuration
input_img = Input(shape=(featureMatrix.shape[1],));
encoded0 = encoded = Dense(320)(input_img);
encoded = BatchNormalization()(encoded);
encoded = Activation('relu')(encoded);
encoded = Dropout(0.3)(encoded);

encoded1 = encoded = Dense(80)(encoded);
encoded = BatchNormalization()(encoded);

decoded = Dense(320)(encoded);
decoded = BatchNormalization()(decoded);
decoded = Activation('relu')(decoded);
decoded = Dropout(0.3)(decoded);

decoded = Dense(featureMatrix.shape[1])(decoded);

ae = Model(input=input_img, output=decoded);
ae.compile(loss='mean_squared_error', optimizer='adam');

encoder = Model(input=input_img, output=encoded);
encoder.compile(loss='mean_squared_error', optimizer='adam');

encoded0.trainable = False;
encoded1.trainable = False;
classd = Dense(160)(encoded);
classd = Activation('relu')(classd);
classd = Dropout(0.5)(classd);
classd = Dense(20)(classd);
classd = Activation('softmax')(classd);

aednn = Model(input=input_img, output=classd)
aednn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);

# AutoEncoder
ae.load_weights(modelPath + ".ae");

ae.fit(featureMatrix, featureMatrix, batch_size=64, nb_epoch=100, validation_split=0.1, callbacks=[ModelCheckpoint(modelPath + ".ae", monitor='val_loss', save_best_only=True, save_weights_only=True), EarlyStopping(monitor='val_loss', patience=3, mode='min')]);

ae.load_weights(modelPath + ".ae");

# DNN
aednn.fit(labeledFeatureMatrix, to_categorical(labelMatrix, nb_classes=20), class_weight=classWeight, batch_size=64, nb_epoch=100, validation_split=0.1, callbacks=[ModelCheckpoint(modelPath, monitor='val_loss', save_best_only=True), EarlyStopping(monitor='val_loss', patience=3, mode='min')]);

print numpy.bincount(numpy.argmax(aednn.predict(featureMatrix), axis=1));
