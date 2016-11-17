from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, BatchNormalization

def get():
	model = Sequential()
	model.add(Convolution2D(192, 5, 5, input_shape=(32, 32, 3)))
	model.add(BatchNormalization(epsilon=1e-03))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Convolution2D(256, 1, 1))
	model.add(BatchNormalization(epsilon=1e-03))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Convolution2D(384, 1, 1))
	model.add(BatchNormalization(epsilon=1e-03))
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 1, 1))
	model.add(BatchNormalization(epsilon=1e-03))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3))
	model.add(BatchNormalization(epsilon=1e-03))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(512, init='he_normal'))
	model.add(BatchNormalization(epsilon=1e-03))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, init='he_normal'))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model
