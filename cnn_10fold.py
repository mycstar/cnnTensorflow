from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
import numpy

#array length
arrayLength=88
seed = 6
numpy.random.seed(seed)
#read file
dataset = numpy.loadtxt("22.csv", delimiter=",")
X = dataset[:,0:arrayLength]
Y = dataset[:,arrayLength]
batch_size = 10
num_classes = 2
epochs = 10
img_x, img_y = 1, 88

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # create model
	x_train = X[train].reshape(X[train].shape[0], img_x, img_y, 1)
	x_test = X[test].reshape(X[test].shape[0], img_x, img_y, 1)
	input_shape = (img_x, img_y, 1)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	
	y_train = keras.utils.to_categorical(Y[train], num_classes)
	y_test = keras.utils.to_categorical(Y[test], num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1),
                 	activation='relu',
                 	input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
	model.add(Conv2D(64, (1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(1, 1)))
	model.add(Flatten())
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy,
              	optimizer=keras.optimizers.Adam(),
              	metrics=['accuracy'])
	class AccuracyHistory(keras.callbacks.Callback):
    		def on_train_begin(self, logs={}):self.acc = []

    		def on_epoch_end(self, batch, logs={}):self.acc.append(logs.get('acc'))	
	history = AccuracyHistory()
	model.fit(x_train, y_train,
          	batch_size=batch_size,
          	epochs=epochs,
          	verbose=1,
          	validation_data=(x_test, y_test),
          	callbacks=[history])
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


