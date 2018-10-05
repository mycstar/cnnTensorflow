from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy

#array length
arrayLength=8
seed = 6
numpy.random.seed(seed)
#read file
dataset = numpy.loadtxt("leukemia.csv", delimiter=",")
X = dataset[:,0:arrayLength]
Y = dataset[:,arrayLength]
batch_size = 10
num_classes = 2
epochs = 10
img_x, img_y = 1, 8

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

input_shape = (1, 8, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,  optimizer=keras.optimizers.Adam(),  metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}): self.acc = []
  def on_epoch_end(self, batch, logs={}): self.acc.append(logs.get('acc'))

history = AccuracyHistory()
scoring = {'prec_macro': 'precision_macro', 'rec_micro': make_scorer(recall_score, average='macro')}

scores = cross_validate(model, X, Y, scoring=scoring, cv=kfold)

