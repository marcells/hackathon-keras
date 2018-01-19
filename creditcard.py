import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from imblearn.under_sampling import RandomUnderSampler
def recall(y_true, y_pred):
   """Recall metric.

   Only computes a batch-wise average of recall.

   Computes the recall, a metric for multi-label classification of
   how many relevant items are selected.
   """
   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
   possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
   recall = true_positives / (possible_positives + K.epsilon())
   return recall


# 1. Read data
data = pd.read_csv("creditcard.csv").values
np.random.shuffle(data)


rus = RandomUnderSampler(return_indices=False, random_state=42)

dataLength = len(data)
trainLength = int(dataLength*0.8)
trainData = data[:trainLength]
testData = data[trainLength:]
validateData = trainData[:int(trainLength*0.2)]
trainData = trainData[int(trainLength*0.2):]

# Split data
#xValidateFraudData = validateData[:,0:30]
#yValidateFraudData = validateData[:,30]

xValidateData = validateData[:,0:30]
yValidateData = validateData[:,30]

xTestData = testData[:,0:30]
yTestData = testData[:,30]

xTrainData = trainData[:,0:30]
yTrainData = trainData[:,30]


xTrainData, yTrainData = rus.fit_sample(xTrainData, yTrainData)

print(np.array(xTrainData).shape)
print(np.array(yTrainData).shape)

# 2. Define Model
model = Sequential()
model.add(Dense(30, input_dim=30))#, activation='sigmoid'))
#model.add(Dense(50, activation='sigmoid'))
model.add(Dense(60))#, activation='sigmoid'))
model.add(Dense(5))#, activation='sigmoid'))
#model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#model = Sequential()
#model.add(Dense(30, input_shape =(30, )))
#model.add(Activation('relu'))
#model.add(Dense(1))
#model.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# 3. Train Model
model.fit(xTrainData, yTrainData, validation_data=(xValidateData, yValidateData), epochs=20, batch_size=16)

# evaluate the model
scores = model.evaluate(xTestData, yTestData)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

yPredicted = []
for a in model.predict(xTestData):
    if(a > 0):
        yPredicted.append(1)
    else:
        yPredicted.append(0)

score = recall_score(yTestData, yPredicted, average=None)
print(score)