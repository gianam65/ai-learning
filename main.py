#1. load du lieu va chia Train, Val va Test
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy as np
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

X = dataset[:, 0:8]
y = dataset[:, 8]

X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size= 0.2)

# #2. Xay model
# model = Sequential()
# model.add(Dense(16, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# #3. sumary model
# model.summary()
#
# #4. compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# #5. train model
# model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val))
#
# #6. save model
# model.save("mymodel.h5")

#7. load model
model = load_model("mymodel.h5")

loss, acc = model.evaluate(X_test, y_test)
print("loss", loss)
print("acc", acc)

X_new = X_test[10]
y_new = y_test[10]

X_new = np.expand_dims(X_new, axis=0)
y_predict = model.predict(X_new)
print("Du doan: ", y_predict)
print("Thuc te: ", y_new)

