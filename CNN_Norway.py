import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cv2 import imread

#import tensorflow as tf
#import os
#tf.random.set_seed(1)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

images = []
for i in range(1730):
    image = imread('/Users/jacob/Documents/Microsoft Visual Studio Code Projects/MachineLearningEngineers/data/leafes_256x170/leaf_'+str(i)+'.jpg')
    images.append(image)

data = np.array(images)
data = data / 255.0
print(data.shape)
labels_train=np.load('/Users/jacob/Documents/Microsoft Visual Studio Code Projects/MachineLearningEngineers/data/labels.npy')
labels = ['healthy', 'rust', 'scab']

from keras.utils import to_categorical
labels_train = to_categorical(labels_train, num_classes = 3)
print(labels_train.shape)

X_train, X_val, y_train, y_val = train_test_split(data, labels_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization
from keras.optimizers import RMSprop,Adam, Nadam, SGD
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(BatchNormalization(input_shape = (170,256,3)))
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (170,256,3), kernel_initializer='HeNormal',kernel_regularizer='l1'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', kernel_initializer='HeNormal',kernel_regularizer='l1'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', kernel_initializer='HeNormal',kernel_regularizer='l1'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', kernel_initializer='HeNormal',kernel_regularizer='l1'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation = "relu", kernel_initializer='HeNormal',kernel_regularizer='l1'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu", kernel_initializer='HeNormal', kernel_regularizer='l1'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(3, activation = "softmax"))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 200  
batch_size = 60

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False, 
        rotation_range=5,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False)  

datagen.fit(X_train)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=8)

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val, y_val), steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[early_stopping])

print("Accuracy of the model is --> " , model.evaluate(X_val, y_val, batch_size=batch_size)[1]*100 , "%")
print("Loss of the model is --> " , model.evaluate(X_val, y_val, batch_size=batch_size)[0])

plt.figure()
plt.plot(history.history["loss"],label = "Train Loss", color = "black")
plt.plot(history.history["val_loss"],label = "Validation Loss", color = "darkred", marker = "+", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model Loss", color = "darkred", size = 13)
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["accuracy"],label = "Train Accuracy", color = "black")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy", color = "darkred", marker = "+", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model Accuracy", color = "darkred", size = 13)
plt.legend()
plt.show()

import matplotlib

# We make predictions using the model we have created.
Y_pred = model.predict(X_val)
# argmax = To briefly mention it, it will give the index of the value with the highest value.
Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# We do the same for the y_val values. because we will compare these values. 
Y_true = np.argmax(y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,linecolor="gray", fmt= '.1f',ax=ax, xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label", color = "blue")
plt.ylabel("True Label", color = "green")
plt.title("Confusion Matrix", color = "darkred", size = 15)
plt.show()
