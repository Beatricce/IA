## Building the model
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(13,activation="softmax"))


#model.summary()


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])



from imutils import paths
import os , random
import cv2
from keras.preprocessing import image as Kimage
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np


imagePaths = []

for i in range(15):
    if( i == 3 or i == 4 or i == 6 or i == 8 or i == 10 or i == 12):
        imagePaths.append(list(paths.list_images('numbers/number_'+str(i)+'/')))

data = []
labels = []
j = 0
for i in range(len(imagePaths)):
        for imagePath in imagePaths[i]:
            image = Image.open(imagePath)
            image = image.resize((28,28))
            image = image.convert('L')
            image = np.array(image)
            image = image.reshape(28, 28,1)
            data.append(image)
            if(i==0):
                j=3
            if(i==1):
                j=4
            if(i==2):
                j=6
            if(i==3):
                j=8
            if(i==4):
                j=10
            if(i==5):
                j=12
            label = str(j)
            labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


print("data matrix size: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
print("data shape: ", data.shape)
print("data shape: ", labels.shape)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, random_state=42, shuffle=True)

model.fit(trainX, trainY,epochs=12, validation_data=(testX, testY))

## Importing testrom PIL import Image
import numpy as np
from PIL import Image

#image = Image.open("teste.png")  #3
image = Image.open("numbers/number_10/number_149.png")  #3
image2 = Image.open("numbers/discard_3/number_221.png")  #6
#image2 = Image.open("numbers/number_8/number_327.png")  #8
# plt.imshow(image2)

image = image.resize((28, 28))
image = image.convert('L')

image = np.array(image, dtype="float") / 255.0
image2 = image2.resize((28, 28))
image2 = image2.convert('L')
plt.imshow(image2)
image2 = np.array(image2, dtype="float") / 255.0
image = image.reshape(1, 28, 28, 1)
image2 = image2.reshape(1, 28, 28, 1)


pred = model.predict(image2)
pred.argmax()