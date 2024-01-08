import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
#from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images/255.0

model = models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer = 'adam', 
loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy'])
history = model.fit(train_images, train_labels, epochs = 5)


plt.figure(figsize = (8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.plot(history.history['loss'], label = "Loss")
plt.title("Training Accuracy and Loss")
plt.legend()


predictions = model.predict(test_images)
plt.figure(figsize = (10, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap = plt.cm.binary)
    plt.title("Predicted : %d" %np.argmax(predictions[i]))
    plt.axis('off')

plt.show()