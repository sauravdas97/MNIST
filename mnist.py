"""
Original file is located at
    https://colab.research.google.com/drive/1u_S2kf_PGiWhKZyakfLaD9hoO-1An-68

Deep Neural Network for MNIST Classification
>The dataset has 70,000 images (28x28 pixels) of handwritten digits.
>The target is to develop an algorithm that detects which digit is written.
>This is a classification problem having 10 classes (0 to 9 digits).
We will build a neural network with 2 layers.

Action Plan
* Prepare Data and Preprocess it. Create training, validation and test.
* Outline model and select the activation function
* Set the appropirate optimizer and loss function
* Make it learn
* Test Accuracy

##Relevant Packages
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds


mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised = True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples # returns float total number of validation data
num_validation_samples = tf.cast(num_validation_samples, tf.int64) # returns total number of validation data

num_test_samples = mnist_info.splits['test'].num_examples # returns float
num_test_samples = tf.cast(num_validation_samples, tf.int64) # returns total number of validation data

"""Scaling
>Define a function called: scale, that will take an MNIST image and its label we make sure the value is a float.
Since the possible values for the inputs are 0 to 255, if we divide each element by 255, we would get the desired result -> all elements will be between 0 and 1.
"""

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image/=255.
  return image, label


scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

"""Shuffle data
We need to shuffle data to get random dataset
* this BUFFER_SIZE parameter is here for cases when we're dealing with enormous datasets
* then we can't shuffle the whole dataset in one go because we can't fit it all in memory
* so instead TF only stores BUFFER_SIZE samples in memory at a time and shuffles them
* BUFFER_SIZE in between - a computational optimization to approximate uniform shuffling
"""

BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

# creating the validation dataset taking form the train dataset
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

"""Mini Batch Gradient Descent
Mini-batch gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients.
Thus, instead of calclating the error and updating the weights every time, we can update it batch by batch thus it will decrease the overall time to compute, because we updatad the values in a batch rather that updating it one by one.
"""

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validations_targets = next(iter(validation_data))

"""Model

Outline the model
Here, we will give the input layer size and output layer size.

* Here, each observation is 28x28x1 pixels, therefore it is a tensor of rank 3, and we don't know how to feed such input into our net, so we must flatten the images.
TF 'Flatten' method takes 28x28x1 tensor and orders it into (781,1) vector
or (28x28x1,) = (784,) vector
This is a 2 layer structure of a deep neural network so there is one input layer, 2 hidden layers and one output layer.
"""

input_size = 784
output_size = 10
hidden_layer_size = 150 # the number of nodes in one hidden

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape = (28,28,1)),
  tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # first hidden layer
  tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # second hidden layer
  tf.keras.layers.Dense(output_size, activation='softmax')
])

"""Optimizer
* So here sprse_categorical_crossentropy comes with one hot encoding
* metric is a function that is used to judge the performance of your model
"""

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

"""###Training
What happens inside an epoch
* At the start of each epoch the training loass is set to 0
* The algorithm will iterate over the preset number of batches all form the  train_data
* The weights and biases will be updated as many times there are batches
* We will get back the loss function indication how the training is going
* We will also see the training accuracy, because of the metrics parameter  defined in .fit() method
* At the end of epoch the algo will forward propagate the whoe validation set.
* Once we reach the max number of epochs the training will be over
"""

NUM_EPOCHS = 5
model.fit(train_data, epochs = NUM_EPOCHS, validation_data=(validation_inputs, validations_targets), verbose = 2)

"""#Testing
.evaluate(test_data) : it returns a list containing the loss and accuracy
"""

test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss : {0:.2f} \n Test accuracy : {1:.2f}%'.format(test_loss, test_accuracy*100))