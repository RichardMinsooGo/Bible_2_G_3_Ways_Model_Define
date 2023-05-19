'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

'''
D2. Load Cifar10 data / Only for Toy Project
'''

# print(tf.__version__)
cifar10 = tf.keras.datasets.cifar10

# load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Change data type as float. If it is int type, it might cause error
'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

# One-Hot Encoding
from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

'''
D4. EDA(? / Exploratory data analysis)
'''
import matplotlib.pyplot as plt

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

'''
D5. Build dataset
'''
batch_size = 100
# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
# if it is RGB type, type shall be [image_size, image_size, 3]
# For MNIST or Fashion MNIST, it need to reshape
# X_train = X_train[..., tf.newaxis]
# X_test = X_test[..., tf.newaxis]

print(X_train.shape)
    
# It fills data as much as the input buffer_size and randomly samples and replaces it with new data.
# Perfect shuffling requires a buffer size greater than or equal to the total size of the data set.
# If you use a buffer_size smaller than the small number of data, 
# random shuffling occurs within the data as much as the initially set buffer_size.

shuffle_size = 100000

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)).shuffle(shuffle_size).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np

'''
M2. Set Hyperparameters
'''

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 30
learning_rate = 0.001

'''
M3. Build NN model
'''
from tensorflow.keras import Input
# in the case of Keras or TF2, type shall be [image_size, image_size, 3] for RGB
def create_model():
    inputs = Input(shape=(32, 32, 3))
    conv1      = Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    pool1      = MaxPool2D(padding='SAME')(conv1)
    conv2      = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(pool1)
    pool2      = MaxPool2D(padding='SAME')(conv2)
    pool3_flat = Flatten()(pool2)
    dense4     = Dense(units=128, activation=tf.nn.relu)(pool3_flat)
    drop4      = Dropout(rate=0.4)(dense4)
    
    logits = Dense(10, activation='softmax')(drop4)
    
    model = Model(inputs=inputs, outputs=logits)

    return model

model = create_model()

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_shapes.png', show_shapes=True)

'''
M4. Optimizer
'''

optimizer = tf.keras.optimizers.Adam()

'''
M5. Define Loss Function
'''

criterion = losses.CategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = metrics.Mean(name='test_loss')
test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

'''
M6. Define train loop
'''

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = criterion(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    
'''
M7. Define validation / test loop
'''

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = criterion(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)

'''
M8. Define Episode / each step process
'''

from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for images, labels in train_ds:
            
            train_step(images, labels)
            
            loss_val= train_loss.result()
            acc     = train_accuracy.result()*100
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
            
'''
M9. Model evaluation
'''
with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
    test_losses = []
    test_accuracies = []
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        
        loss_val= test_loss.result()
        acc     = test_accuracy.result()*100

        test_losses.append(loss_val)
        test_accuracies.append(acc)

        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
