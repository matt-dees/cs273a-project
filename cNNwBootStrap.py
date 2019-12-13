import random
import numpy as np
import tensorflow
import cv2           #pip install opencv-python
import imutils as im #pip install imutils
import matplotlib.pyplot as plt

## model imports ##
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape
import tensorflow.keras as tf

# Load the fashion-mnist pre-shuffled train data and test data. 
#Make sure to have mnist_reader.py in the same directory
import mnist_reader

x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='t10k')
#(x_train, y_train), (x_test, y_test) = tf.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# function to define, compile, and train model
def modelC(x_train, y_train):
    inputs = Input(shape=(784,))
    x = BatchNormalization()(inputs)            #normalize
    x = Dropout(0.2)(x)                         #regularize
    x = Reshape((28, 28, 1))(x)                 #reshape for filters
    x = Conv2D(32, (3,3), activation='relu')(x) #3X3 filters
    x = Dropout(0.5)(x)
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)        #downsampling with 2x2 max pooling layer
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)        #hidden layer with relu activation function
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=tf.losses.categorical_crossentropy,
             optimizer=tf.optimizers.Adadelta(), #adadelta optimizer and cross_entropy loss function
             metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, to_categorical(y_train), epochs=30, shuffle=True, validation_split=0.33)
    return history

# function to plot results
def showHistory(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()
    
#function to visualize data and predictions after training. Not yet perfected    
def visualize(testX,testY,model):
    # initialize our list of output images
    images = []
    # initialize the label names
    labelNames = ["top", "trouser", "pullover", "dress", "coat","sandal", "shirt", "sneaker", "bag", "ankle boot"]
    # randomly select a few testing fashion items
    for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
        # classify the clothing
        probs = model.predict(testX[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = labelNames[prediction[0]]
 
        # extract the image from the testData if using "channels_first"
        # ordering
        #image = (testX[i][0] * 255).astype("uint8")
 
        # otherwise we are using "channels_last" ordering
        #else:
        image = (testX[i] * 255).astype("uint8")
        # initialize the text label color as green (correct)
        color = (0, 255, 0) 
        # otherwise, the class label prediction is incorrect
        if prediction[0] != np.argmax(testY[i]):
            color = (0, 0, 255)
 
        # merge the channels into one image and resize the image from
        # 28x28 to 96x96 so we can better see it and then draw the
        # predicted label on the image
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,color, 2)
 
        # add the image to our list of output images
        images.append(image)
 
    # construct the montage for the images
    montage = im.build_montages(images, (96, 96), (4, 4))[0]
 
    # show the output montage
    cv2.imshow("Fashion MNIST", montage)
    cv2.waitKey(0)
    
#history = modelC(x_train, y_train) #uncomment for non bootstrap result
#showHistory(history) #uncomment for non bootstrap result
#test_result = history.model.evaluate(x_test, to_categorical(y_test)) #uncomment for non bootstrap result
#visualize(x_train, y_train,testmodel)

###BOOTSTRAP
#pip install tf-nightly or pip install tensorflow-gpu
#https://www.tensorflow.org/install/gpu & https://www.youtube.com/watch?v=KZFn0dvPZUQ for guidance in enabling/installing GUP support on tensorflow
#Run with tensorflow gpu (otherwise, it takes too long to compute with CPU)

number_of_bootstraps = 100 #industry recommended is 1k ~ 10k but this would take us too long

test_acc = []
train_acc = []

for i in range(number_of_bootstraps):
    
    quantity = x_train.shape[0] #Use x_train.shape[0] or more if required computing power is present (if not maybe go for 10k?)
    #Speicial thanks to Kunshu Wang for letting us run this step on his computer w/ GEFORCE RTX2080 TI (Other wise it would've taken us days for this step)
    bootstrap_train_index = np.random.choice(x_train.shape[0],quantity)

    bootstrap_x_train = []
    for i in bootstrap_train_index:
        bootstrap = x_train[i]
        bootstrap_x_train.append(bootstrap)
    x_train = np.asarray(bootstrap_x_train)

    bootstrap_y_train = []
    for i in bootstrap_train_index:
        bootstrap = y_train[i]
        bootstrap_y_train.append(bootstrap)
    y_train = np.asarray(bootstrap_y_train)
    history = modelC(x_train, y_train)
    showHistory(history)
    test_result = history.model.evaluate(x_test, to_categorical(y_test))
    test_acc.append(max(history.history['val_acc']))
    train_acc.append(test_result[1])
    
plt.plot(train_acc)
plt.plot(test_acc)
plt.xlabel('# of Bootstraps')
plt.ylabel('Max Accuracy')
plt.legend(['train_accuracy', 'test_accuracy'], loc = 'best')
plt.show()

print('cNN validation train accuracy: Min =',min(train_acc),'Max =',max(train_acc))
cnn_train_mean = np.mean(train_acc)
cnn_train_stderr = np.std(train_acc)/np.sqrt(number_of_bootstraps)
cnn_train_CI = (cnn_train_mean - 1.96*cnn_train_stderr, cnn_train_mean + 1.96*cnn_train_stderr) #95% confidence
print('95% confidence interval of cNN validation train accuracy:', cnn_train_CI)

print('cNN test accuracy: Min =',min(test_acc),'Max =',max(test_acc))
cnn_test_mean = np.mean(test_acc)
cnn_test_stderr = np.std(test_acc)/np.sqrt(number_of_bootstraps)
cnn_test_CI = (cnn_test_mean - 1.96*cnn_test_stderr, cnn_test_mean + 1.96*cnn_test_stderr) #95% confidence
print('95% confidence interval of cNN test accuracy:', cnn_test_CI)
###BOOTSTRAP
