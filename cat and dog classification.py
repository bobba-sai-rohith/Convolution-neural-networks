import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation

# Initialising the CNN model as sequential type
classifier = Sequential()

#relu is used as the activation fuction as it best to use with the convolution layer to remove the -ve values from the image feature maps
#adding convolution layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# adding Max-pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer and Max pooling layers
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening layer : to fatten the data from a 2D matrix form to flat sing array like format to feed it into the neural network
classifier.add(Flatten())

# Full connection layer of neural network with 128 nodes in network
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#output layer with one output node
#sigmoid is used as activation fuction usally for the binary classifications 
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the network designed
classifier.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

#image Preprocessing

from keras.preprocessing.image import ImageDataGenerator
#creating object of ImagedataGenerator for training images rescaling image -normalising it,setting zoom range,sheer range and setting the horizantal and vertical flips
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
                                   
#creating object of ImagedataGenerator for testing images rescaling image -normalising it -you can add horzantal,vertical flips and zoom,sheer ranges if you want which might better accuracy
test_datagen = ImageDataGenerator(rescale = 1./255)


#binary class mode is used as there are only two classes cat and dog
#loading data and appliying above operations by ImagedataGenerator object for train and test images
#target size will take all the images to that petucular size
#batch size is used for upadating the weights after that perticular batch of images

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')


#to train the model
#nb_epoch=no of times model should b trained

classifier.fit_generator(training_set,samples_per_epoch = 8000,nb_epoch = 5,validation_data = test_set,nb_val_samples = 2000)
       
#saving the model
classifier.save('catvsdog--model')




#reloading the model for future use
mod=keras.models.load_model('catvsdog--model')



#predicting the output of the trained model by giving any random image 
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/test_set/cats/cat.4001.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=mod.predict(test_image)
if result[0][0]>=0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)

             
