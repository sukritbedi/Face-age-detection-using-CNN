import scipy.io
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import keras
from keras.preprocessing import image

def datenum_to_datetime(datenum):
       days = datenum % 1
       hours = days % 1 * 24
       minutes = hours % 1 * 60
       seconds = minutes % 1 * 60
       exact_date = datetime.fromordinal(int(datenum)) + timedelta(days=int(days)) + timedelta(hours=int(hours)) + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) - timedelta(days=366)
       return exact_date.year

def getImagePixels(image_path):
       img = image.load_img("wiki_crop/%s" % image_path[0], grayscale=False, target_size=target_size)
       x = image.img_to_array(img).reshape(1, -1)[0]
       #x = preprocess_input(x)
       return x

mat = scipy.io.loadmat('wiki_crop/wiki.mat')

instances = mat['wiki'][0][0][0].shape[1]
 
columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

df = pd.DataFrame(index = range(0,instances), columns = columns)
 
for i in mat:
       if i == "wiki":
              current_array = mat[i][0][0]
for j in range(len(current_array)):
       df[columns[j]] = pd.DataFrame(current_array[j][0])


df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)
df['age'] = df['photo_taken'] - df['date_of_birth']

#remove pictures does not include face
df = df[df['face_score'] != -np.inf]
 
#check threshold
df = df[df['face_score'] >= 3]
 
df = df.drop(['name','face_score','second_face_score','date_of_birth','face_location'], axis=1)

#some guys seem to be greater than 100. some of these are paintings. remove these old guys
df = df[df['age'] <= 100]

#some guys seem to be unborn in the data set
df = df[df['age'] > 0]

histogram_age = df['age'].hist(bins=df['age'].nunique())
histogram_gender = df['gender'].hist(bins=df['gender'].nunique())

target_size = (224, 224)
 
#df['pixels'] = df['full_path'].apply(getImagePixels)

#Design of the neural network
model = keras.models.Sequential()
model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
 
model.add(keras.layers.Convolution2D(4096, (7, 7), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Convolution2D(4096, (1, 1), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Convolution2D(2622, (1, 1)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Activation('softmax'))

model.load_weights('vgg_face_weights.h5')

for layer in model.layers[:-7]:
       layer.trainable = False
 
base_model_output = Sequential()
base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)
 
age_model = Model(inputs=model.input, outputs=base_model_output)
