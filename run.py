import keras
import numpy as np
import scipy.io
import pandas as pd
from datetime import datetime, timedelta

def datenum_to_datetime(datenum):
       days = datenum % 1
       hours = days % 1 * 24
       minutes = hours % 1 * 60
       seconds = minutes % 1 * 60
       exact_date = datetime.fromordinal(int(datenum)) + timedelta(days=int(days)) + timedelta(hours=int(hours)) + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) - timedelta(days=366)
       return exact_date.year

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
 
base_model_output = keras.models.Sequential()
base_model_output = keras.layers.Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = keras.layers.Flatten()(base_model_output)
base_model_output = keras.layers.Activation('softmax')(base_model_output)
 
age_model = keras.models.Model(inputs=model.input, outputs=base_model_output)

age_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
 
checkpointer = keras.callbacks.ModelCheckpoint(filepath='age_model.hdf5', monitor = "val_loss", verbose=1, save_best_only=True, mode = 'auto')
 
scores = []
epochs = 250; batch_size = 256
 
for i in range(epochs):
       print("epoch ",i)
 
       ix_train = np.random.choice(train_x.shape[0], size=batch_size)
        
       score = age_model.fit(train_x[ix_train], train_y[ix_train], epochs=1, validation_data=(test_x, test_y), callbacks=[checkpointer])
        
       scores.append(score)


