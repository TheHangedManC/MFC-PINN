import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input, Conv2D, MaxPooling2D, Permute, Reshape, add, Dense, Dropout, Flatten, RepeatVector, Permute, multiply, Lambda, concatenate, Activation
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


IMAGE_DIMS = (224, 224, 3)

cwt_path1 = './dataset/cwt/'
cwt_txt1 = './dataset/train15-1.txt'

cwt_path2 = './dataset/cwt/'
cwt_txt2 = './dataset/val15-1.txt'

cwt_path3 = './dataset/cwt/'
cwt_txt3 = './dataset/test15-1.txt'

x_train_savepath = './dataset/15-1/x_train.npy'
y1_train_savepath = './dataset/15-1/y1_train.npy'
y2_train_savepath = './dataset/15-1/y2_train.npy'

x_test_savepath = './dataset/15-1/x_test.npy'
y1_test_savepath = './dataset/15-1/y1_test.npy'
y2_test_savepath = './dataset/15-1/y2_test.npy'

delta_t_train_savepath = './dataset/15-1/delta_t_train.npy'
delta_t_test_savepath = './dataset/15-1/delta_t_test.npy'

x_test_savepathc = './dataset/15-1/x_testc.npy'
y1_test_savepathc = './dataset/15-1/y1_testc.npy'
y2_test_savepathc = './dataset/15-1/y2_testc.npy'
delta_t_test_savepathc = './dataset/15-1/delta_t_testc.npy'

def generateds2(path, txt):
    f = open(txt, 'r')  # open txt
    contents = f.readlines()  # read all rows
    f.close()  # close txt
    x, y1_, y2_, delta_t1, delta_t2, delta_t3 = [], [], [], [], [], []

    for content in contents:
        value = content.split()
        img_path = path + value[0]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        img = img_to_array(img)
        img = img / 255.
        x.append(img)
        y1_.append((value[1]))
        y2_.append((value[2]))
        delta_t1.append((value[3]))
        delta_t2.append((value[4]))
        delta_t3.append((value[5]))
        print('loading : ' + content)
    delta_t = np.stack((delta_t1, delta_t2, delta_t3), axis=-1)
    x = np.array(x)
    y1_ = np.array(y1_)
    y1_ = y1_.astype(np.float32)
    y2_ = np.array(y2_)
    y2_ = y2_.astype(np.float32)
    delta_t = np.array(delta_t)
    delta_t = delta_t.astype(np.float32)

    return x, y1_, y2_, delta_t

if os.path.exists(x_train_savepath) and os.path.exists(y1_train_savepath) and os.path.exists(y2_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y1_test_savepath) and os.path.exists(y2_test_savepath) and os.path.exists(delta_t_train_savepath) and os.path.exists(delta_t_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y1_train = np.load(y1_train_savepath)
    y2_train = np.load(y2_train_savepath)

    x_test_save = np.load(x_test_savepath)
    y1_test = np.load(y1_test_savepath)
    y2_test = np.load(y2_test_savepath)

    x_train = np.reshape(x_train_save, (len(x_train_save), IMAGE_DIMS[1], IMAGE_DIMS[0], 3))
    x_test = np.reshape(x_test_save, (len(x_test_save), IMAGE_DIMS[1], IMAGE_DIMS[0], 3))
    delta_t_train = np.load(delta_t_train_savepath)
    delta_t_test = np.load(delta_t_test_savepath)

    x_test_savec = np.load(x_test_savepathc)
    y1_testc = np.load(y1_test_savepathc)
    y2_testc = np.load(y2_test_savepathc)
    x_testc = np.reshape(x_test_savec, (len(x_test_savec), IMAGE_DIMS[1], IMAGE_DIMS[0], 3))
    delta_t_testc = np.load(delta_t_test_savepathc)

else:
    print('-------------Generate Datasets-----------------')
    x_train, y1_train, y2_train, delta_t_train = generateds2(cwt_path1, cwt_txt1)

    x_testc, y1_testc, y2_testc, delta_t_testc = generateds2(cwt_path2, cwt_txt2)

    x_test, y1_test, y2_test, delta_t_test = generateds2(cwt_path3, cwt_txt3)
    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y1_train_savepath, y1_train)
    np.save(y2_train_savepath, y2_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y1_test_savepath, y1_test)
    np.save(y2_test_savepath, y2_test)
    np.save(delta_t_train_savepath, delta_t_train)
    np.save(delta_t_test_savepath, delta_t_test)
    x_test_savec = np.reshape(x_testc, (len(x_testc), -1))
    np.save(x_test_savepathc, x_test_savec)
    np.save(y1_test_savepathc, y1_testc)
    np.save(y2_test_savepathc, y2_testc)
    np.save(delta_t_test_savepathc, delta_t_testc)

x_val = x_testc
y1_val = y1_testc
y2_val = y2_testc
delta_t_val = delta_t_testc

def custom_loss(y_true, y_pred):

    loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]) + tf.square(y_true[:, 1] - y_pred[:, 1]))

    return loss


input_layer = Input(shape=IMAGE_DIMS)

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
flatten = Flatten()(pool4)
dense1 = Dense(4096, activation='relu')(flatten)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(4096, activation='relu')(dropout1)

output = Dense(2, activation='linear')(dense2)
model = Model(inputs=input_layer, outputs=output, name='cnn_model')
model.compile(optimizer=Adam(lr=0.001), loss=custom_loss, metrics=['mae'])
y_true_train = np.concatenate((y1_train[:, np.newaxis], y2_train[:, np.newaxis]), axis=1)
y_true_val = np.concatenate((y1_val[:, np.newaxis], y2_val[:, np.newaxis]), axis=1)
history=model.fit(x_train, y_true_train, batch_size=32, epochs=300, validation_data=(x_val, y_true_val), validation_freq=1)
model_json = model.to_json()
with open("./result/15-1/cnn/cwt_model.json",'w')as json_file:
    json_file.write(model_json)
model.save_weights('./result/15-1/cnn/cwt_model.h')
model.summary()
with open("./result/15-1/cnn/cwt.txt", 'w') as f1:
    for i in range(0, len(history.history['loss'])):
        a1 = history.history['loss'][i]
        b1 = history.history['val_loss'][i]
        a2 = history.history['mae'][i]
        b2 = history.history['val_mae'][i]
        f1.write(str(a1) + ' ' + str(a2) + ' ' + str(b1) + ' ' + str(b2) + "\n")
        
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
json_file = open("./result/15-1/cnn/model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./result/15-1/cnn/model.h")

cnn_predicted_y = loaded_model.predict(x_test)
pinn_predicted_y1 = np.array(cnn_predicted_y[:, 0:1])*10
pinn_predicted_y2 = np.array(cnn_predicted_y[:, 1:2])*10
y1_test=y1_test*10
y2_test=y2_test*10

mse_y1 = mean_squared_error(y1_test, pinn_predicted_y1)
mse_y2 = mean_squared_error(y2_test, pinn_predicted_y2)
print(mse_y1)
print(mse_y2)
mae_y1 = mean_absolute_error(y1_test, pinn_predicted_y1)
mae_y2 = mean_absolute_error(y2_test, pinn_predicted_y2)
print(mae_y1)
print(mae_y2)
r2_y1 = r2_score(y1_test, pinn_predicted_y1)
r2_y2 = r2_score(y2_test, pinn_predicted_y2)
print(r2_y1)
print(r2_y2)