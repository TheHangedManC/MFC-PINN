import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input, Conv2D, MaxPooling2D, Permute, Reshape, add, Dense, Dropout, Flatten, RepeatVector, Permute, multiply, Lambda, concatenate, Activation
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMAGE_DIMS = (224, 224, 3)

cwt_path = './dataset/cwt/'
cwt_txt = './dataset/15.txt'

x_train_savepath = './dataset/15-2/x_train.npy'
y1_train_savepath = './dataset/15-2/y1_train.npy'
y2_train_savepath = './dataset/15-2/y2_train.npy'

x_test_savepath = './dataset/15-2/x_test.npy'
y1_test_savepath = './dataset/15-2/y1_test.npy'
y2_test_savepath = './dataset/15-2/y2_test.npy'

delta_t_train_savepath = './dataset/15-2/delta_t_train.npy'
delta_t_test_savepath = './dataset/15-2/delta_t_test.npy'

def generateds(path, txt):
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
    x1, x2, y1_1, y1_2 = train_test_split(x, y1_, test_size=0.2, random_state=456)
    x1, x2, y2_1, y2_2 = train_test_split(x, y2_, test_size=0.2, random_state=456)
    delta_t1, delta_t2, y1_1, y1_2 = train_test_split(delta_t, y1_, test_size=0.2, random_state=456)

    return x1, y1_1, y2_1,x2, y1_2, y2_2, delta_t1, delta_t2

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
else:
    print('-------------Generate Datasets-----------------')
    x_train, y1_train, y2_train, x_test, y1_test, y2_test, delta_t_train, delta_t_test = generateds(cwt_path, cwt_txt)

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

x_val_size = len(x_test) // 2
x_val = x_test[:x_val_size]
x_test = x_test[x_val_size:]

y1_val_size = len(y1_test) // 2
y1_val = y1_test[:y1_val_size]
y1_test = y1_test[y1_val_size:]

y2_val_size = len(y2_test) // 2
y2_val = y2_test[:y2_val_size]
y2_test = y2_test[y2_val_size:]

delta_t_val_size = len(delta_t_test) // 2
delta_t_val = delta_t_test[:delta_t_val_size]
delta_t_test = delta_t_test[delta_t_val_size:]

# Constants for the physics equations
E = 6.9e10
miu= 0.35
G = E/2/(1+miu)
rho = 2700.0

# Speed calculation
v_longitudinal = np.sqrt(E / rho)
v_shear = np.sqrt(G / rho)

pos1 = tf.constant([-2, -2], dtype=tf.float32)
pos2 = tf.constant([22, 22], dtype=tf.float32)
pos3 = tf.constant([22, -2], dtype=tf.float32)


def custom_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])+tf.square(y_true[:, 1] - y_pred[:, 1]))

    x_pred, y_pred = y_pred[:, 0], y_pred[:, 1]
    # Calculate the distances
    t1 = y_true[:, 2]
    t2 = y_true[:, 3]
    t3 = y_true[:, 4]
    v = v_shear * 100
    d1 = v * t1 / 3 / 1e6
    d2 = v * t2 / 3 / 1e6
    d3 = v * t3 / 3 / 1e6

    # Calculate the equations
    eq1 = tf.sqrt(tf.square(x_pred - pos1[0]) + tf.square(y_pred - pos1[1])) - tf.sqrt(
        tf.square(x_pred - pos2[0]) + tf.square(y_pred - pos2[1])) - d1
    eq2 = tf.sqrt(tf.square(x_pred - pos2[0]) + tf.square(y_pred - pos2[1])) - tf.sqrt(
        tf.square(x_pred - pos3[0]) + tf.square(y_pred - pos3[1])) - d2
    eq3 = tf.sqrt(tf.square(x_pred - pos3[0]) + tf.square(y_pred - pos3[1])) - tf.sqrt(
        tf.square(x_pred - pos1[0]) + tf.square(y_pred - pos1[1])) - d3

    loss += tf.reduce_mean(tf.square(eq1) + tf.square(eq2) + tf.square(eq3))
    return loss

def mse(y_true, y_pred):

    loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])+tf.square(y_true[:, 1] - y_pred[:, 1]))

    return loss


# train data
train_t1 = delta_t_train[:, 0]
train_t2 = delta_t_train[:, 1]
train_t3 = delta_t_train[:, 2]

val_t1 = delta_t_val[:, 0]
val_t2 = delta_t_val[:, 1]
val_t3 = delta_t_val[:, 2]

# Define the model
def build_trilateration_model():
    model_input = Input(shape=(3,))
    x = Dense(4096, activation='relu')(model_input)
    x = Dense(4096, activation='relu')(x)
    output = Dense(2, activation='linear')(x)  # Output both x and y coordinates
    model = Model(inputs=model_input, outputs=output, name='pinn_model')
    return model

y_true_train = np.concatenate((y1_train[:, np.newaxis], y2_train[:, np.newaxis], train_t1[:, np.newaxis], train_t2[:, np.newaxis], train_t3[:, np.newaxis]), axis=1)
y_true_val = np.concatenate((y1_val[:, np.newaxis], y2_val[:, np.newaxis], val_t1[:, np.newaxis], val_t2[:, np.newaxis], val_t3[:, np.newaxis]), axis=1)

pinn_model = build_trilateration_model()
pinn_model.compile(optimizer=Adam(lr=0.001), loss=custom_loss, metrics=[mse])
history = pinn_model.fit(delta_t_train, y_true_train, epochs=300, batch_size=32, validation_data=(delta_t_val, y_true_val), validation_freq=1)

model_json = pinn_model.to_json()
with open("./result/15-2/pinn/t_model.json", 'w') as json_file:
    json_file.write(model_json)
pinn_model.save_weights('./result/15-2/pinn/t_model.h')
pinn_model.summary()
with open("./result/15-2/pinn/t.txt", 'w') as f1:
    for i in range(0, len(history.history['loss'])):
        a1 = history.history['loss'][i]
        b1 = history.history['val_loss'][i]
        a2 = history.history['mse'][i]
        b2 = history.history['val_mse'][i]
        f1.write(str(a1) + ' ' + str(a2) + ' ' + str(b1) + ' ' + str(b2) + "\n")



