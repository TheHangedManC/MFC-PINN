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
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
IMAGE_DIMS = (224, 224, 3)
with open('shuchu.txt', 'w') as file:
    for i in range(5,16,5):
        for j in range(1, 4, 1):
            name=str(i)+'-'+str(j)
            content = f"{name}\n"
            file.write(content)
            x_train_savepath = './dataset/'+name+'/x_train.npy'
            y1_train_savepath = './dataset/'+name+'/y1_train.npy'
            y2_train_savepath = './dataset/'+name+'/y2_train.npy'
            x_test_savepath = './dataset/'+name+'/x_test.npy'
            y1_test_savepath = './dataset/'+name+'/y1_test.npy'
            y2_test_savepath = './dataset/'+name+'/y2_test.npy'
            delta_t_train_savepath = './dataset/'+name+'/delta_t_train.npy'
            delta_t_test_savepath = './dataset/'+name+'/delta_t_test.npy'
            x_test_savepathc = './dataset/'+name+'/x_testc.npy'
            y1_test_savepathc = './dataset/'+name+'/y1_testc.npy'
            y2_test_savepathc = './dataset/'+name+'/y2_testc.npy'
            delta_t_test_savepathc = './dataset/'+name+'/delta_t_testc.npy'

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

            x_val = x_testc
            y1_val = y1_testc
            y2_val = y2_testc
            delta_t_val = delta_t_testc
            y1_test=y1_test*10
            y2_test=y2_test*10

            json_file = open("./result/"+name+"/cnn/cwt_model.json")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("./result/"+name+"/cnn/cwt_model.h")
            cnn_predicted_y = loaded_model.predict(x_test)
            cnn_predicted_y1 = np.array(cnn_predicted_y[:, 0:1])*10
            cnn_predicted_y2 = np.array(cnn_predicted_y[:, 1:2])*10


            json_file = open("./result/"+name+"/pinn/model.json")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("./result/"+name+"/pinn/model.h")
            cnn_predicted_y = loaded_model.predict(delta_t_test)
            pinn_predicted_y1 = np.array(cnn_predicted_y[:, 0:1])*10
            pinn_predicted_y2 = np.array(cnn_predicted_y[:, 1:2])*10


            json_file = open("./result/"+name+"/cnn-pinn/model.json")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("./result/"+name+"/cnn-pinn/model.h")
            cnn_predicted_y = loaded_model.predict([delta_t_test, x_test])
            cnnpinn_predicted_y1 = np.array(cnn_predicted_y[:, 0:1])*10
            cnnpinn_predicted_y2 = np.array(cnn_predicted_y[:, 1:2])*10

            size = len(y1_test)
            p1 = [1] * size
            p2 = [2] * size
            p3 = [3] * size
            combined_array = np.column_stack(
                (p1, pinn_predicted_y1, y1_test, p2, cnn_predicted_y1, y1_test, p3, cnnpinn_predicted_y1, y1_test))
            np.savetxt(name+'x.txt', combined_array, delimiter='\t')

            combined_array = np.column_stack(
                (p1, pinn_predicted_y2, y2_test, p2, cnn_predicted_y2, y2_test, p3, cnnpinn_predicted_y2, y2_test))
            np.savetxt(name+'y.txt', combined_array, delimiter='\t')

            combined_array = np.column_stack((pinn_predicted_y1, pinn_predicted_y2, cnn_predicted_y1, cnn_predicted_y2,
                                              cnnpinn_predicted_y1, cnnpinn_predicted_y2, y1_test, y2_test))
            np.savetxt(name+'line.txt', combined_array, delimiter='\t')

            combined_array = np.column_stack((cnnpinn_predicted_y1, cnnpinn_predicted_y2, y1_test, y2_test))
            np.savetxt(name+'.txt', combined_array, delimiter='\t')