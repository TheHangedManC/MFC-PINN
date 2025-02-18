import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input, Conv2D, MaxPooling2D, Permute, Reshape, add, Dense, Dropout, Flatten, RepeatVector, Permute, multiply, Lambda, concatenate, Activation
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import cv2

IMAGE_DIMS = (224, 224, 3)

cwt_path = './dataset/cwt/'
cwt_txt = './dataset/10-3.txt'

x_train_savepath = './dataset/10-3/x_train.npy'
y1_train_savepath = './dataset/10-3/y1_train.npy'
y2_train_savepath = './dataset/10-3/y2_train.npy'

x_test_savepath = './dataset/10-3/x_test.npy'
y1_test_savepath = './dataset/10-3/y1_test.npy'
y2_test_savepath = './dataset/10-3/y2_test.npy'

delta_t_train_savepath = './dataset/10-3/delta_t_train.npy'
delta_t_test_savepath = './dataset/10-3/delta_t_test.npy'

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
    x1, x2, y1_1, y1_2 = train_test_split(x, y1_, test_size=0.2, random_state=123)
    x1, x2, y2_1, y2_2 = train_test_split(x, y2_, test_size=0.2, random_state=123)
    delta_t1, delta_t2, y1_1, y1_2 = train_test_split(delta_t, y1_, test_size=0.2, random_state=123)

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

json_file = open("./result/10-3/pinn/t_model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./result/10-3/pinn/t_model.h")
pinn_predicted_y = loaded_model.predict(delta_t_test)
pinn_predicted_y1 = np.array(pinn_predicted_y[:, 0:1])*10
pinn_predicted_y2 = np.array(pinn_predicted_y[:, 1:2])*10

json_file = open("./result/10-3/cnn/cwt_model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./result/10-3/cnn/cwt_model.h")
cnn_predicted_y = loaded_model.predict(x_test)
cnn_predicted_y1 = np.array(cnn_predicted_y[:, 0:1])*10
cnn_predicted_y2= np.array(cnn_predicted_y[:, 1:2])*10

json_file = open("./result/10-3/cnn-pinn/model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./result/10-3/cnn-pinn/model.h")
cnnpinn_predicted_y = loaded_model.predict([delta_t_test, x_test])
cnnpinn_predicted_y1 = np.array(cnnpinn_predicted_y[:, 0:1])*10
cnnpinn_predicted_y2= np.array(cnnpinn_predicted_y[:, 1:2])*10

y1_test=y1_test*10
y2_test=y2_test*10

size = len(y1_test)
p1 = [1] * size
p2 = [2] * size
p3 = [3] * size
combined_array = np.column_stack((p1, pinn_predicted_y1, y1_test, p2, cnn_predicted_y1, y1_test, p3, cnnpinn_predicted_y1, y1_test))
np.savetxt('10-3x.txt', combined_array, delimiter='\t')

combined_array = np.column_stack((p1, pinn_predicted_y2, y2_test, p2, cnn_predicted_y2, y2_test, p3, cnnpinn_predicted_y2, y2_test))
np.savetxt('10-3y.txt', combined_array, delimiter='\t')

combined_array = np.column_stack((pinn_predicted_y1, pinn_predicted_y2, cnn_predicted_y1, cnn_predicted_y2, cnnpinn_predicted_y1, cnnpinn_predicted_y2, y1_test, y2_test))
np.savetxt('10-3-line.txt', combined_array, delimiter='\t')

combined_array = np.column_stack((cnnpinn_predicted_y1, cnnpinn_predicted_y2, y1_test, y2_test))
np.savetxt('10-3.txt', combined_array, delimiter='\t')

mse_y1 = mean_squared_error(y1_test, pinn_predicted_y1)
print(mse_y1)
mse_y2 = mean_squared_error(y2_test, pinn_predicted_y2)
print(mse_y2)

mae_y1 = mean_absolute_error(y1_test, pinn_predicted_y1)
print(mae_y1)
mae_y2 = mean_absolute_error(y2_test, pinn_predicted_y2)
print(mae_y2)

r2_y1 = r2_score(y1_test, pinn_predicted_y1)
print(r2_y1)
r2_y2 = r2_score(y2_test, pinn_predicted_y2)
print(r2_y2)




mse_y1 = mean_squared_error(y1_test, cnn_predicted_y1)
print(mse_y1)
mse_y2 = mean_squared_error(y2_test, cnn_predicted_y2)
print(mse_y2)

mae_y1 = mean_absolute_error(y1_test, cnn_predicted_y1)
print(mae_y1)
mae_y2 = mean_absolute_error(y2_test, cnn_predicted_y2)
print(mae_y2)

r2_y1 = r2_score(y1_test, cnn_predicted_y1)
print(r2_y1)
r2_y2 = r2_score(y2_test, cnn_predicted_y2)
print(r2_y2)



mse_y1 = mean_squared_error(y1_test, cnnpinn_predicted_y1)
print(mse_y1)
mse_y2 = mean_squared_error(y2_test, cnnpinn_predicted_y2)
print(mse_y2)

mae_y1 = mean_absolute_error(y1_test, cnnpinn_predicted_y1)
print(mae_y1)
mae_y2 = mean_absolute_error(y2_test, cnnpinn_predicted_y2)
print(mae_y2)

r2_y1 = r2_score(y1_test, cnnpinn_predicted_y1)
print(r2_y1)
r2_y2 = r2_score(y2_test, cnnpinn_predicted_y2)
print(r2_y2)