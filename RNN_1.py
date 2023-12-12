import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

from sklearn import preprocessing

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os


def Normalize_DF(df_1):
    pd_header = df_1.columns.tolist()

    for each in pd_header:
        df_1[each] = preprocessing.MaxAbsScaler().fit_transform(df_1[each].values.reshape(-1, 1))

    return df_1






def Model_RNN_1(X_train,Y_train,X_val,Y_val):

    regressor = Sequential()
    # adding first RNN layer and dropout regulatization
    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True, 
                input_shape = (X_train.shape[1],X_train.shape[2]))
                )

    regressor.add(
        Dropout(0.2)
                )


    # adding second RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
                )

    regressor.add(
        Dropout(0.2)
                )

    # adding third RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
                )

    regressor.add(
        Dropout(0.2)
                )


    # adding Fourth RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
                )

    regressor.add(
        Dropout(0.2)
                )


    # adding fourth RNN layer and dropout regulatization

    regressor.add(
        SimpleRNN(units = 50)
                )

    regressor.add(
        Dropout(0.2)
                )

    # adding the output layer
    regressor.add(Dense(units = 1,activation="sigmoid"))

    # compiling RNN
    regressor.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics = ["accuracy"])
    
    history = regressor.fit(X_train, Y_train, epochs = 500, batch_size = 128)
    
    # Plotting Loss vs Epochs
    plt.figure(figsize =(10,7))
    plt.plot(history.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Simple RNN model, Loss vs Epoch")
    plt.show()
    
    Y_pred = regressor.predict(X_train)
    Y_pred = Y_pred.flatten()
    Y_pred[Y_pred >= 0.8] = 1
    Y_pred[Y_pred < 0.8] = 0

    plt.figure(figsize = (30,10))
    plt.plot(Y_pred, color = "b", label = "y_pred" )
    plt.plot(Y_train, color = "g", label = "y_train")
    plt.xlabel("Days")
    plt.ylabel("Open price")
    plt.title("Simple RNN model, Predictions with input X_train vs y_train")
    plt.legend()
    plt.savefig('TrainGTvsTrainPred_v2.png')
    plt.show()


    Y_val_pred = regressor.predict(X_val)
    Y_val_pred = Y_val_pred.flatten()
    Y_val_pred[Y_val_pred >= 0.8] = 1
    Y_val_pred[Y_val_pred < 0.8] = 0

    plt.figure(figsize = (30,10))
    plt.plot(Y_val_pred, color = "b", label = "y_pred" )
    plt.plot(Y_val, color = "g", label = "y_train")
    plt.xlabel("Days")
    plt.ylabel("Open price")
    plt.title("Simple RNN model, Predictions with input X_train vs y_train")
    plt.legend()
    plt.savefig('ValGTvsValPred_v2.png')
    plt.show()





if __name__=="__main__":
    print("main")
    symbol = "ITC.NS.csv"
    data_1 = pd.read_csv(symbol)

    data_1 = data_1[250:]    # Filter out first 250 data, to avoid outliers / remove improper data

    testdatano = 50
    length_data = len(data_1)     # rows that data has

    data = data_1.iloc[:length_data-testdatano]         # Remove last 50 row for testing

    length_data = len(data)     # rows that data has
    split_ratio = 0.7           # %70 train + %30 validation
    length_train = round(length_data * split_ratio)  
    length_validation = length_data - length_train
    print("Data length :", length_data)
    print("Train data length :", length_train)
    print("Validation data lenth :", length_validation)


    #x_train_data = data[:length_train].iloc[:,1:-1] 
    x_train_data = data[:length_train].iloc[:,5:-1] 
    y_train_data = data[:length_train].iloc[:,-1] 

    x_train_data = Normalize_DF(x_train_data)

    x_train_data = np.array(x_train_data)
    y_train_data = np.array(y_train_data)
    #train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object
    print(x_train_data)
    print(y_train_data)

    print(x_train_data.shape)
    print(y_train_data.shape)

    time_step = 5
    x_tr = []
    y_tr = []
    for i in range(time_step, length_train):
        x_tr.append(x_train_data[i-time_step:i])
        y_tr.append(y_train_data[i-1])
        #print(x_train_data[i-time_step:i])
        #print(y_train_data[i])
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr)

    print("X Train shape:",x_tr.shape)
    print("Y Train shape:",y_tr.shape)




    #x_validation_data = data[length_train:].iloc[:,1:-1]
    x_validation_data = data[length_train:].iloc[:,5:-1]
    y_validation_data = data[length_train:].iloc[:,-1]

    x_validation_data = Normalize_DF(x_validation_data)

    #print(validation_data)
    x_validation_data = np.array(x_validation_data)
    y_validation_data = np.array(y_validation_data)


    print(x_validation_data)
    print(y_validation_data)

    print(x_validation_data.shape)
    print(y_validation_data.shape)

    print(len(x_validation_data))
    length_validation = len(x_validation_data)

    x_va = []
    y_va = []
    for i in range(time_step, length_validation):
        x_va.append(x_validation_data[i-time_step:i])
        y_va.append(y_validation_data[i])

    x_va = np.array(x_va)
    y_va = np.array(y_va)

    print("X Validation shape:",x_va.shape)
    print("Y Validation shape:",y_va.shape)




    #################### NN Training ############

    print(x_tr.shape, x_tr.shape[1],x_tr.shape[2])
    Model_RNN_1(x_tr,y_tr,x_va,y_va)
