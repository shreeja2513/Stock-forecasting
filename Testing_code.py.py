import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import random

def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')

def predict_func(data):
    random.seed(10)
    dat=data
    dat2=dat.interpolate(method='cubic', order=3)
    dat2=dat2.dropna()
    dat3=dat2.reset_index()['Close']
    scale=MinMaxScaler(feature_range=(0,1))
    dat4=scale.fit_transform(np.array(dat3).reshape(-1,1))
    dat4.shape
    lstmmodel=load_model("model_1.h5")
    time_step=3
    prevdata=dat4[-time_step:]
    next_list=[]
    counter = 0

    while counter < 2:
       input_data = prevdata[-time_step:].reshape(1, time_step, 1)
       next_price = lstmmodel.predict(input_data)
       next_list.append(next_price[0, 0])
       prevdata = np.append(prevdata, next_price, axis=0)
       counter += 1

    next_prices = scale.inverse_transform(np.array(next_list).reshape(-1, 1))
    return np.array(next_prices).flatten().tolist()
     
if __name__== "__main__":
    evaluate()