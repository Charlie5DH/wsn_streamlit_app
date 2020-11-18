import os
import datetime

import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# resetting backedn to matplotlib in case was changed to plotly
pd.options.plotting.backend = "matplotlib"

sns.set(style='whitegrid', palette='deep', font_scale=1.2)

N_TIMESTEPS_IN = 256
N_TIMESTEPS_OUT = 16
N_FEATURES = 3
N_HIDDEN = 128
EPOCHS = 100
BATCH_SIZE = 256

@st.cache
def get_data(path_of_file = 'Data/single_feature.csv'):
    '''
    Reading data from CSV file
    '''
    data = pd.read_csv(path_of_file, parse_dates=['Timestamp'], index_col='Timestamp')
    return data

def load_pretrained_model(name='model'):
  json_file = open(name+'.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = tf.keras.models.model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(name+'.h5')
  print("Loaded model from disk")
  return loaded_model

def get_errors(model, train_X, test_X, train_y, test_y):
    
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    MAE_train = abs(train_predictions - train_y).mean()
    MAE_test = abs(test_predictions - test_y).mean()
    if train_y.shape[1] == 3:
        mae_overall_train = abs(train_predictions-train_y).mean(axis=(1))
        mae_overall_test = abs(test_predictions-test_y).mean(axis=(1))
        mse_test = mean_squared_error(test_y, test_predictions)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(test_y, test_predictions)
    else:
        mae_overall_train = abs(train_predictions-train_y).mean(axis=(1,2))
        mae_overall_test = abs(test_predictions-test_y).mean(axis=(1,2))
        mse_test = mean_squared_error(test_y.reshape(-1,3), test_predictions.reshape(-1,3))
        r2_test = r2_score(test_y.reshape(-1,3), test_predictions.reshape(-1,3)) 

    rmse_test = np.sqrt(mse_test)
    return MAE_train, MAE_test, mae_overall_train, mae_overall_test, mse_test, r2_test, rmse_test

# split a multivariate sequence into samples
def multi_step_output(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def forecast_for_tx(model, train_X, normalized_train, n_timesteps_out, pred_range=100, threshold=0.08):
  '''
  Function to make a forecast checking the error in prediction.
  If the error is too big, then the real values are copied t the prediction
  so next prediction will consider real points. This will diminish the error in
  forecast. Considering only error in temperature to make it simple.
  '''
  n_timesteps_in = train_X.shape[1]
  n_features = train_X.shape[-1]
  
  # next_bath starts with the last batch of the training set
  next_batch = train_X[-1, :, :]
  predictions_forecast = []
  next_step_truth_list = []
  counter = 0
  error_stack = []

  # Repeat a number of future timesteps
  for ii in range(pred_range):
    # Reshape to 3D for prediction
    next_step_pred = model.predict(next_batch.reshape(1, n_timesteps_in, n_features))
    # Reshape to 2D
    next_step_pred = next_step_pred.reshape(n_timesteps_out, n_features)

    # the real values are in normalized_train. Starting from train_len is the testing data set
    # which is the continuation (or truth) of the values predicted in from the last training batch
    # but only n_timesteps_out amounth. We have to shift the timesteps in validation by n_timesteps_out times 
    next_step_truth = normalized_train[train_len:][ii * n_timesteps_out : n_timesteps_out*(ii+1)]
    next_step_truth_list.append(next_step_truth)
    # calculate error
    error_temp = abs(next_step_truth[:,0] - next_step_pred[:,0])
    #error_vmppt = abs(next_step_truth[:,1] - next_step_pred[:,1])
    #error_vpanel = abs(next_step_truth[:,2] - next_step_pred[:,2])
    #error = abs(next_step_truth - next_step_pred).mean()

    # Stack the rolling MAE for counting the points
    # Next we have to ensamble the data with the points and 
    # not with the sequences but for now let's just count
    error_temp = error_temp.reshape(-1,1).mean(axis=1)
    error_stack.append(error_temp)

    #transmitting points with higher error
    prep_batch = next_step_pred
    for elem in range(len(error_temp)):
      if error_temp[elem] > threshold:
        prep_batch[elem] = next_step_truth[elem]
        counter+=1

    predictions_forecast.append(prep_batch)
    # Take the next batch for predcition
    next_batch = np.row_stack([next_batch[n_timesteps_out:], prep_batch])

  # convert to numpy & reshape predictions
  predictions_forecast = np.array(predictions_forecast).reshape(-1,3)
  # convert to numpy & reshape real values
  forecast_truth = np.array(next_step_truth_list).reshape(-1,3)
  # Stack error and real values
  forecast_result = np.column_stack([predictions_forecast,forecast_truth])

  return error_stack, forecast_result, counter

def normalize_and_split(data, init_date='2019-03', end_date='2019-04-20', freq='5Min', train_ratio=0.8):
    '''
    Resample and split the dataset by date and ratio.
    Select a date range, a frequency to resample and the ratio for training
    data: Dataframe to resample and split
    init_date: Initial Date
    end_date: Final Date of split
    freq: frequency to resample
    '''
    # The data is higly irregular so let's resample it to 10 min and take the mean
    resampled = data.resample(freq).mean()
    resampled = resampled.fillna(resampled.bfill())
    # Split by date
    train_data = resampled[init_date:end_date]

    train_len = int(train_ratio * len(train_data))
    # Normalize
    scaler = MinMaxScaler()
    normalized_train = scaler.fit_transform(train_data)

    return normalized_train, train_len, train_data

def split_sequences_multivariate(sequences, n_steps=32):
    
    '''
    Split a multivariate sequence into samples for single feature prediction
    Taken and adapted from Machinelearningmastery.
    Split the training set into segments of a specified timestep
    and creates the labels.
    '''
    #n_steps = n_steps+1
    # Place the column of the feature to predict at the end of the dataset
    #sequences = np.concatenate([X_train, X_train[:,0].reshape(-1,1)],axis=1)
    
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    
    #print(np.shape(X),np.shape(y))
    return np.array(X), np.array(y)

def plot_mae_overall(MAE_train, MAE_test, mae_overall_train, mae_overall_test, suptitle):

    fig = px.line(mae_overall_test, title="MAE Test", template="plotly_white",
              labels=dict(index="Timesteps", value="MAE", variable="MAE"))
    # Update layout properties, Add figure title
    fig.update_layout(showlegend=True, autosize=True,
                    title_text="MAE Test {}".format(MAE_test),
                    legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                    title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    template='plotly_white')
    st.plotly_chart(fig)

def plot_lstm(model, X, y, feature_index=0):

    prediction = model.predict(X)
    if prediction.shape[-1] == 1:
        pred_df = pd.DataFrame(data= np.concatenate((model.predict(X), y.reshape(-1,1)), axis=1), columns=['Prediction', 'True'])
    else:
        pred_df = pd.DataFrame(data= np.concatenate((model.predict(X)[:,feature_index].reshape(-1,1),
                                                     y[:,feature_index].reshape(-1,1)), axis=1), columns=['Prediction', 'True'])
    rmse = np.sqrt(mse_test)
    r2 = r2_score(y, prediction)

    fig = px.line(pred_df, template="plotly_white", labels=dict(index="Timesteps", value="Temperature", variable="Truth/Pred"))
    fig.update_layout(showlegend=True, autosize=True,
                    title_text='Prediction and Truth rmse = {}, r2 score = {}'.format(np.round(rmse,4), np.round(r2,4)),
                    legend={'orientation':"h", 'yanchor':"bottom", 'y':1.04, 'xanchor':"right", 'x':0.95},
                    title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    template='plotly_white')
    st.plotly_chart(fig)

def plot_single_seq(model, X, y, feature_index=0):

    prediction = model.predict(X)[:,7,:]
    y = y[:,7,:]
    pred_df = pd.DataFrame(data= np.column_stack((prediction.reshape(-1,3)[:,feature_index],
                                                y.reshape(-1,3)[:,feature_index])),
                            columns=['Prediction', 'True'])

    mse = mean_squared_error(y.reshape(-1,3), prediction.reshape(-1,3))
    rmse = np.sqrt(mse)
    r2 = r2_score(y.reshape(-1,3), prediction.reshape(-1,3)) 

    fig = px.line(pred_df, template="plotly_white", labels=dict(index="Timesteps", value="Data", variable="True/Pred"))
    fig.update_layout(showlegend=True, autosize=True,
                    title_text='Prediction and Truth rmse = {}, r2 score = {}'.format(np.round(rmse,4), np.round(r2,4)),
                    legend={'orientation':"h", 'yanchor':"bottom", 'y':1.04, 'xanchor':"right", 'x':0.95},
                    title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    template='plotly_white')
    st.plotly_chart(fig)

st.title("Forecasting with attention in WSN")
st.markdown('This App uses a Seq2Seq model with an attention mechanism '
'for prediction in a WSN using the raw data from the sensor. '
'A percent of the transmitted data over a period of three days is given '
'considering different error thresholds. Results show that the '
'model can save a considerable amount of data in transmission '
'and still maintain a good performance in prediction.')

# Get dataframe 
data = get_data(path_of_file = 'Data/single_feature.csv')

#-------------------------------Date selection----------------------------------#
st.sidebar.header('User Inputs')
st.sidebar.markdown('Please select a date to split the data. Use only dates from 2017-02-10 to 2020-01-01')

start_date = st.sidebar.date_input('Start date', datetime.datetime(2019, 3, 1))  # DATE SELECTIONS
end_date = st.sidebar.date_input('End date', datetime.datetime(2019, 4, 20))  # DATE SELECTIONS

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

train_ratio = st.sidebar.slider('Select a train ratio', 0.0, 0.95, 0.8, 0.05)  
freq = st.sidebar.selectbox('Select a resample frequency', ('5Min', '10Min', '30Min', '60Min','12H','24H'))

try:
    normalized_train, train_len, train_data = normalize_and_split(data, init_date=start_date, end_date=end_date,
     train_ratio=train_ratio, freq=freq)
except:
    st.sidebar.markdown('You chose an incorrect date. Default vales will be Selected')
    normalized_train, train_len, train_data = normalize_and_split(data, train_ratio=train_ratio, freq=freq)

#---------------------------'Splitting and Normalizing'--------------------------------#
st.subheader('Splitting and Normalizing')
st.text('Shapes after Splitting \nLenght of Data {}\nLenght of Train {}'.format(len(normalized_train),train_len))
#st.dataframe(pd.DataFrame(data=normalized_train, columns=['Temperature', 'VPanel','VMPPT']), 800, 200)
st.markdown('Below are the time series after splitting and resampling. ' 
'You can tap the legend to hide a feature')

#-----------------------------------------Plotly Figure---------------------------------------------#
fig = px.line(train_data, title="Sensory Data", template="plotly_white",
              labels=dict(index="Time", value="Data", variable="Sensors"))
# Update layout properties, Add figure title
fig.update_layout(showlegend=True, autosize=True,
                 title_text="Sensory Data",
                 legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                 title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                 template='plotly_white')
st.plotly_chart(fig)

#-----------------------------------------Load Model---------------------------------------------#

model_sel = st.sidebar.radio('Select the pretrained model. Select Attention to see Forecast', ('LSTM', 'GRU','CNN', 'DNN', 'Attention'))

train_X, train_y = split_sequences_multivariate(normalized_train[:train_len], n_steps=N_TIMESTEPS_IN)
test_X, test_y = split_sequences_multivariate(normalized_train[train_len:], n_steps=N_TIMESTEPS_IN)

if model_sel == 'LSTM':
    model = load_pretrained_model(name='Model/lstm_model')   # LOAD THE MODEL
if model_sel == 'GRU':
    model = load_pretrained_model(name='Model/gru_model')   # LOAD THE MODEL
if model_sel == 'CNN':
    model = load_pretrained_model(name='Model/cnn_model')   # LOAD THE MODEL
if model_sel == 'DNN':
    model = load_pretrained_model(name='Model/dnn_model')   # LOAD THE MODEL
if model_sel == 'Attention':
    model = load_pretrained_model(name='Model/overfitted_att')   # LOAD THE MODEL
    # IF ATTENTION THE TIMESTEPS ARE DIFFERENT
    N_TIMESTEPS_IN = 256
    N_TIMESTEPS_OUT = 8

    train_X, train_y = multi_step_output(normalized_train[:train_len], n_steps_in=N_TIMESTEPS_IN, n_steps_out=N_TIMESTEPS_OUT)
    test_X, test_y = multi_step_output(normalized_train[train_len:], n_steps_in=N_TIMESTEPS_IN, n_steps_out=N_TIMESTEPS_OUT)

mae_train, mae_test, mae_overall_train, mae_overall_test, mse_test, r2_test, rmse_test = get_errors(model, train_X, test_X, train_y, test_y)
plot_mae_overall(mae_train, mae_test, mae_overall_train, mae_overall_test, suptitle=model_sel)

if model_sel != 'Attention':
    plot_lstm(model, test_X, test_y, feature_index=0)
else:
    plot_single_seq(model, test_X, test_y, feature_index=0)
    # get the errors
    threshold = st.sidebar.slider('Select the threshold', 0.01, 0.2, 0.08, 0.01)  # SLIDER TO SELECT K

    error_stack, forecast_result, counter = forecast_for_tx(model, train_X, normalized_train, N_TIMESTEPS_OUT, threshold=threshold)

    forecast_result_df = pd.DataFrame(data=forecast_result, columns=['temp_pred','vmppt_pred','vpanel_pred','temp','vmppt','vpanel'])
    forecast_result_df = forecast_result_df.set_index(train_data[train_len:][:forecast_result_df.shape[0]].index)
    err = np.array(error_stack).reshape(-1,1)

    #---------------------------PLOT Forecast-------------------------------------------#

    forecast_result_df['error'] = err
    forecast_result_df['Transmitted'] = (forecast_result_df['error'] > threshold)
    points_tx = pd.DataFrame(data = forecast_result_df[forecast_result_df['Transmitted']==True]['temp'], 
    index=forecast_result_df[forecast_result_df['Transmitted']==True].index)

    st.markdown('Next figure shows a comparison between the real data and a prediction '
    'with a threshold of 0.08 and the transmitted points marked')

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=forecast_result_df.index, y=forecast_result_df.temp_pred,
                mode='lines', name='Predicted Temperature'))
    fig.add_trace(               
        go.Scatter(x=forecast_result_df.index, y=forecast_result_df.temp,
                mode='lines', name='Temperature'))
    fig.add_trace(
        go.Scatter(x=points_tx.index, y=points_tx.temp, name='Points Transmitted',mode='markers'))    
    fig.update_layout(showlegend=True, autosize=True,
                    title_text="Temperature Predictions",
                    legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                    title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    template='plotly_white')
    st.plotly_chart(fig)

    st.success('Number of Points to Transmit: {}'.format(counter))

    #--------------------------------------CHECKING MULTIPLE THRESHOLDS-------------------------#

    count_list = []
    percent_list=[]
    error_list = []
    std_list = []
    rolling_mae = []

    sel_range = st.sidebar.slider('Select range', 0.01, 0.2, (0.02, 0.1), 0.005)  # SLIDER TO SELECT K
    #---------------------------SELECT THE NUMBER OF POINTS----------------------------------#
    n_points = st.sidebar.slider('Select number of points', 5, 50, 10, 1)  # SLIDER TO SELECT K

    for ii in np.linspace(sel_range[0], sel_range[1], n_points):
        error_stack, forecast_result, counter = forecast_for_tx(model, train_X, normalized_train, N_TIMESTEPS_OUT, threshold=ii)
        count_list.append(counter)
        percent_list.append((counter*100)/forecast_result.shape[0])
        error_list.append(abs(forecast_result[:,0] - forecast_result[:,3]).mean() * 43.5)
        std_list.append(abs(forecast_result[:,0] - forecast_result[:,3]).std())
        rolling_mae.append(abs(forecast_result[:,0] - forecast_result[:,3]).reshape(-1,1).mean(axis=1))

    st.markdown('Next figure shows the MAE between the real values '
    'read by the sensor and the entire forecast period when using '
    'different thresholds and in a secondary axis the percent of '
    'points transmitted from the sensor to the BS.')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=np.linspace(0.020, 0.15, 50)*43.5, y=np.array(error_list),
                mode='lines+markers', name='MAE (C\N{DEGREE SIGN})'), secondary_y=False)
    fig.add_trace(               
        go.Scatter(x=np.linspace(0.020, 0.15, 50)*43.5, y=percent_list,
                mode='lines+markers', name='Transmitted (%)'), secondary_y=True)   
    # Set x-axis title
    fig.update_xaxes(title_text="Thresholds")
    fig.update_yaxes(title_text="<b>MAE (C\N{DEGREE SIGN})</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Transmitted (%)</b>", secondary_y=True)

    fig.update_layout(showlegend=True, autosize=True,
                    title_text="Prediction performance by thresholds",
                    legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                    title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    template='plotly_white')
    st.plotly_chart(fig)


    #--------------------------------Comparing two forecast------------------------------------------#
    # Threshols = 0.03, 0.07
    error_stack, forecast_result, counter = forecast_for_tx(model, train_X, normalized_train, N_TIMESTEPS_OUT, threshold=0.02)
    forecast_result_df = pd.DataFrame(data=forecast_result, columns=['temp_pred','vmppt_pred','vpanel_pred','temp','vmppt','vpanel'])
    forecast_result_df = forecast_result_df.set_index(train_data[train_len:][:forecast_result_df.shape[0]].index)
    # Threshols = 0.08
    error_stack_2, forecast_result_2, counter_2 = forecast_for_tx(model, train_X, normalized_train, N_TIMESTEPS_OUT, threshold=0.057)
    forecast_result_df_2 = pd.DataFrame(data=forecast_result_2, columns=['temp_pred','vmppt_pred','vpanel_pred','temp','vmppt','vpanel'])
    forecast_result_df_2 = forecast_result_df_2.set_index(train_data[train_len:][:forecast_result_df_2.shape[0]].index)

    st.markdown('Next Figure shows a comparison with Model 1'
    'between the real data and two final predictions, one with a '
    'small threshold of 1.3 degrees and 30.62 percent of transmitted points '
    'and another with a threshold of 3 degrees and 10 percent of transmitted '
    'points. We can see that, with a maximum error of 3 degrees the '
    'predicted data still captures the patterns and the abrupt changes '
    'of the real data, saving up to 90 percent of transmission.')

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=forecast_result_df['temp'].index, y=forecast_result_df.temp*43.5,
                mode='lines', name='Real Values'))
    fig.add_trace(               
        go.Scatter(x=forecast_result_df['temp_pred'].index, y=forecast_result_df.temp_pred*43.5,
                mode='lines', name='Threshold of 1 C\N{DEGREE SIGN}'))
    fig.add_trace(
        go.Scatter(x=forecast_result_df_2['temp_pred'].index, y=forecast_result_df_2.temp_pred*43.5,
                name='Threshold of 3 C\N{DEGREE SIGN}', mode='lines'))  

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="<b>Temperature</b>")

    fig.update_layout(showlegend=True, autosize=True,
                    title_text="Forecast with different thresholds",
                    legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                    title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    template='plotly_white')
    st.plotly_chart(fig)









