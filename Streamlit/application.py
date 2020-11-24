import os
import io
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

from dataset import Dataset
from model_forecast import forecast_model
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# resetting backedn to matplotlib in case was changed to plotly
pd.options.plotting.backend = "matplotlib"

sns.set(style='whitegrid', palette='deep', font_scale=1.2)

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

st.title("Forecasting with attention in WSN")
st.markdown('This App uses a Seq2Seq model with an attention mechanism '
'for prediction in a WSN using the raw data from the sensor. '
'A percent of the transmitted data over a period of three days is given '
'considering different error thresholds. Results show that the '
'model can save a considerable amount of data in transmission '
'and still maintain a good performance in prediction.')

# Get dataframe 
st.sidebar.header('User Inputs')
module = st.sidebar.selectbox('Select one of the following modules',
 ("34.B2.9F.A9","00.57.FE.0E","00.57.FE.0F", "00.57.FE.06","00.57.FE.09",
 "00.57.FE.05", "00.57.FE.03", "29.E5.5A.24", "A7.CB.0A.C0","00.57.FE.04",
 "01.E9.39.32", "A4.0D.82.38", "9F.8D.AC.91",  "50.39.E2.80"))

data = Dataset(module=module) 

st.sidebar.markdown('Please select a date to split the data. Use only dates from 2017-02-10 to 2020-01-01')

start_date = st.sidebar.date_input('Start date', datetime.datetime(2019, 3, 1))  # DATE SELECTIONS
end_date = st.sidebar.date_input('End date', datetime.datetime(2019, 4, 20))  # DATE SELECTIONS

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

train_ratio = st.sidebar.slider('Select a train ratio', 0.0, 0.95, 0.8, 0.05)
data.split_dataset(init_date=start_date, end_date=end_date, train_ratio=train_ratio)

resample_freq = st.sidebar.selectbox('Select a resample frequency', ('5Min', '10Min', '30Min', '60Min','12H','24H'))
data.resample_dataset(freq=resample_freq)
st.dataframe(data.dataset)

data.plot_data()

feature = st.sidebar.selectbox('Select a feature to plot', (data.features.values))
data.distribution(features=feature)
date_sel = st.sidebar.selectbox('Select the date form', ('day','month', 'week','weekday','hour','daylight'))
data.date_plot(feature=feature, date=date_sel)

data.normalize_dataset()

train = st.sidebar.radio('Select if train or load pretrained',('Load Pretrained', 'Train'))
if train=='Train':
    model_sel = st.sidebar.selectbox('Select a model for training. Select Attention to see Forecast', 
    ('LSTM', 'GRU','CNN', 'DNN', 'Seq2Seq', 'Attention'))
    model = forecast_model()

    n_steps_in = st.sidebar.selectbox('Select number of input timestamps', (32, 64, 128, 256, 324, 512))
    n_steps_out = st.sidebar.selectbox('Select number of output timestamps', (4, 8, 16, 64, 128))
    feature_index = st.sidebar.slider('Select the index of the feature to predict. Check the Dataset to be sure', 0, data.n_features, 0, 1)
    loss = st.sidebar.selectbox('Loss', ('mean_squared_error', 'mean_absolute_error'))
    epochs = st.sidebar.slider('Select Numer of Epochs for training', 5, 100, 10, 5)
    batch_size = st.sidebar.slider('Select Batch Size', 32, 512, 256, 32)
    save = st.sidebar.radio('Save Model',('Save not', 'Save Model'))

    if ((model_sel == 'Seq2Seq') or (model_sel == 'Attention')):
        data.create_supervised_mimo(n_steps_in=n_steps_in, n_steps_out=n_steps_out)                        # create supervise training set
        model.set_model(input_shape=data.input_shape, output_shape=data.output_shape, model_type=model_sel)  # define model
    else:
        data.create_supervised(n_steps_in=n_steps_in)                        # create supervise training set
        model.set_model(input_shape=data.input_shape, model_type=model_sel)  # define model

    st.markdown('{} Model Summary'.format(model_sel))
    model_summary_string = get_model_summary(model.model)   # conver summary to string
    st.code(model_summary_string)                           # print_summary

    model.compile(loss=loss, metrics=['mae'], optimizer='Adam')

    st.markdown('An {} model will be trained using {} loss, ' 
    '{} as metric and with {} optimizer for {} epochs in batches of {}'
    .format(model_sel, loss, 'MAE', 'Adam', epochs, batch_size))

    placeholder = st.empty()
    placeholder.warning('Training Model, this can take a while')
    model.train_model(train_X=data.train_X, train_y=data.train_y, epochs=epochs,
    batch_size=batch_size, verbose=0, validation_split=0.25, columns=['mae','val_mae'])
    placeholder.success('Model Trained')

    model.plot_model_results()
    
    model.prediction_errors(data.train_X, data.test_X, data.train_y, data.test_y, data.n_features)
    st.subheader('Prediction Errors')
    st.markdown('Mean Absolute Error in Testing {}'.format(model.mae_test))
    st.markdown('Mean Squared Error in Testing {}'.format(model.mse_test))
    st.markdown('R2 Score in Testing {}'.format(model.r2_test))
    st.markdown('Root Mean Squared Error in Testing {}'.format(model.rmse_test))
    
    model.plot_distribution_error(data.train_X, data.test_X, data.train_y, data.test_y)

    if ((model_sel == 'Seq2Seq') or (model_sel == 'Attention')):
        model.plot_single_seq(data.test_X, data.test_y, n_features=data.n_features, index=data.n_steps_out, feature_index=feature_index)
        model.forecast_att(data.train_X, data.train_y)
    else:
        model.plot_simple_model(data.test_X, data.test_y, feature_index=feature_index)
    if save=='Save Model':
        model.serialize_model(name=str(model_sel+'_in_'+n_steps_in+'_out_'+n_steps_out))
        st.sidebar.success('Model Saved')
else:
    model = forecast_model()
    model.load_model(name='model_att_forecast')
    data.drop_feature()
    feature_index = st.sidebar.slider('Select the index of the feature to predict. Check the Dataset to be sure', 0, data.n_features, 0, 1)
    n_steps_in = 324
    n_steps_out = 16
    
    st.sidebar.success('Attention Model Loaded')
    st.markdown('{} Model Summary'.format('Loaded'))
    model_summary_string = get_model_summary(model.model)   # conver summary to string
    st.code(model_summary_string)                           # print_summary
    
    data.create_supervised_mimo(n_steps_in=n_steps_in, n_steps_out=n_steps_out)   # create supervise training set
    
    model.prediction_errors(data.train_X, data.test_X, data.train_y, data.test_y, data.n_features)
    st.subheader('Prediction Errors')
    st.markdown('Mean Absolute Error in Testing {} '.format(model.mae_test))
    st.markdown('Mean Squared Error in Testing {}'.format(model.mse_test))
    st.markdown('R2 Score in Testing {}'.format(model.r2_test))
    st.markdown('Root Mean Squared Error in Testing {}'.format(model.rmse_test))

    model.plot_distribution_error(data.train_X, data.test_X, data.train_y, data.test_y)
    
    model.plot_single_seq(data.test_X, data.test_y, n_features=data.n_features, index=data.n_steps_out, feature_index=feature_index)
    model.forecast_att(train_X=data.train_X, train_y=data.train_y, feature=feature_index, forecast_range=20)



