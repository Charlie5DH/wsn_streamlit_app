import os
import gdown
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ESSENTIALS_FILENAME = Path(__file__).parents[1].resolve() / 'Data/single_feature.csv'

class myCallback(tf.keras.callbacks.Callback):
  def __init__(self, mae=0.03):
    super(myCallback, self).__init__()
    self.mae = mae

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_mae') <= self.mae):
      print("\nReached 0.03 mae so cancelling training!")
      self.model.stop_training = True

class forecast_model():
    '''
    A class to contain the dataset and add some specific visualitazion
    and normalization functions, also to load a custom dataset.
    '''
    def __init__(self, model_type='LSTM'):
        self.kind = model_type
        self.model = None
        self.epochs = 100
        self.batch_size = 256
        self.loss = 'mean_squared_error'
        self.metrics = ['mae']
        self.optimizer = tf.keras.optimizers.Adam
        self.history = None
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        self.mae_train = None
        self.mae_test = None
        self.mae_overall_train = None
        self.mae_overall_test = None 
        self.mse_test = None
        self.r2_test = None
        self.rmse_test = None
    
    @classmethod
    def data_dir_name(self):
        return Path(__file__).parents[1].resolve() / 'Model'
    
    def LSTM(self, input_shape, units=32, dropout=0.2):
        '''Create LSTM model'''
        n_features = input_shape[-1]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=units, input_shape=input_shape,
                                            return_sequences=True, dropout=dropout))
        model.add(tf.keras.layers.LSTM(units=units, return_sequences=False,
                                        dropout=dropout))
        #model_LSTM.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(n_features))
        return model

    def GRU(self, input_shape, units=64, dropout=0.2):
        '''Create LSTM model'''
        n_features = input_shape[-1]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(units=units, dropout=0.2, return_sequences=True, input_shape=input_shape)) 
        model.add(tf.keras.layers.GRU(units=units, input_shape=input_shape,
                                            return_sequences=False, dropout=dropout))
        #model_LSTM.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(n_features))
        return model

    def DNN(self, input_shape, units=64, dropout=0.2):
        
        model = tf.keras.Sequential()
        #model.add(tf.keras.layers.Reshape(input_shape, input_shape=(input_shape,)))
        model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=input_shape[1]))
        return model

    def CNN(self, input_shape, filters=64, kernel_size=3, dropout=0.2):

        n_features =  input_shape[-1]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape,
                                            activation='relu')) 
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(n_features))
        return model

    def Seq2Seq(self, input_shape, output_shape, dropout=0.2, momentum=0.9, n_hidden=128):

        input_train = tf.keras.layers.Input(shape=input_shape)
        output_train = tf.keras.layers.Input(shape=output_shape)

        encoder_last_h1, encoder_last_h2, encoder_last_c = \
        tf.keras.layers.LSTM(n_hidden, activation='relu', dropout=dropout,
                            return_sequences=False,
                            return_state=True)(input_train)

        # Batch normalisation is added because we want to avoid gradient
        # explosion caused by the activation function ELU in the encoder.
        encoder_last_h1 = tf.keras.layers.BatchNormalization(momentum=0.9)(encoder_last_h1)
        encoder_last_c = tf.keras.layers.BatchNormalization(momentum=0.9)(encoder_last_c)

        # Create copies of las hidden state
        decoder = tf.keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h1)

        # nitial_state: List of initial state tensors to be passed to the first call of the cell 
        #(optional, defaults to None which causes creation of zero-filled initial state tensors).
        # In this case initial state is the output from encoder
        decoder = tf.keras.layers.LSTM(n_hidden, activation='relu', dropout=dropout,
                                    return_state=False,
                                    return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])

        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_train.shape[2]))(decoder)
        model = tf.keras.Model(inputs=input_train, outputs=out)
        return model

    def Attention(self, input_shape, output_shape, dropout=0.2, momentum=0.9, n_hidden=128):

        input_train = tf.keras.layers.Input(shape=input_shape)
        output_train = tf.keras.layers.Input(shape=output_shape)

        encoder_stack_h, encoder_last_h, encoder_last_c = \
        tf.keras.layers.LSTM(n_hidden, activation='tanh', 
                            return_sequences=True,
                            return_state=True)(input_train)

        #encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        #encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        # Repeat the last hidden state of encoder 20 times, and use them as input to decoder LSTM.
        decoder_input = tf.keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h)

        decoder_stack_h = tf.keras.layers.LSTM(n_hidden, activation='tanh', return_state=False,
                                            return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        attention = tf.keras.layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = tf.keras.layers.Activation('softmax', name='Softmax')(attention)
        context = tf.keras.layers.dot([attention, encoder_stack_h], axes=[2,1])
        #context = BatchNormalization(momentum=0.6)(context)
        # Now we concat the context vector and stacked hidden states of decoder, 
        # and use it as input to the last dense layer.
        decoder_combined_context = tf.keras.layers.concatenate([context, decoder_stack_h])
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_train.shape[2]))(decoder_combined_context)
        model = tf.keras.models.Model(inputs=input_train, outputs=out)
        return model

    def set_model(self, input_shape, output_shape=None, model_type='LSTM', 
    filters=64, units=64, dropout=0.2, momentum=0.9, kernel_size=3):
        if model_type == 'LSTM':
            self.model = self.LSTM(input_shape, units=units, dropout=dropout)
        if model_type == 'GRU':
            self.model = self.GRU(input_shape, units=units, dropout=dropout)
        if model_type == 'DNN':
            self.model = self.DNN(input_shape, units=units, dropout=dropout)
        if model_type == 'CNN':
            self.model = self.CNN(input_shape, filters=filters, dropout=dropout, kernel_size=kernel_size)
        if model_type == 'Seq2Seq':
            self.model = self.Seq2Seq(input_shape, output_shape, dropout=dropout, momentum=momentum, n_hidden=units)
        if model_type == 'Attention':
            self.model = self.Attention(input_shape, output_shape, dropout=dropout, momentum=momentum, n_hidden=units)
        self.kind = model_type

    def summary(self):
        return self.model.summary()

    def compile(self, loss='mean_squared_error', metrics=['mae'], optimizer='Adam'):
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train_model(self, train_X, train_y, epochs=100, batch_size=256, 
    patience=5, verbose=1, validation_split=0.25, columns=['mae','val_mae'],
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]):
        
        self.callbacks = callbacks
        
        if verbose==0:
            print('Training model')
        history = self.model.fit(train_X, train_y, validation_split=validation_split,
                                epochs=epochs, verbose=verbose, callbacks=callbacks, 
                                batch_size=batch_size)
        self.history = pd.DataFrame(history.history, columns=columns)

    def _get_errors(self, train_X, test_X, train_y, test_y, n_features):
        
        train_predictions = self.model.predict(train_X)
        test_predictions = self.model.predict(test_X)
        mae_train = abs(train_predictions - train_y).mean()
        mae_test = abs(test_predictions - test_y).mean()
        
        if train_y.shape[1] == n_features:
            mae_overall_train = abs(train_predictions-train_y).mean(axis=(1))
            mae_overall_test = abs(test_predictions-test_y).mean(axis=(1))
            mse_test = mean_squared_error(test_y, test_predictions)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(test_y, test_predictions)
        else:
            mae_overall_train = abs(train_predictions-train_y).mean(axis=(1,2))
            mae_overall_test = abs(test_predictions-test_y).mean(axis=(1,2))
            mse_test = mean_squared_error(test_y.reshape(-1,n_features), test_predictions.reshape(-1,n_features))
            r2_test = r2_score(test_y.reshape(-1,n_features), test_predictions.reshape(-1,n_features)) 
        rmse_test = np.sqrt(mse_test)
        return mae_train, mae_test, mae_overall_train, mae_overall_test, mse_test, r2_test, rmse_test

    def prediction_errors(self, train_X, test_X, train_y, test_y, n_features):
        self.mae_train, self.mae_test, self.mae_overall_train, self.mae_overall_test, self.mse_test, self.r2_test,self.rmse_test = self._get_errors(train_X, test_X, train_y, test_y, n_features)

    def plot_model_results(self):

        fig = px.line(self.history, title="Training Results", template="plotly_white",
              labels=dict(index='Epochs',value="Error", variable="Metrics"))
        # Update layout properties, Add figure title
        fig.update_layout(showlegend=True, autosize=True,
                        title_text="Training History",
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        st.plotly_chart(fig)

    def plot_mae_overall(self):

        fig = px.line(self.mae_overall_test, title="Error", template="plotly_white",
              labels=dict(index='Time',value="Error", variable="error"))
        # Update layout properties, Add figure title
        fig.update_layout(showlegend=True, autosize=True,
                        title_text="Error Over Testing Set",
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        st.plotly_chart(fig)

    def plot_simple_model(self, X, y, feature_index=0):

        prediction = self.model.predict(X)
        pred_df = pd.DataFrame(data= np.concatenate((prediction[:,feature_index].reshape(-1,1),
        y[:,feature_index].reshape(-1,1)), axis=1), columns=['Prediction', 'True'])

        fig = px.line(pred_df, title="Prediction and Truth", template="plotly_white",
        labels=dict(index='Time',value="Data", variable="variable"))
        # Update layout properties, Add figure title
        fig.update_layout(showlegend=True, autosize=True,title_text='Prediction and Truth',
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        st.plotly_chart(fig)

    def plot_single_seq(self, X, y, feature_index=0, n_features=4, index=8):
        index = index-1
        prediction = self.model.predict(X)[:,index,:]
        y = y[:,index,:]
        pred_df = pd.DataFrame(data= np.column_stack((prediction.reshape(-1,n_features)[:,feature_index],
                                                    y.reshape(-1,n_features)[:,feature_index])), 
                                                    columns=['Prediction', 'True'])
        
        fig = px.line(pred_df, title="Prediction and Truth", template="plotly_white",
        labels=dict(index='Time',value="Data", variable="variable"))
        # Update layout properties, Add figure title
        fig.update_layout(showlegend=True, autosize=True,title_text='Prediction and Truth',
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        st.plotly_chart(fig)

    def plot_distribution_error(self, train_X, test_X, train_y, test_y):
        #train_predictions = self.model.predict(train_X)
        test_predictions = self.model.predict(test_X)           

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=test_predictions.flatten(), name='Predictions'))
        fig.add_trace(go.Histogram(x=test_y.flatten(), name='Real Values'))
         
        fig.update_layout(showlegend=True, autosize=True, barmode='overlay',
                        title_text="Histogram of Prediction and Truth",
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig)

    def forecast_att(self, train_X, train_y, forecast_range=20, feature=0):

        n_timesteps_in = train_X.shape[1]
        n_timesteps_out = train_y.shape[1]
        n_features = train_X.shape[2]
        batch = train_X[-1, :, :]
        predictions_forecast = batch
        
        # future timesteps to forcast times X horizon timesteps
        for i in range(forecast_range):
            one_step_pred = self.model.predict(batch.reshape(1, n_timesteps_in, n_features))
            batch = np.concatenate([batch[n_timesteps_out:], one_step_pred.reshape(n_timesteps_out, n_features)], axis=0)
            predictions_forecast = np.row_stack([predictions_forecast, one_step_pred.reshape(n_timesteps_out, n_features)])
        
        forecast = np.concatenate([train_y[:, n_timesteps_out-1, :], predictions_forecast], axis=0)
        forecast[:train_y.shape[0]] = None
        
        fig = make_subplots()
        fig.add_trace(
            go.Scatter(x=np.arange(0, train_y[:, 0, feature].shape[0], 1), y=train_y[:, 0, feature],
                    mode='lines', name='Training Values'))
        fig.add_trace(               
            go.Scatter(x=np.arange(0, forecast[:,feature].shape[0], 1), y=forecast[:,feature],
                    mode='lines', name='Forecast'))   
        fig.update_xaxes(title_text="<b>Time</b>")
        fig.update_yaxes(title_text="<b>Data</b>")
        fig.update_layout(showlegend=True, autosize=True,
                        title_text="Forecast",
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        st.plotly_chart(fig)

    def serialize_model(self, name='model'):
        '''
        Save model and history
        '''
        dir_path = self.data_dir_name()
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(dir_path / str(name+'.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(dir_path / str(name+'.h5'))
        self.history.to_csv(dir_path / str(name+'.csv'))
        print("Saved model to disk")

    def load_model(self, name='model'):
        '''
        Load Model from disk
        '''
        dir_path = self.data_dir_name()
        json_file = open(dir_path / str(name+'.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(dir_path / str(name+'.h5'))
        print("Loaded model from disk")
        self.model = loaded_model


def main():
    model = forecast_model(model_type='LSTM')
    model.set_model(input_shape=(100,3), model_type='LSTM')
    print(model.kind)
    print(model.summary())

if __name__ == "__main__":
    main()