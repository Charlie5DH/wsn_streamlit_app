import os
import gdown
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ESSENTIALS_FILENAME = Path(__file__).parents[1].resolve() / 'Data/single_feature.csv'

class Dataset():
    '''
    A class to contain the dataset and add some specific visualitazion
    and normalization functions, also to load a custom dataset.
    '''
    def __init__(self, module='34.B2.9F.A9'):
        self.all_modules = ["00.57.FE.04","00.57.FE.0E","00.57.FE.0F", "00.57.FE.06",
        "00.57.FE.09","00.57.FE.05", "00.57.FE.03", "29.E5.5A.24", "A7.CB.0A.C0",
        "34.B2.9F.A9","01.E9.39.32", "A4.0D.82.38", "9F.8D.AC.91",  "50.39.E2.80"]      
        self.dataset = self._load_dataset(name=module)
        self.features = self._remove_literals().columns
        self.n_features = len(self.features)
        self.init_date = '2019-03-01' 
        self.end_date = '2019-04-20'
        self.train_ratio = 0.8
        self.train_len = int(self.train_ratio * len(self.dataset))
        self.resample_freq = None
        self.normalized_data = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.n_steps_in = 256
        self.n_steps_out = 8
        self.input_shape = None
        self.output_shape = None

    @classmethod
    def data_dir_name(self):
        return Path(__file__).parents[1].resolve() / 'Data'
    
    def set_res_freq(self, frequency='5Min'):
        self.resample_freq = frequency
    
    def _download_dataset(self):
        '''
        Download the original Dataset
        '''
        if os.path.exists(self.data_dir_name() / 'Dataset.csv'):
            return 
        else:
            print('Downloading Dataset')
            share_id = '1FrHvWn6LV07Cr1v8F4M5h3x2uOiuNQNC'
            url = 'https://drive.google.com/uc?id='+share_id
            output = str(self.data_dir_name() / 'Dataset.csv')
            gdown.download(url, output, quiet=False)
    
    def _load_dataset(self, name):
        '''
        Load a dataset from the Data directory
        '''
        path_of_file = str(self.data_dir_name() / name)+'.csv'
        df = pd.read_csv(path_of_file, parse_dates=['Timestamp'], index_col='Timestamp')
        try:
            df = df.drop(['Unnamed: 0'], axis=1)
        except:
            pass
        return df

    def get_features(self, features):
        try:
            return self.dataset[[features]]
        except:
            print('Wrong Feature')
    
    def distribution(self, features='Temp_Mod'):
        '''
        Distribution plot of a feature in dataset 
        '''
        fig = px.histogram(self.dataset, x=features, marginal='box')            
        
        fig.update_layout(showlegend=True, autosize=True,
                 title_text="Distribution Plot",
                 legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                 title={'y':1.0, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                 template='plotly_white')
        st.plotly_chart(fig)
    
    def _get_date_features(self):
        '''
        Function for adding date features to any dataset.
        The function first resets the index considering the Timestamp
        is set as index.
        Adds hour, day, month, week, weekday and daylight features.
        Considers day from 7:00 am to 7:00 Pm.
        '''
        
        data = self.dataset.reset_index(inplace=False)
        data['day'] = data['Timestamp'].dt.day
        data['month'] = data['Timestamp'].dt.month
        data['week'] = data['Timestamp'].dt.week
        data['weekday'] = data['Timestamp'].dt.weekday
        data['hour'] = data['Timestamp'].dt.hour
        data['daylight'] = ((data['hour'] >= 7) & (data['hour'] <= 19)).astype(int)
        data.set_index('Timestamp', drop=True, inplace=True)
        return data

    def date_plot(self, feature='Temp_Mod', date='daylight'):
        data = self._get_date_features()
        fig = px.box(data, x=date, y=feature, color=date)            
        fig.update_layout(showlegend=True, autosize=True,
                title_text="Box Plot of {} by {}".format(feature, date),
                template='plotly_white')
        st.plotly_chart(fig)
    
    def _remove_literals(self):
        return self.dataset.drop(['Module','Type'],axis=1)
    
    def plot_data(self):
        try:
            data = self._remove_literals()
        except:
            data = self.dataset
        fig = px.line(data, title="Sensory Data", template="plotly_white",
              labels=dict(index="Time", value="Data", variable="Sensors"))
        # Update layout properties, Add figure title
        fig.update_layout(showlegend=True, autosize=True,
                        title_text="Sensory Data",
                        legend={'orientation':"h", 'yanchor':"bottom", 'y':1.02, 'xanchor':"right", 'x':0.95},
                        title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                        template='plotly_white')
        st.plotly_chart(fig)

    def _split(self, init_date='2019-03', end_date='2019-04-20', train_ratio=0.8):
        '''
        returns the splitted dataset by date
        '''
        train_data = self.dataset[init_date:end_date]
        return train_data
    
    @st.cache
    def split_dataset(self, init_date='2019-03', end_date='2019-04-20', train_ratio=0.8):
        '''
        Split the dataset by date and
        '''
        self.init_date = init_date
        self.end_date = end_date
        self.train_ratio = train_ratio
        train_data = self._split(init_date=init_date, end_date=end_date, train_ratio=train_ratio)
        self.dataset = train_data
        self.train_len = int(train_ratio * len(train_data))

    def _resample(self, freq='5Min'):
        '''
        Resample and split the dataset by date and ratio.
        freq: frequency to resample
        '''
        # The data is higly irregular so let's resample it to 10 min and take the mean
        resampled = self.dataset.resample(freq).mean()
        resampled = resampled.fillna(resampled.bfill())

        return resampled

    def _normalize(self):
        # Normalize
        scaler = MinMaxScaler()
        normalized_train = scaler.fit_transform(self.dataset)
        return normalized_train
    
    @st.cache
    def resample_dataset(self, freq='5Min'):
        self.resample_freq = freq
        self.dataset = self._resample(freq=freq)
        self.train_len = int(self.train_ratio * len(self.dataset))
    
    def normalize_dataset(self):
        self.normalized_data = self._normalize()

    def _split_sequences_mimo(self, train=True, n_steps_in=256, n_steps_out=8):
        '''
        Split a multivariate sequence into samples for multiple 
        Split the training set into segments of a specified timestep
        and creates the labels.
        The Function can be called without using the normalizing first
        '''
        if self.normalized_data is None:
            sequences = self._resample()
            sequences = self._normalize()
        if self.normalized_data is not None:
            sequences = self.normalized_data
        if train==True:
            sequences = sequences[:self.train_len]
        else:
            sequences = sequences[self.train_len:]
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequences):
                break
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def _split_sequences(self, train=True, n_steps_in=256):

        '''
        Split a multivariate sequence into samples for
        Taken and adapted from Machinelearningmastery.
        Split the training set into segments of a specified timestep
        and creates the labels.
        '''
        if self.normalized_data is None:
            sequences = self._resample()
            sequences = self._normalize()
            if train:
                sequences = sequences[:self.train_len]
            else:
                sequences = sequences[self.train_len:]
        else:
            if train:
                sequences = self.normalized_data[:self.train_len]
            else:
                sequences = self.normalized_data[self.train_len:]
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the dataset
            if end_ix > len(sequences)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)    
    
    @st.cache
    def create_supervised_mimo(self, n_steps_in=256, n_steps_out=8):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.input_shape = (n_steps_in, self.n_features)
        self.output_shape = (n_steps_out, self.n_features)
        self.train_X, self.train_y = self._split_sequences_mimo(train=True, n_steps_in=n_steps_in, n_steps_out=n_steps_out)
        self.test_X, self.test_y = self._split_sequences_mimo(train=False, n_steps_in=n_steps_in, n_steps_out=n_steps_out)

    @st.cache
    def create_supervised(self, n_steps_in=256):
        self.train_X, self.train_y = self._split_sequences(train=True, n_steps_in=n_steps_in)
        self.test_X, self.test_y = self._split_sequences(train=False, n_steps_in=n_steps_in)
        self.n_steps_in = n_steps_in
        self.input_shape = (n_steps_in, self.n_features)
    
    def drop_feature(self, feature_name='VBus'):
        self.dataset = self.dataset.drop([feature_name], axis=1)
        self.features = self.dataset.columns
        self.n_features = len(self.features)
        if self.normalized_data is not None:
            self.normalize_dataset()

def main():
    data = Dataset(module="50.39.E2.80")
    print(data.dataset.head())
    print(data.train_len)
    print(data.train_ratio)
    
    print('\n Splitting Dataset')
    #print(data.features)
    data.split_dataset(init_date='2019-01-01', end_date='2019-04-20')
    print(data.dataset.head())
    print('Train Len ', data.train_len)
    print('Data len', data.dataset.shape)
    print('Train Ratio ', data.train_ratio)

    print('\n Resampled dataset')
    data.resample_dataset(freq='10Min')
    print(data.dataset.head())
    print('Reample Freq ', data.resample_freq)
    
    print('\n Normalized Data')
    data.normalize_dataset()
    print(data.normalized_data)
    print(data.normalized_data.shape)

    print('\n Supervised Train')
    data.create_supervised_mimo()
    print(data.normalized_data[:data.train_len])
    print(data.normalized_data[data.train_len:])
    print('train_X shape ', data.train_X.shape)
    print('train_y shape ', data.train_y.shape)
    print('test_X shape ', data.test_X.shape)
    print('test_y shape ', data.test_y.shape)

    print('\n Supervised Train')
    data.create_supervised()
    print(data.normalized_data[:data.train_len])
    print(data.normalized_data[data.train_len:])
    print('train_X shape ', data.train_X.shape)
    print('train_y shape ', data.train_y.shape)
    print('test_X shape ', data.test_X.shape)
    print('test_y shape ', data.test_y.shape)

    print(data.n_features)

    print(data.features.values)
    #data.distribution(features=['Temp_Mod'])
    #data.date_plot(date='month')
    #data.plot_data()


if __name__ == "__main__":
    main()