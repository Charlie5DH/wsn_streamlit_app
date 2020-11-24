## Streamlit Application for Wireless Sensor Network Forecasting

The app can be found in [here](https://share.streamlit.io/charlie5dh/wsn_streamlit_app/main/Streamlit/attention_st.py)

If you want to changel it clone the Repo and follow the instructions.

## Requirements

- Keras 2.0.3
- TensorFlow 2.0.0
- Python3
- sickit-learn 0.18.2, numpy, pandas

## Set up the Python environment

Run `conda env create` to create an environment called `WSN`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries. (`conda env create -f environment.yml`)
We will install Python libraries using `pip-sync`, however, which will let us do three nice things:

Or you can run

```
conda env create --prefix ./env --file environment.yml
```

To create the environment as sub-directory

So, after running `conda env create`, activate the new environment and install the requirements:

```sh
conda activate WSN or conda activate ./env
pip install -r requirements.txt
```

If you add, remove, or need to update versions of some requirements, edit the `.in` files, then run

```
pip-compile requirements.in && pip-compile requirements-dev.in
```

## References

- Antayhua RA, Pereira MD, Fernandes NC, Rangel de Sousa F. Exploiting the RSSI Long-Term Data of a WSN for the RF Channel Modeling in EPS Environments. Sensors. 2020 Jan;20(11):3076.
- https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
- https://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
- https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
- https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
- https://www.machinelearningplus.com/time-series/time-series-analysis-python/
- http://ataspinar.com/
- https://towardsdatascience.com/anomaly-detection-with-time-series-forecasting-c34c6d04b24a

[Journals]: http://cloud.traceback.com.br/wsn/wsn_001/journal_ufsc.html
[Features]: https://drive.google.com/file/d/1FrHvWn6LV07Cr1v8F4M5h3x2uOiuNQNC/view?usp=sharing
[RSSI]: https://drive.google.com/file/d/1CJ2gMGHWHt7aM0wH0L7lAgxefcpQZTRV/view?usp=sharing
[Snapshots]: http://cloud.traceback.com.br/wsn/dashlist_cdsa.html
