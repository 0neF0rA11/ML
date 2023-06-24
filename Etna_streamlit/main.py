import numpy as np
import pandas as pd
import copy

import streamlit as st

from etna.models import CatBoostPerSegmentModel
from etna.pipeline import Pipeline
from etna.metrics import SMAPE
from etna.datasets import TSDataset
from etna.transforms import MeanTransform, MedianTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import LogTransform
from etna.analysis import plot_backtest, plot_forecast


@st.cache_data
def generate_df(segments, mean, dispersion):
    lst = []
    np.random.seed(42)
    for i in range(segments):
        df = pd.DataFrame({"timestamp": pd.date_range("20230101", periods=365),
                           "segment": f"segment_{i}",
                           "target": list(map(int, np.random.normal(mean, dispersion, 365)))})
        lst.append(df)
    result = pd.concat(lst, axis=0)
    result.loc[result["target"] < 0, 'target'] = 0
    df3 = TSDataset.to_dataset(result)
    return TSDataset(df3, freq="D")


st.title("A brief introduction to etna")

st.write("Generation dataset")
count_segments = st.slider("Count segments", min_value=2, max_value=10, value=5, step=1)
col1, col2 = st.columns(2)
middle = col1.number_input("Mean value in segments", min_value=5, value=30, step=1)
disp = col2.number_input("Dispersion", min_value=1, max_value=50, value=5, step=1)

ts = generate_df(count_segments, middle, disp)
st.write(ts.head())

transform_list = []
with st.sidebar:
    st.header("Choose modernization")
    if st.checkbox("LagTransform"):
        if st.checkbox("&emsp;Range"):
            col1, col2 = st.columns(2)
            left = col1.number_input("Left", min_value=1, max_value=30, value=1, step=1)
            right = col1.number_input("Right", min_value=left, max_value=31, value=left+1, step=1)
            transform_list.append(
                LagTransform(in_column="target", out_column="lags", lags=list(range(left, right + 1))))
        if st.checkbox("&emsp;Selection"):
            value = st.number_input("Lag number", min_value=1, max_value=31, value=1, step=1)
            try:
                transform_list.append(LagTransform(in_column="target", out_column="lags", lags=[value]))
            except ValueError:
                pass

    if st.checkbox("MeanTransform"):
        col3, col4 = st.columns(2)
        seasonality = col3.number_input("seasonality", min_value=1, value=1, max_value=31, step=1, key="s1")
        window = col4.number_input("Window", min_value=2, max_value=365 // seasonality, value=2, step=1, key="w1")
        transform_list.append(
            MeanTransform(in_column="target", out_column=f"mean_{window}_{seasonality}", window=window,
                          seasonality=seasonality, alpha=0.5))

    if st.checkbox("MedianTransform"):
        col5, col6 = st.columns(2)
        seasonality = col5.number_input("seasonality", min_value=1, value=1, max_value=31, step=1, key="s2")
        window = col6.number_input("Window", min_value=2, max_value=365 // seasonality, value=2, step=1, key="w2")
        transform_list.append(
            MedianTransform(in_column="target", out_column=f"median_{window}_{seasonality}", window=window,
                            seasonality=seasonality))

    if st.checkbox("LogTransform"):
        transform_list.append(LogTransform(in_column="target", inplace=st.checkbox("Is inplace?"), out_column="Log"))

    if st.checkbox("DateFlagTransform"):
        try:
            transform_list.append(DateFlagsTransform(out_column="date_flags",
                                                     day_number_in_week=st.checkbox("&emsp;Day number in week"),
                                                     day_number_in_month=st.checkbox("&emsp;Day number in month"),
                                                     week_number_in_month=st.checkbox("&emsp;Week number in month"),
                                                     is_weekend=st.checkbox("&emsp;Is weekend")))
        except ValueError:
            st.caption("&emsp;&emsp;Select at least 1 item")

st.title("Transformed dataset")
copy_ts = copy.deepcopy(ts)
copy_ts.fit_transform(transform_list)
st.write(copy_ts.head())

HORIZON = st.slider("Horizon", min_value=7, max_value=184, value=7, step=1)

model = CatBoostPerSegmentModel()
pipe = Pipeline(model=model, transforms=transform_list, horizon=HORIZON)
smape = SMAPE()

fit_button = st.button("Fit")
if fit_button:
    metrics, forecast, info = pipe.backtest(ts=ts, metrics=[smape], n_folds=1)
    st.write("SMAPE: ")
    st.write(metrics)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Back test")
    st.pyplot(plot_backtest(forecast_df=forecast, ts=ts))
    st.title(f"Forecast forecast for {HORIZON} days")
    pipe.fit(ts)
    forecast = pipe.forecast()
    st.pyplot(plot_forecast(forecast_ts=forecast, train_ts=ts, n_train_samples=365))
