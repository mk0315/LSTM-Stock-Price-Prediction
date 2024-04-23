import numpy as np
import pandas as pd
import pandas_datareader as data
import streamlit as st
import plotly.graph_objects as go
import sklearn
from keras.models import load_model

start = '2010-01-01'
end = '2024-04-08'

st.title('Stock Trend Predictor📈📉')

user_input = st.text_input('Enter Stock ticker', 'TSLA')

df = data.get_data_tiingo(user_input, api_key='cd979b3785f284399d665e0f55251c969f818226', start=start, end=end)
df = df.reset_index()
# Showing dataframe to user
st.subheader('Data from 2010 to 2024')
st.write(df.describe())

st.subheader('Closing price vs Time chart')
fig = go.Figure()

# Add a line trace for the closing price vs. time
fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Closing Price'))

# Update layout
fig.update_layout(title='Closing Price vs Time Chart', xaxis_title='Date', yaxis_title='Closing Price')

# Display the Plotly figure
st.plotly_chart(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200 Moving Avg')
fig2 = go.Figure()

# Add trace for the closing price
fig2.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close'))
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
# Add trace for the moving average (assuming ma100 and ma200 are already defined)
fig2.add_trace(go.Scatter(x=df['date'], y=ma100, mode='lines', name='Moving Average (100 days)', line=dict(color='red')))
fig2.add_trace(go.Scatter(x=df['date'], y=ma200, mode='lines', name='Moving Average (200 days)', line=dict(color='green')))

# Update layout
fig2.update_layout(title='Stock Close Price with Moving Averages',
                  xaxis_title='Date',
                  yaxis_title='Price')

st.plotly_chart(fig2)

# Splitting data into training and testing
data_training = pd.DataFrame(df['close'][0: int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df) * 0.70): int(len(df))])

# Using Min Max scaler since LSTMS are scale variant
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Loading LSTM model
model = load_model('LSTM_Model_New.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# Inverse transform the predicted values to the original scale
scale_factor = 1 / scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Get the dates corresponding to the test data
test_dates = df['date'].tail(len(data_testing))

# Final Graph
st.subheader('Predictions vs Original')
trace1 = go.Scatter(x=test_dates, y=data_testing['close'], mode='lines', name='Original Price', line=dict(color='blue'))
trace2 = go.Scatter(x=test_dates, y=y_predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='orange'))

# Create layout
layout = go.Layout(title='Original vs Predicted Price', xaxis=dict(title='Time'), yaxis=dict(title='Price'))

# Create figure
fig3 = go.Figure(data=[trace1, trace2], layout=layout)

# Show plot
st.plotly_chart(fig3)
