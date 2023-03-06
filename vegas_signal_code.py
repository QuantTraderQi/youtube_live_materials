import numpy as np
import pandas as pd
import ccxt
import time
import dateutil
from datetime import datetime
from functools import reduce
from scipy.signal import argrelextrema
from ta import add_all_ta_features
import ta

# define the market
exchange_f = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # or 'margin'
    }})

symbol = 'NEO/USDT'
timeframe = "1h"
limit = 1000
df = pd.DataFrame(exchange_f.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
df['symbol'] = symbol
df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol', 'Symbol']

df['Datetime'] = df['Datetime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(x / 1000.)))

df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['ema_144'] = df['Close'].ewm(span=144, adjust=False).mean()
df['ema_169'] = df['Close'].ewm(span=169, adjust=False).mean()
df['ema_576'] = df['Close'].ewm(span=576, adjust=False).mean()
df['ema_676'] = df['Close'].ewm(span=676, adjust=False).mean()

df['ema_12_lag'] = df['ema_12'].shift(1)
df['ema_144_lag'] = df['ema_144'].shift(1)

# Check if the conditions for a bullish trend are met
is_bullish_trend = (
    (df['ema_12_lag'] < df['ema_144_lag'])
    & (
        (df['Close'] > df['ema_12'])
        & (df['ema_12'] > df['ema_144'])
        & (df['ema_144'] > df['ema_169'])
        & (df['ema_169'] > df['ema_576'])
        & (df['ema_576'] > df['ema_676'])
    )
)

# Update the 'is_bullish' column
df.loc[is_bullish_trend, 'is_bullish'] = True

df[df['is_bullish']==True]
