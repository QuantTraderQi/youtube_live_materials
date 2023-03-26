# 最前面放几个我写的函数：
# * collect_data: 抓去币圈所有k线数据
# * plot_pat: k线作图加上局部高低点的连线
# * 和谐扫描函数，呕心沥血的集大成之作
# * detect_harmonic

def collect_data(timeframe='4h', limit=500):
    
    # This function downloads candlestick data from
    # binance futures market
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

    all_coins_f = list(exchange_f.load_markets().keys())
    coins = [x for x in all_coins_f if "/USDT" in x]
    coins = [x for x in coins if "_" not in x]

    all_candles_f = []
    for symbol in coins:

        try:

            df = pd.DataFrame(exchange_f.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
            df['symbol'] = symbol

            df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol', 'Symbol']

            df['Datetime'] = df['Datetime'].apply(
                lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(x / 1000.)))

            df['momentum_rsi'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
            df['rsi_21'] = ta.momentum.RSIIndicator(df['Close'], 21).rsi()
            df['rsi_21_sma_55'] = df['rsi_21'].rolling(window=55).mean()

            df = df[:-1]

            df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['ema_144'] = df['Close'].ewm(span=144, adjust=False).mean()
            df['ema_169'] = df['Close'].ewm(span=169, adjust=False).mean()
            df['ema_576'] = df['Close'].ewm(span=576, adjust=False).mean()
            df['ema_676'] = df['Close'].ewm(span=676, adjust=False).mean()

            all_candles_f.append(df)
        except:
            pass

    all_candles_f = pd.concat(all_candles_f)
    all_candles_f['Datetime'] = pd.to_datetime(all_candles_f['Datetime'])

    return all_candles_f

def plot_pat(data, symbol, order = 10):
    import numpy as np
    import pandas as pd
    from scipy.signal import argrelextrema
    import plotly.graph_objects as go

    data = data[data['Symbol'] == symbol]

    dt = data.Datetime

    data = data.set_index('Datetime')

    price = data.Close

    high = data.High

    low = data.Low

    max_idx = list(argrelextrema(high.values, np.greater, order=order)[0])
    min_idx = list(argrelextrema(low.values, np.less, order=order)[0])

    peak_1 = high.values[max_idx]
    peak_2 = low.values[min_idx]

    peaks_p = list(peak_1) + list(peak_2)

    peaks_idx = list(max_idx) + list(min_idx)

    peaks_idx_dt = np.array(dt.values[peaks_idx])

    peaks_p = np.array(list(peak_1) + list(peak_2))

    final_data = pd.DataFrame({"price": peaks_p, "datetime": peaks_idx_dt})

    final_data = final_data.sort_values(by=['datetime'])

    peaks_idx_dt = final_data.datetime

    peaks_p = final_data.price

    current_idx = np.array(list(final_data.datetime[-4:]) + list(dt[-1:]))

    current_pat = np.array(list(final_data.price[-4:]) + list(low[-1:]))


    start = min(current_idx)

    end = max(current_idx)

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]

    data = data.reset_index()
    data_n = data[data['Datetime'] >= start]

    candlestick = go.Candlestick(x=data_n['Datetime'], open=data_n['Open'], high=data_n['High'], low=data_n['Low'],
                                    close=data_n['Close'])

    pat = go.Scatter(x=current_idx, y=current_pat, line={'color': 'blue'})

    fig = go.Figure(data=[candlestick, pat])

    fig.layout.xaxis.type = 'category'

    fig.update_layout(
        width=800,
        height=600)

    fig.layout.xaxis.rangeslider.visible = False
    fig.show()

    def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1


def plot_pat(data, symbol, order = 10):
    import numpy as np
    import pandas as pd
    from scipy.signal import argrelextrema
    import plotly.graph_objects as go

    data = data[data['Symbol'] == symbol]

    dt = data.Datetime

    data = data.set_index('Datetime')

    price = data.Close

    high = data.High

    low = data.Low

    max_idx = list(argrelextrema(high.values, np.greater, order=order)[0])
    min_idx = list(argrelextrema(low.values, np.less, order=order)[0])

    peak_1 = high.values[max_idx]
    peak_2 = low.values[min_idx]

    peaks_p = list(peak_1) + list(peak_2)

    peaks_idx = list(max_idx) + list(min_idx)

    peaks_idx_dt = np.array(dt.values[peaks_idx])

    peaks_p = np.array(list(peak_1) + list(peak_2))

    final_data = pd.DataFrame({"price": peaks_p, "datetime": peaks_idx_dt})

    final_data = final_data.sort_values(by=['datetime'])

    peaks_idx_dt = final_data.datetime

    peaks_p = final_data.price

    current_idx = np.array(list(final_data.datetime[-4:]) + list(dt[-1:]))

    current_pat = np.array(list(final_data.price[-4:]) + list(low[-1:]))


    start = min(current_idx)

    end = max(current_idx)

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]

    data = data.reset_index()
    data_n = data[data['Datetime'] >= start]

    candlestick = go.Candlestick(x=data_n['Datetime'], open=data_n['Open'], high=data_n['High'], low=data_n['Low'],
                                    close=data_n['Close'])

    pat = go.Scatter(x=current_idx, y=current_pat, line={'color': 'blue'})

    fig = go.Figure(data=[candlestick, pat])

    fig.layout.xaxis.type = 'category'

    fig.update_layout(
        width=800,
        height=600)

    fig.layout.xaxis.rangeslider.visible = False
    fig.show()


def peak_detect(df, order=10):
    # this function is to detect four price peaks,then
    # combine them with latest data point to define four
    # moiving segments for harmonic pattern detection.
    import numpy as np
    import pandas as pd
    import ccxt
    import time
    import dateutil
    from datetime import datetime
    from functools import reduce
    from scipy.signal import argrelextrema

    dt = df.Datetime

    df = df.set_index('Datetime')

    price = df.Close

    high = df.High

    low = df.Low

    max_idx = list(argrelextrema(high.values, np.greater, order=order)[0])
    min_idx = list(argrelextrema(low.values, np.less, order=order)[0])

    peak_1 = high.values[max_idx]
    peak_2 = low.values[min_idx]

    peaks_p = list(peak_1) + list(peak_2)

    peaks_idx = list(max_idx) + list(min_idx)

    peaks_idx_dt = np.array(dt.values[peaks_idx])

    peaks_p = np.array(list(peak_1) + list(peak_2))

    final_df = pd.DataFrame({"price": peaks_p, "datetime": peaks_idx_dt})

    final_df = final_df.sort_values(by=['datetime'])

    peaks_idx_dt = final_df.datetime

    peaks_p = final_df.price

    current_idx = np.array(list(final_df.datetime[-4:]) + list(dt[-1:]))

    current_pat = np.array(list(final_df.price[-4:]) + list(low[-1:]))


    start = min(current_idx)

    end = max(current_idx)

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]
    moves = [XA, AB, BC, CD]
    symbol = df['Symbol'].unique().tolist()
    return current_idx, current_pat, start, end, moves, high, low, final_df, symbol


def bull_bat(moves, symbol, current_pat):
    try:
        err_allowed = 0.1
        XA = moves[0]
        AB = moves[1]
        BC = moves[2]
        CD = moves[3]

        M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
        W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

        AB_range = np.array([0.382 - err_allowed, 0.5 + err_allowed]) * abs(XA)
        BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)
        PRZ = round(current_pat[1] - 0.886*abs(XA),4)
        SL = round(current_pat[0],4)
        TP1 = round(current_pat[3] - 0.618*abs(CD),4)
        TP2 = round(current_pat[3] - 0.5*abs(CD),4)
        TP3 = round(current_pat[3] - 0.382*abs(CD),4)        

        bat_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 0.7*abs(XA) and abs(CD) <= CD_range[1])

        if M_pat and bat_pat:
            return listToString(symbol).replace("/USDT", ""), PRZ,SL, TP1, TP2, TP3

        else:
            return ([])
    except Exception as e:
        return ([])


def bear_bat(moves, symbol, current_pat):
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.382 - err_allowed, 0.5 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] + 0.886*abs(XA),4)
    SL = round(current_pat[0],4)
    TP1 = round(current_pat[3] + 0.618*abs(CD),4)
    TP2 = round(current_pat[3] + 0.5*abs(CD),4)
    TP3 = round(current_pat[3] + 0.382*abs(CD),4)

    bat_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 0.7*abs(XA) and abs(CD) <= CD_range[1])
    # bat_pat = True

    if W_pat and bat_pat:
        return listToString(symbol).replace("/USDT", ""), PRZ, SL, TP1, TP2, TP3

    else:
        return ([])


def bull_gartley(moves, symbol, current_pat):
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.13 - err_allowed, 1.618 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] - 0.786*abs(XA), 4)
    SL = round(current_pat[0],4)
    TP1 = round(current_pat[3] - 0.618*abs(CD),4)
    TP2 = round(current_pat[3] - 0.5*abs(CD),4)
    TP3 = round(current_pat[3] - 0.382*abs(CD),4)

    gartley_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 0.6*abs(XA) and abs(CD) <= CD_range[1])
    # gartley_pat = True

    if M_pat and gartley_pat:
        return listToString(symbol).replace("/USDT", ""), PRZ, SL, TP1, TP2, TP3

    else:
        return ([])


def bear_gartley(moves, symbol, current_pat):
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.13 - err_allowed, 1.618 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] + 0.786*abs(XA), 4)
    SL = round(current_pat[0],4)
    TP1 = round(current_pat[3] + 0.618*abs(CD),4)
    TP2 = round(current_pat[3] + 0.5*abs(CD),4)
    TP3 = round(current_pat[3] + 0.382*abs(CD),4)
    gartley_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 0.6*abs(XA) and abs(CD) <= CD_range[1])

    if W_pat and gartley_pat:
        return listToString(symbol).replace("/USDT", ""), PRZ,SL, TP1, TP2, TP3

    else:
        return ([])


def bull_crab(moves, symbol, current_pat):
    import numpy as np
    import pandas as pd
    import ccxt
    import time
    import dateutil
    from datetime import datetime
    from functools import reduce
    from scipy.signal import argrelextrema
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([2.618 - err_allowed, 3.618 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] - 1.618*abs(XA),4)
    SL = PRZ*0.98
    TP1 = round(current_pat[3] - 0.618*abs(CD),4)
    TP2 = round(current_pat[3] - 0.5*abs(CD),4)
    TP3 = round(current_pat[3] - 0.382*abs(CD),4)

    crab_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 1.3*abs(XA) and abs(CD) <= CD_range[1])

    if M_pat and crab_pat:
        return listToString(symbol).replace("/USDT", ""), PRZ, SL, TP1, TP2, TP3

    else:
        return ([])


def bear_crab(moves, symbol, current_pat):
    import numpy as np
    import pandas as pd
    import ccxt
    import time
    import dateutil
    from datetime import datetime
    from functools import reduce
    from scipy.signal import argrelextrema    
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.382 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([2.618 - err_allowed, 3.618 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] + 1.618*abs(XA),4)
    SL = PRZ*1.02
    TP1 = round(current_pat[3] + 0.618*abs(CD),4)
    TP2 = round(current_pat[3] + 0.5*abs(CD),4)
    TP3 = round(current_pat[3] + 0.382*abs(CD),4)

    crab_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 1.3*abs(XA) and abs(CD) <= CD_range[1])

    if W_pat and crab_pat:
        return listToString(symbol).replace("/USDT", ""), PRZ, SL, TP1, TP2, TP3

    else:
        return ([])


def bull_butterfly(moves, symbol, current_pat):
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.24 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] - 1.27*abs(XA),4)
    SL = PRZ*0.98
    TP1 = round(current_pat[3] - 0.618*abs(CD),4)
    TP2 = round(current_pat[3] - 0.5*abs(CD),4)
    TP3 = round(current_pat[3] - 0.382*abs(CD),4)

    butterfly_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 1.2*abs(XA) and abs(CD) <= CD_range[1])

    if M_pat and butterfly_pat:
        return listToString(symbol).replace("/USDT", ""), PRZ, SL, TP1, TP2, TP3
    else:
        return ([])


def bear_butterfly(moves, symbol, current_pat):
    err_allowed = 0.1
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.24 + err_allowed]) * abs(BC)
    PRZ = round(current_pat[1] + 1.27*abs(XA),4)
    SL = PRZ*0.98
    TP1 = round(current_pat[3] + 0.618*abs(CD),4)
    TP2 = round(current_pat[3] + 0.5*abs(CD),4)
    TP3 = round(current_pat[3] + 0.382*abs(CD),4)
    butterfly_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= 1.2*abs(XA) and abs(CD) <= CD_range[1])

    if W_pat and butterfly_pat:

        return listToString(symbol).replace("/USDT", ""), PRZ, SL, TP1, TP2, TP3
    else:
        return ([])


def detect_harmonic(data, order=10):
    import numpy as np
    import pandas as pd
    import ccxt
    import time
    import dateutil
    from datetime import datetime
    from functools import reduce
    from scipy.signal import argrelextrema

    coins = data['Symbol'].unique().tolist()


    bull_bats = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bull_coin = bull_bat(moves, symbol, current_pat)

            if bull_coin != []:
                bull_bats.append(bull_coin[0])
                bull_bats.append(bull_coin[1])
                bull_bats.append(bull_coin[2])
                bull_bats.append(bull_coin[3])
                bull_bats.append(bull_coin[4])
                bull_bats.append(bull_coin[5])
        except:
            pass


    bear_bats = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bear_coin = bear_bat(moves, symbol, current_pat)

            if bear_coin != []:
                bear_bats.append(bear_coin[0])
                bear_bats.append(bear_coin[1])
                bear_bats.append(bear_coin[2])
                bear_bats.append(bear_coin[3])
                bear_bats.append(bear_coin[4])
                bear_bats.append(bear_coin[5])
        except:
            pass


    bull_gartleys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bull_coin = bull_gartley(moves, symbol, current_pat)

            if bull_coin != []:
                bull_gartleys.append(bull_coin[0])
                bull_gartleys.append(bull_coin[1])
                bull_gartleys.append(bull_coin[2])
                bull_gartleys.append(bull_coin[3])
                bull_gartleys.append(bull_coin[4])
                bull_gartleys.append(bull_coin[5])
        except:
            pass


    bear_gartleys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bear_coin = bear_gartley(moves, symbol, current_pat)

            if bear_coin != []:
                bear_gartleys.append(bear_coin[0])
                bear_gartleys.append(bear_coin[1])
                bear_gartleys.append(bear_coin[2])
                bear_gartleys.append(bear_coin[3])
                bear_gartleys.append(bear_coin[4])
                bear_gartleys.append(bear_coin[5])
        except:
            pass


    bull_crabs = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bull_coin = bull_crab(moves, symbol, current_pat)

            if bull_coin != []:
                bull_crabs.append(bull_coin[0])
                bull_crabs.append(bull_coin[1])
                bull_crabs.append(bull_coin[2])
                bull_crabs.append(bull_coin[3])
                bull_crabs.append(bull_coin[4])
                bull_crabs.append(bull_coin[5])
        except:
            pass


    bear_crabs = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bear_coin = bear_crab(moves, symbol, current_pat)

            if bear_coin != []:
                bear_crabs.append(bear_coin[0])
                bear_crabs.append(bear_coin[1])
                bear_crabs.append(bear_coin[2])
                bear_crabs.append(bear_coin[3])
                bear_crabs.append(bear_coin[4])
                bear_crabs.append(bear_coin[5])
        except:
            pass


    bull_butterflys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bull_coin = bull_butterfly(moves, symbol, current_pat)

            if bull_coin != []:
                bull_butterflys.append(bull_coin[0])
                bull_butterflys.append(bull_coin[1])
                bull_butterflys.append(bull_coin[2])
                bull_butterflys.append(bull_coin[3])
                bull_butterflys.append(bull_coin[4])
                bull_butterflys.append(bull_coin[5])
        except:
            pass


    bear_butterflys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data_new,
                                                                                                        order=order)
            bear_coin = bear_butterfly(moves, symbol, current_pat)

            if bear_coin != []:
                bear_butterflys.append(bear_coin[0])
                bear_butterflys.append(bear_coin[1])
                bear_butterflys.append(bear_coin[2])
                bear_butterflys.append(bear_coin[3])
                bear_butterflys.append(bear_coin[4])
                bear_butterflys.append(bear_coin[5])
        except:
            pass


    harmonic = pd.DataFrame({"bull_bats 看涨蝙蝠": ",".join(np.array(bull_bats)),
                                "bear_bats 看跌蝙蝠": ",".join(np.array(bear_bats)),
                                "bull_crabs 看涨螃蟹": ",".join(np.array(bull_crabs)),
                                "bear_crabs 看跌螃蟹": ",".join(np.array(bear_crabs)),
                                "bull_gartleys 看涨加特里": ",".join(np.array(bull_gartleys)),
                                "bear_gartleys 看跌加特里": ",".join(np.array(bear_gartleys)),
                                "bull_butterflys 看涨蝴蝶": ",".join(np.array(bull_butterflys)),
                                "bear_butterflys 看跌蝴蝶": ",".join(np.array(bear_butterflys))}, index=[0]).T
    harmonic.columns = ["coins"]
    return harmonic
