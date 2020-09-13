# %%
import numpy as np
import pandas as pd
import math
import baostock as bs
import plotly.graph_objects as go
import plotly.express as px

# %%
lg = bs.login()
stock_id = 'sh.600172'
rs = bs.query_history_k_data_plus(stock_id,
                                  "date, code, open, high, low, close, preclose, volume, amount, pctChg",
                                  start_date='2019-01-01', end_date='2020-08-07', frequency='d',
                                  adjustflag='2')

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

result = pd.DataFrame(data_list, columns=rs.fields)
print(result.tail())
result = result.set_index(pd.DatetimeIndex(result['date']))
result['open'] = result['open'].astype('float')
result['high'] = result['high'].astype('float')
result['low'] = result['low'].astype('float')
result['close'] = result['close'].astype('float')
result['volume'] = result['volume'].astype('float')

# %%
fig = go.Figure(data=[go.Candlestick(x=result.index,
                                     open=result['open'],
                                     high=result['high'],
                                     low=result['low'],
                                     close=result['close'],
                                     increasing_line_color='red',
                                     decreasing_line_color='green')])

fig.show()

# %%
low_list = result['low'].rolling(9, min_periods=1).min()
high_list = result['high'].rolling(9, min_periods=1).max()
rsv = (result['close'] - low_list) / (high_list - low_list) * 100
result['K'] = rsv.ewm(com=2).mean()
result['D'] = result['K'].ewm(com=2).mean()
result['J'] = 3 * result['K'] - 2 * result['D']

# %%
result.index.name = 'date'
result['KDJ_cross'] = ''
kdj_position = result['K'] > result['D']
result.loc[kdj_position[(kdj_position == True) &
                         (kdj_position.shift() == False)].index, 'KDJ_cross'] = 'gold'
result.loc[kdj_position[(kdj_position == False) &
                         (kdj_position.shift() == True)].index, 'KDJ_cross'] = 'death'
fig = px.line(result[['K', 'D', 'J']])
fig.show()

# %%
result['buy_sell'] = result['KDJ_cross'].shift(1).apply(lambda x: 'buy' if x == 'gold' else('sell' if x == 'death' else 'hold'))

# %%
import talib

# %%
def init(context):
    context.s1 = "000001.XSHE"

    # 使用MACD需要设置长短均线和macd平均线的参数
    context.SHORTPERIOD = 12
    context.LONGPERIOD = 26
    context.SMOOTHPERIOD = 9
    context.OBSERVATION = 100


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合状态信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！

    # 读取历史数据，使用sma方式计算均线准确度和数据长度无关，但是在使用ema方式计算均线时建议将历史数据窗口适当放大，结果会更加准确
    prices = history_bars(context.s1, context.OBSERVATION,'1d','close')

    # 用Talib计算MACD取值，得到三个时间序列数组，分别为macd, signal 和 hist
    macd, signal, hist = talib.MACD(prices, context.SHORTPERIOD,
                                    context.LONGPERIOD, context.SMOOTHPERIOD)

    plot("macd", macd[-1])
    plot("macd signal", signal[-1])

    # macd 是长短均线的差值，signal是macd的均线，使用macd策略有几种不同的方法，我们这里采用macd线突破signal线的判断方法

    # 如果macd从上往下跌破macd_signal

    if macd[-1] - signal[-1] < 0 and macd[-2] - signal[-2] > 0:
        # 获取当前投资组合中股票的仓位
        curPosition = get_position(context.s1).quantity
        #进行清仓
        if curPosition > 0:
            order_target_value(context.s1, 0)

    # 如果短均线从下往上突破长均线，为入场信号
    if macd[-1] - signal[-1] > 0 and macd[-2] - signal[-2] < 0:
        # 满仓入股
        order_target_percent(context.s1, 1)

# %%
